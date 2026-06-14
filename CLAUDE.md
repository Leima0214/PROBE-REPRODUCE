# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (into conda env "probe")
pip install -r requirements.txt timm matplotlib scikit-learn

# Convert RDD2022 Pascal VOC XML → PROBE JSONL manifests
python scripts/convert_rdd2022.py --xml-dir data/images --output-dir data

# Train (CPU fallback automatic)
python scripts/train.py --config configs/probe_base.yaml --device cuda

# Quick viz run (fewer epochs, skip viz with --no-viz)
python scripts/train.py --config configs/probe_base.yaml --device cuda --epochs 5
```

PyTorch multiprocessing on Windows needs `KMP_DUPLICATE_LIB_OK=TRUE` set in the environment.

## Architecture

This is the WACV 2026 PROBE method: self-supervised visual prompting for cross-domain road damage detection. The paper is two-phase: **(1) target-domain prototype discovery (SPEM)** followed by **(2) SSL pretraining** with SimSiam + prompt consistency + DAPA MMD. A lightweight detection head is then trained on limited source labels (not yet implemented in the scaffold).

The key architectural insight is that the frozen ViT backbone is wrapped in `PromptEnhancedViT`, which injects learnable prompt tokens (derived from target prototypes via `PromptProjector`) at specific transformer layers. The ViT itself never gets gradients — only `PromptProjector`, `SimSiamHeads`, and `DomainAlignmentHead` are trainable during pretraining.

### Module dependency graph

```
scripts/train.py
  ├── probe.data.road_damage.RoadDamageDataset    ← JSONL → PIL images + bbox/label tensors
  ├── probe.models.prompts.TargetPrototypeDiscovery ← PCA + K-means → PrototypeState
  ├── probe.models.prompts.PromptProjector         ← MLP: PCA space → ViT embed space
  ├── probe.models.detector.PromptEnhancedViT      ← Frozen ViT + prompt injection wrapper
  ├── probe.models.detector.PROBEModel             ← backbone + detection head
  ├── probe.engine.self_training.SimSiamHeads      ← projector + predictor MLPs
  ├── probe.engine.self_training.DomainAlignmentHead ← 2-layer MLP for MMD projection
  └── probe.engine.self_training.probe_pretrain_step ← single-step: encode → 3 losses → backward
```

### Data flow

RDD2022 zip → `convert_rdd2022.py` → three JSONL manifests:

| Manifest | Content | Transform |
|---|---|---|
| `source_train.jsonl` | Labeled images (boxes + class ids) | `eval_transform` (resize + normalize) |
| `target_unlabeled.jsonl` | Unlabeled images for SPEM + SSL | None (PIL → per-batch `simsiam_transform`) |
| `val.jsonl` | Held-out labeled images | `eval_transform` |

`RoadDamageDataset` returns `(image, target_dict)` where `target_dict = {"boxes": [N,4], "labels": [N], "image_id": [1]}`. Boxes are `[x1,y1,x2,y2]` in pixel coordinates. The dataset can be used with or without labels — unlabeled samples have empty `boxes`/`labels` lists.

### Training pipeline (train.py)

```
Phase 1: discover_prototypes()
  unlabeled images → frozen ViT.patch_embed → PCA(768→50) → K-means(10 centroids) → PrototypeState

Phase 2: per-epoch loop
  zip(source_loader, target_loader) → source images + two SimSiam views of target
  → probe_pretrain_step():
      model.encode(source)     → [CLS] features
      model.encode(target_v1)  → [CLS] features + prompt tokens
      model.encode(target_v2)  → [CLS] features
      loss = simsiam_loss(v1, v2) + prompt_weight * InfoNCE(feats, prompts) + dapa_weight * MMD²(source, target)
      backward + optimizer.step()
```

### Key implementation details

- **Prompt injection**: `PromptInjector` concatenates prompt tokens before image tokens at layers {0, 6}, then removes them after the block. This matches the paper's Shallow+Mid design.
- **ViT stays frozen**: `PromptEnhancedViT.freeze_backbone()` sets `requires_grad=False` and `eval()` on the wrapped ViT. Must be re-called after `model.train()` in the training loop.
- **Detection head isn't trained during pretraining**: `LightweightDetectionHead` is part of `PROBEModel` but its parameters are intentionally excluded from the optimizer. The paper trains it separately on labeled source data after pretraining.
- **Variable bbox counts**: different images have different numbers of boxes, so DataLoader needs `collate_fn=detection_collate` (returns `(stacked_images, tuple_of_target_dicts)`).

### Visualization (scripts/visualize.py)

Three plot types, auto-generated during `train.py` runs (disable with `--no-viz`):

1. **SPEM T-SNE** (`checkpoints/viz/spem_tsne.png`) — PCA-space patch embeddings colored by nearest prototype, with prototypes as red stars
2. **DAPA alignment** (before/after) — source vs target CLS features after DAPA projection, T-SNE colored by domain. Good alignment = points intermix.
3. **Loss curves** (`checkpoints/viz/loss_curves.png`) — 2×2 grid of total/ssl/prompt/dapa losses per epoch

### GPU memory constraints

ViT-Base with batch=32 needs ~15-18GB VRAM. Training uses the full batch size from config. On 6GB cards, reduce `batch_size` in the config and enable AMP autocast.

### RDD2022 class mapping

```python
D00 → 0   # longitudinal crack
D10 → 1   # transverse crack
D20 → 2   # alligator crack
D40 → 3   # pothole
Repair → 4
```
