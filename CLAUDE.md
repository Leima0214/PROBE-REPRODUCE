# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (into conda env "probe")
pip install -r requirements.txt

# Convert RDD2022 Pascal VOC XML → PROBE JSONL manifests
python scripts/convert_rdd2022.py --xml-dir data/images --output-dir data

# Full pipeline: SPEM → SSL pretraining → detection head training
python scripts/train.py --config configs/probe_base.yaml --device cuda

# Phase 3 only (detection head training from pretrained checkpoint)
python scripts/train.py --config configs/probe_base.yaml --phase 3 \
    --resume checkpoints/probe_final.pt --device cuda

# Quick viz run (fewer epochs)
python scripts/train.py --config configs/probe_base.yaml --device cuda --epochs 5

# Single image inference
python scripts/infer.py --config configs/probe_base.yaml \
    --checkpoint checkpoints/probe_det_best.pt \
    --image data/images/China_MotorBike_000000.jpg --output result.png

# Batch evaluation (mAP on validation set)
python scripts/infer.py --config configs/probe_base.yaml \
    --checkpoint checkpoints/probe_det_best.pt --eval --device cuda
```

PyTorch multiprocessing on Windows needs `KMP_DUPLICATE_LIB_OK=TRUE` set in the environment.

## Architecture

This is the WACV 2026 PROBE method: self-supervised visual prompting for cross-domain road damage detection. The pipeline is three-phase:

1. **Phase 1: SPEM** — target-domain prototype discovery (PCA + K-means on frozen ViT patch features)
2. **Phase 2: SSL Pretraining** — SimSiam + prompt consistency (InfoNCE) + DAPA MMD²
3. **Phase 3: Detection Training** — FCOS-style dense detection head training on labeled source data

The key architectural insight is that the frozen ViT backbone is wrapped in `PromptEnhancedViT`, which injects learnable prompt tokens (derived from target prototypes via `PromptProjector`) at specific transformer layers. The ViT itself never gets gradients — only `PromptProjector`, `SimSiamHeads`, and `DomainAlignmentHead` are trainable during pretraining. In Phase 3, only `LightweightDetectionHead` is trained; the backbone stays frozen.

### Module dependency graph

```
scripts/train.py
  ├── probe.data.road_damage.RoadDamageDataset    ← JSONL → PIL images + bbox/label tensors
  ├── probe.models.prompts.TargetPrototypeDiscovery ← PCA + K-means → PrototypeState
  ├── probe.models.prompts.PromptProjector         ← MLP: PCA space → ViT embed space
  ├── probe.models.detector.PromptEnhancedViT      ← Frozen ViT + prompt injection wrapper
  ├── probe.models.detector.LightweightDetectionHead ← FCOS-style cls/box/ctr branches
  ├── probe.models.detector.PROBEModel             ← backbone + detection head
  ├── probe.engine.self_training.SimSiamHeads      ← projector + predictor MLPs
  ├── probe.engine.self_training.DomainAlignmentHead ← 2-layer MLP for MMD projection
  ├── probe.engine.self_training.probe_pretrain_step ← Phase 2: encode → 3 losses → backward
  └── probe.engine.detection                       ← Phase 3: FCOS losses, NMS, mAP
```

### Data flow

RDD2022 zip → `convert_rdd2022.py` → three JSONL manifests:

| Manifest | Content | Transform |
|---|---|---|
| `source_train.jsonl` | Labeled images (boxes + class ids) | `eval_transform` (resize + normalize) |
| `target_unlabeled.jsonl` | Unlabeled images for SPEM + SSL | None (PIL → per-batch `simsiam_transform`) |
| `val.jsonl` | Held-out labeled images | `eval_transform` |

`RoadDamageDataset` returns `(image, target_dict)` where `target_dict = {"boxes": [N,4], "labels": [N], "image_id": [1]}`. Boxes are `[x1,y1,x2,y2]` in pixel coordinates.

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

Phase 3: train_detection_head()
  source_loader → labeled images
  → model.encode(image) → patch_tokens
  → detection_head(patch_tokens) → {class_logits, boxes: [l,t,r,b], centerness}
  → detection_loss():
      encode_boxes(GT → [l*,t*,r*,b*] targets)
      sigmoid_focal_loss(cls_pred, cls_target)
      giou_loss(decoded_pred_boxes, GT_boxes)
      BCE_loss(centerness_pred, centerness_target)
  → backward + optimizer.step()
  → every N epochs: evaluate_map() on val set
```

### Detection head architecture (FCOS-style)

```
patch_tokens [B, 196, 768]
  → reshape → [B, 768, 14, 14]
  → stem: Conv3×3(768→384)-BN-GELU
  → ├─ cls_branch:  Conv1×1(384→128)-GELU-Conv1×1(128→5)   → class logits
  → ├─ box_branch:  Conv1×1(384→128)-GELU-Conv1×1(128→4)    → [l,t,r,b] / stride
  → └─ ctr_branch:  Conv1×1(384→128)-GELU-Conv1×1(128→1)    → centerness logit
```

Box encoding: `[l,t,r,b]` distances from grid centre to box edges, normalised by stride (16px).
Grid: 14×14 locations at stride=16, each cell centre at `(j*16+8, i*16+8)`.

### Key implementation details

- **Prompt injection**: `PromptInjector` concatenates prompt tokens before image tokens at layers {0, 6}, then removes them after the block. This matches the paper's Shallow+Mid design.
- **ViT stays frozen**: `PromptEnhancedViT.freeze_backbone()` sets `requires_grad=False` and `eval()` on the wrapped ViT. Must be re-called after `model.train()` in the training loop.
- **Detection head isn't trained during Phase 2**: `LightweightDetectionHead` is part of `PROBEModel` but its parameters are intentionally excluded from the Phase 2 optimizer. It is trained separately in Phase 3 on labeled source data.
- **Variable bbox counts**: different images have different numbers of boxes, so DataLoader needs `collate_fn=detection_collate` (returns `(stacked_images, tuple_of_target_dicts)`).
- **Phase 3 from checkpoint**: Use `--phase 3 --resume <ckpt>` to train only the detection head. Old checkpoints with monolithic `detection_head.head.*` keys are handled by filtering.
- **Box encoding is FCOS [l,t,r,b]**: The detection head predicts distances from the grid centre to each box boundary, normalised by stride. At decode time, predictions are denormalised and converted to [x1,y1,x2,y2].

### RDD2022 class mapping

```python
D00 → 0   # longitudinal crack
D10 → 1   # transverse crack
D20 → 2   # alligator crack
D40 → 3   # pothole
Repair → 4
```
