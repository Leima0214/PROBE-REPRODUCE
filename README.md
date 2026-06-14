# PROBE

Official PyTorch implementation scaffold for **PROBE: Self-Supervised Visual Prompting for Cross-Domain Road Damage Detection** (WACV 2026).

This repository contains the paper-aligned core architecture: **SPEM** for target-domain prototype prompts, **DAPA** for prompt-conditioned domain alignment, SimSiam-style self-supervised training heads, and a lightweight ViT-token detection head. Dataset download scripts, final checkpoints, and exact benchmark recipes will be released separately.

## Method Overview

```text
unlabeled target images
        |
frozen ViT patch embeddings
        |
PCA (D -> d0) + K-means
        |
target visual prototypes
        |
2-layer MLP prompt projector
        |
prompt tokens injected at shallow + mid ViT layers
        |
SimSiam SSL + prompt consistency + DAPA linear MMD
        |
lightweight detection head trained on limited source labels
```

## Core Components

- **SPEM**: discovers target visual prototypes with PCA + K-means, then maps them to prompt tokens through a shallow MLP.
- **Prompt injection**: inserts prompts at shallow and mid transformer layers, matching the paper's Shallow+Mid design.
- **Prompt consistency**: pulls each image feature toward its own prompt mean with an InfoNCE-style objective.
- **DAPA**: aligns prompt-conditioned source and target representations with a linear-kernel MMD loss.
- **Detection head**: reshapes final ViT patch tokens into a feature map and applies Conv-BN-GELU, Conv-GELU, and a 1x1 prediction layer.

## Repository Layout

```text
configs/probe_base.yaml           Minimal PROBE configuration
scripts/train.py                  Module initialization and training entrypoint
scripts/infer.py                  Inference entrypoint placeholder
src/probe/models/prompts.py       SPEM prototype discovery and prompt projection
src/probe/models/detector.py      Prompt-enhanced ViT wrapper and detection head
src/probe/engine/self_training.py SimSiam, prompt consistency, and DAPA step
src/probe/data/road_damage.py     Road-damage sample schema
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/probe_base.yaml
python scripts/infer.py --config configs/probe_base.yaml --checkpoint path/to/checkpoint.pt
```

The current release is intentionally a research scaffold. It documents the method interfaces and critical control flow without bundling private datasets or final training recipes.

## Citation

```bibtex
@inproceedings{xiao2026probe,
  title={Self-Supervised Visual Prompting for Cross-Domain Road Damage Detection},
  author={Xiao, Xi and Wang, Zhuxuanzi and Mo, Mingqiao and Liu, Chen and Ma, Chenrui and Li, Yanshu and Krishnaswamy, Smita and Wang, Xiao and Wang, Tianyang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={3514--3524},
  year={2026}
}
```
