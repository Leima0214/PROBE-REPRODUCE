"""PROBE training entrypoint.

Two-phase pipeline:
  1. SPEM discovery: extract frozen ViT patch features from unlabeled target
     images, then run PCA + K-means to produce visual prototypes.
  2. Self-supervised pretraining: SimSiam + prompt consistency + DAPA MMD.

Usage:
    python scripts/train.py --config configs/probe_base.yaml --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import AdamW
from torch.utils.data import DataLoader

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(Path(_PROJECT_ROOT) / "src"))

import timm
import yaml

from probe.data.road_damage import RoadDamageDataset
from probe.engine.self_training import (
    DomainAlignmentHead,
    SimSiamHeads,
    probe_pretrain_step,
)
from probe.models import (
    LightweightDetectionHead,
    PROBEModel,
    PromptEnhancedViT,
    PromptProjector,
    PrototypeState,
    TargetPrototypeDiscovery,
)

try:
    from scripts.visualize import plot_dapa_alignment, plot_loss_curves, plot_spem_tsne
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False


# ---------------------------------------------------------------------------
# SimSiam-style augmentations (two-view)
# ---------------------------------------------------------------------------
def simsiam_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def eval_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def detection_collate(batch: list) -> tuple:
    """Collate variable-size detection targets (boxes/labels differ per image)."""
    images, targets = zip(*batch)
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, 0)
    return images, targets


# ---------------------------------------------------------------------------
# SPEM discovery phase
# ---------------------------------------------------------------------------
@torch.no_grad()
def discover_prototypes(
    dataset: RoadDamageDataset,
    vit_patch_embed: nn.Module,
    discovery: TargetPrototypeDiscovery,
    device: torch.device,
    image_size: int = 224,
    max_samples: int = 500,
) -> PrototypeState:
    """Extract patch features from unlabeled target images and fit SPEM."""
    indices = range(min(len(dataset), max_samples))
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    all_features = []
    for idx in indices:
        img, _ = dataset[idx]
        tensor = transform(img).unsqueeze(0).to(device)
        patch_tokens = vit_patch_embed(tensor)
        if patch_tokens.ndim == 4:
            patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        features = patch_tokens.squeeze(0)
        all_features.append(features)

    patch_features = torch.cat(all_features, dim=0)
    print(f"SPEM: collected {patch_features.shape[0]} patch tokens x {patch_features.shape[1]}d")
    state = discovery.fit(patch_features)
    print(f"SPEM: discovered {state.centroids.shape[0]} prototypes in {discovery.pca_dim}d PCA space")
    return state, patch_features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe_base.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--spem-samples", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Override pretrain_epochs from config")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    args = parser.parse_args()

    # Config ----------------------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Backbone --------------------------------------------------------------
    print("Loading ViT backbone...")
    vit = timm.create_model(cfg["backbone"]["name"], pretrained=True)
    vit.reset_classifier(0)  # discard classifier head

    # PROBE modules ---------------------------------------------------------
    discovery = TargetPrototypeDiscovery(
        pca_dim=cfg["spem"]["pca_dim"],
        num_prototypes=cfg["spem"]["num_prototypes"],
        kmeans_iters=cfg["spem"]["kmeans_iters"],
    )
    prompt_projector = PromptProjector(
        pca_dim=cfg["spem"]["pca_dim"],
        embed_dim=cfg["backbone"]["embed_dim"],
        hidden_dim=cfg["spem"]["prompt_hidden_dim"],
    )
    backbone = PromptEnhancedViT(
        vit,
        prompt_projector,
        injection_layers=tuple(cfg["spem"]["injection_layers"]),
    )
    detection_head = LightweightDetectionHead(
        embed_dim=cfg["backbone"]["embed_dim"],
        hidden_dim=cfg["detection"]["hidden_dim"],
        neck_dim=cfg["detection"]["neck_dim"],
        num_classes=cfg["detection"]["num_classes"],
    )
    model = PROBEModel(backbone, detection_head).to(device)

    ssl_heads = SimSiamHeads(
        embed_dim=cfg["backbone"]["embed_dim"],
        hidden_dim=cfg["ssl"]["hidden_dim"],
        out_dim=cfg["ssl"]["out_dim"],
    ).to(device)

    alignment_head = DomainAlignmentHead(
        embed_dim=cfg["backbone"]["embed_dim"],
        projection_dim=cfg["dapa"]["projection_dim"],
    ).to(device)

    # Phase 1: SPEM discovery -----------------------------------------------
    print("\n=== Phase 1: SPEM Prototype Discovery ===")
    target_dataset = RoadDamageDataset(
        cfg["data"]["target_manifest"],
        cfg["data"]["image_root"],
    )
    prototype_state, spem_patch_features = discover_prototypes(
        target_dataset,
        vit_patch_embed=vit.patch_embed,
        discovery=discovery,
        device=device,
        image_size=args.image_size,
        max_samples=args.spem_samples,
    )
    prototype_state = PrototypeState(
        mean=prototype_state.mean.to(device),
        components=prototype_state.components.to(device),
        centroids=prototype_state.centroids.to(device),
    )

    # --- VIZ: SPEM T-SNE ---------------------------------------------------
    enable_viz = not args.no_viz and _HAS_VIZ
    if not _HAS_VIZ and not args.no_viz:
        print("[viz] matplotlib/scikit-learn not available; install them for visualizations.")
    viz_dir = Path(args.checkpoint_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    if enable_viz:
        plot_spem_tsne(
            spem_patch_features.cpu(),
            prototype_state.centroids.cpu(),
            prototype_state.components.cpu(),
            prototype_state.mean.cpu(),
            viz_dir / "spem_tsne.png",
            max_patches=3000,
        )
    del spem_patch_features  # free CPU memory

    # Phase 2: SSL pretraining ----------------------------------------------
    print("\n=== Phase 2: Self-Supervised Pretraining ===")

    source_dataset = RoadDamageDataset(
        cfg["data"]["source_manifest"],
        cfg["data"]["image_root"],
        transform=eval_transform(args.image_size),
    )
    target_ssl_dataset = RoadDamageDataset(
        cfg["data"]["target_manifest"],
        cfg["data"]["image_root"],
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
        collate_fn=detection_collate,
    )
    # Target loader returns raw PIL images (transforms applied per-step)
    target_loader = DataLoader(
        target_ssl_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
        collate_fn=detection_collate,
    )

    trainable = list(ssl_heads.parameters()) + list(alignment_head.parameters())
    # Only the prompt projector inside the backbone is trainable
    trainable += list(prompt_projector.parameters())
    # Not training detection head or ViT

    optimizer = AdamW(
        trainable,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    # --- VIZ: DAPA alignment BEFORE training (baseline) --------------------
    if enable_viz:
        print("\n[viz] Capturing DAPA alignment baseline (before training)...")
        plot_dapa_alignment(
            model, alignment_head,
            source_dataset, target_ssl_dataset,
            prototype_state, device,
            viz_dir / "dapa_epoch000_before.png",
            num_samples=200, batch_size=cfg["data"]["batch_size"], image_size=args.image_size,
        )

    use_amp = device.type == "cuda"
    ssl_aug = simsiam_transform(args.image_size)
    history: dict[str, list[float]] = {"loss": [], "ssl": [], "prompt": [], "dapa": []}

    total_epochs = args.epochs if args.epochs is not None else cfg["optim"]["pretrain_epochs"]
    for epoch in range(total_epochs):
        ssl_heads.train()
        alignment_head.train()
        model.train()
        model.backbone.freeze_backbone()  # keep ViT in eval mode
        epoch_losses = {"loss": 0.0, "ssl": 0.0, "prompt": 0.0, "dapa": 0.0}

        for step, (source_batch, (target_imgs, _)) in enumerate(
            zip(source_loader, target_loader)
        ):
            source_images, _ = source_batch
            source_images = source_images.to(device)

            # Two augmented views of the same target batch
            target_view1 = torch.stack([ssl_aug(img) for img in target_imgs]).to(device)
            target_view2 = torch.stack([ssl_aug(img) for img in target_imgs]).to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    metrics = probe_pretrain_step(
                        model, ssl_heads, alignment_head,
                        source_images, target_view1, target_view2,
                        prototype_state, optimizer,
                        prompt_weight=cfg["spem"]["prompt_weight"],
                        dapa_weight=cfg["dapa"]["weight"],
                        prompt_temperature=cfg["spem"]["prompt_temperature"],
                    )
            else:
                metrics = probe_pretrain_step(
                    model, ssl_heads, alignment_head,
                    source_images, target_view1, target_view2,
                    prototype_state, optimizer,
                    prompt_weight=cfg["spem"]["prompt_weight"],
                    dapa_weight=cfg["dapa"]["weight"],
                    prompt_temperature=cfg["spem"]["prompt_temperature"],
                )

            for k in epoch_losses:
                epoch_losses[k] += metrics[k]

            if step % args.log_interval == 0:
                print(
                    f"  epoch {epoch:3d} step {step:4d} | "
                    f"loss {metrics['loss']:.4f}  ssl {metrics['ssl']:.4f}  "
                    f"prompt {metrics['prompt']:.4f}  dapa {metrics['dapa']:.4f}"
                )

        # Epoch summary
        n = step + 1
        avg = {k: epoch_losses[k] / n for k in epoch_losses}
        for k, v in avg.items():
            history[k].append(v)
        print(
            f"Epoch {epoch:3d} avg | "
            f"loss {avg['loss']:.4f}  ssl {avg['ssl']:.4f}  "
            f"prompt {avg['prompt']:.4f}  dapa {avg['dapa']:.4f}"
        )

        # Checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"probe_epoch{epoch+1:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ssl_heads": ssl_heads.state_dict(),
                    "alignment_head": alignment_head.state_dict(),
                    "prototype_state": prototype_state,
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = Path(args.checkpoint_dir) / "probe_final.pt"
    torch.save(
        {
            "epoch": cfg["optim"]["pretrain_epochs"],
            "model": model.state_dict(),
            "ssl_heads": ssl_heads.state_dict(),
            "alignment_head": alignment_head.state_dict(),
            "prototype_state": prototype_state,
        },
        final_path,
    )
    print(f"\nFinal checkpoint saved: {final_path}")
    print("Pretraining complete.")

    # --- VIZ: Loss curves + DAPA after training ---
    if enable_viz:
        plot_loss_curves(history, viz_dir / "loss_curves.png")
        print("[viz] Capturing DAPA alignment after training...")
        plot_dapa_alignment(
            model, alignment_head,
            source_dataset, target_ssl_dataset,
            prototype_state, device,
            viz_dir / "dapa_epoch_final_after.png",
            num_samples=200, batch_size=cfg["data"]["batch_size"], image_size=args.image_size,
        )
        print(f"All visualizations saved to {viz_dir}/")
    else:
        print("Pretraining complete. (visualizations skipped — pass without --no-viz to enable)")


if __name__ == "__main__":
    main()
