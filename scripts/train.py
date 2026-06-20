"""PROBE training entrypoint.

Three-phase pipeline:
  1. SPEM discovery: extract frozen ViT patch features from unlabeled target
     images, then run PCA + K-means to produce visual prototypes.
  2. Self-supervised pretraining: SimSiam + prompt consistency + DAPA MMD.
  3. Detection head training: FCOS-style dense prediction on labeled source data.

Usage:
    # Full pipeline (all three phases)
    python scripts/train.py --config configs/probe_base.yaml --device cuda

    # Phase 3 only (from pretrained checkpoint)
    python scripts/train.py --config configs/probe_base.yaml --phase 3 \\
        --resume checkpoints/probe_final.pt --device cuda
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
    compute_probe_losses,
)
from probe.engine.detection import (
    apply_nms,
    collect_detections,
    detection_loss,
    evaluate_map,
    generate_grid,
)
from probe.models import (
    LightweightDetectionHead,
    MoCoPromptConsistencyLoss,
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
) -> tuple[PrototypeState, torch.Tensor]:
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
# Phase 3: detection head training
# ---------------------------------------------------------------------------
def train_detection_head(
    model: PROBEModel,
    prototype_state: PrototypeState,
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Train the lightweight detection head on labeled source data (Phase 3).

    The backbone (PromptEnhancedViT + PromptProjector) stays frozen;
    only the LightweightDetectionHead is optimised.
    """
    print("\n=== Phase 3: Detection Head Training ===")

    image_size = args.image_size
    feature_size = image_size // 16  # ViT-Base/16 → 14×14 grid
    stride = float(image_size) / feature_size  # 16.0

    # Pre-compute the grid of (x_ctr, y_ctr) locations in image pixels
    locations = generate_grid(feature_size, stride, device)
    print(f"Detection grid: {feature_size}×{feature_size}, stride={stride:.1f}px")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    source_dataset = RoadDamageDataset(
        cfg["data"]["source_manifest"],
        cfg["data"]["image_root"],
        transform=eval_transform(image_size),
    )
    val_dataset = RoadDamageDataset(
        cfg["data"]["val_manifest"],
        cfg["data"]["image_root"],
        transform=eval_transform(image_size),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        drop_last=False,
        collate_fn=detection_collate,
    )

    # ------------------------------------------------------------------
    # Limited source labels (paper Table 2: 10 %, 50 %, 100 %)
    # ------------------------------------------------------------------
    label_frac = cfg["detection"].get("source_label_fraction", 1.0)
    if label_frac < 1.0:
        import random as _random
        _random.seed(cfg.get("seed", 42))
        n_full = len(source_dataset)
        n_keep = max(1, int(n_full * label_frac))
        keep_idx = sorted(_random.sample(range(n_full), n_keep))
        source_dataset = torch.utils.data.Subset(source_dataset, keep_idx)
        print(f"Source label fraction {label_frac}: using {n_keep}/{n_full} labeled images")

    source_loader = DataLoader(
        source_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
        collate_fn=detection_collate,
    )

    # ------------------------------------------------------------------
    # Optimiser — only detection head parameters
    # ------------------------------------------------------------------
    det_cfg = cfg.get("detection_optim", {})
    optimizer = AdamW(
        model.detection_head.parameters(),
        lr=det_cfg.get("lr", 1e-4),
        weight_decay=det_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=det_cfg.get("epochs", 50),
        eta_min=det_cfg.get("lr_min", 1e-6),
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    total_epochs = getattr(args, "det_epochs", None) or det_cfg.get("epochs", 50)
    focal_alpha = det_cfg.get("focal_alpha", 0.25)
    focal_gamma = det_cfg.get("focal_gamma", 2.0)
    box_weight = det_cfg.get("box_weight", 1.0)
    ctr_weight = det_cfg.get("ctr_weight", 1.0)
    score_threshold = det_cfg.get("score_threshold", 0.05)
    nms_threshold = det_cfg.get("nms_threshold", 0.5)
    val_interval = det_cfg.get("val_interval", 5)
    use_amp = device.type == "cuda"
    grad_accum = cfg["data"].get("gradient_accumulation", 1)

    best_map = 0.0
    best_epoch = -1
    checkpoint_dir = Path(args.checkpoint_dir)

    # Freeze backbone — only detection head is trainable
    model.backbone.freeze_backbone()
    for param in model.backbone.parameters():
        param.requires_grad = False

    history_det: dict[str, list[float]] = {
        "cls": [], "box": [], "ctr": [], "total": [], "mAP": [],
    }

    for epoch in range(total_epochs):
        model.detection_head.train()
        epoch_losses = {"cls": 0.0, "box": 0.0, "ctr": 0.0, "total": 0.0}
        steps = 0

        optimizer.zero_grad(set_to_none=True)
        for images, targets in source_loader:
            images = images.to(device)

            # Forward — backbone is frozen
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, patch_tokens, _ = model.encode(images, prototype_state)
                    predictions = model.detection_head(patch_tokens)
                    loss_dict = detection_loss(
                        predictions, targets, locations, stride,
                        focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                        box_weight=box_weight, ctr_weight=ctr_weight,
                    )
            else:
                _, patch_tokens, _ = model.encode(images, prototype_state)
                predictions = model.detection_head(patch_tokens)
                loss_dict = detection_loss(
                    predictions, targets, locations, stride,
                    focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                    box_weight=box_weight, ctr_weight=ctr_weight,
                )

            (loss_dict["det_total"] / grad_accum).backward()

            if (steps + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.detection_head.parameters(), max_norm=10.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[f"det_{k}"].item()

            steps += 1
            if steps % args.log_interval == 0:
                print(
                    f"  epoch {epoch:3d} step {steps:4d} | "
                    f"cls {loss_dict['det_cls'].item():.4f}  box {loss_dict['det_box'].item():.4f}  "
                    f"ctr {loss_dict['det_ctr'].item():.4f}  total {loss_dict['det_total'].item():.4f}"
                )

        scheduler.step()

        # Epoch summary
        avg = {k: epoch_losses[k] / max(steps, 1) for k in epoch_losses}
        for k, v in avg.items():
            history_det[k].append(v)
        print(
            f"Epoch {epoch:3d} avg | "
            f"cls {avg['cls']:.4f}  box {avg['box']:.4f}  "
            f"ctr {avg['ctr']:.4f}  total {avg['total']:.4f}  "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # Validation
        if (epoch + 1) % val_interval == 0 or epoch == total_epochs - 1:
            print(f"  Validating mAP@0.5 ...")
            metrics = evaluate_map(
                model,
                val_dataset,
                prototype_state,
                device,
                locations,
                stride,
                num_classes=cfg["detection"]["num_classes"],
                iou_threshold=0.5,
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                image_size=image_size,
            )
            mAP = metrics["mAP@0.5"]
            history_det["mAP"].append(mAP)
            class_aps = {k: v for k, v in metrics.items() if k.startswith("AP_cls_")}
            ap_str = "  ".join(f"c{c.split('_')[-1]}={v:.3f}" for c, v in sorted(class_aps.items()))
            print(f"  Val mAP@0.5: {mAP:.4f}  |  {ap_str}")

            if mAP > best_map:
                best_map = mAP
                best_epoch = epoch
                best_path = checkpoint_dir / "probe_det_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "detection_head": model.detection_head.state_dict(),
                        "prototype_state": prototype_state,
                        "optimizer": optimizer.state_dict(),
                        "mAP": mAP,
                        "class_aps": class_aps,
                    },
                    best_path,
                )
                print(f"  Best model saved (mAP={best_map:.4f}) -> {best_path}")

    # Final checkpoint
    final_path = checkpoint_dir / "probe_det_final.pt"
    torch.save(
        {
            "epoch": total_epochs - 1,
            "model": model.state_dict(),
            "detection_head": model.detection_head.state_dict(),
            "prototype_state": prototype_state,
            "best_mAP": best_map,
            "best_epoch": best_epoch,
        },
        final_path,
    )
    print(f"\nPhase 3 complete.  Best mAP@0.5: {best_map:.4f} (epoch {best_epoch})")
    print(f"Final checkpoint saved: {final_path}")


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
    parser.add_argument("--det-epochs", type=int, default=None, help="Override detection training epochs")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size from config")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run only a specific phase (3 = detection training only, requires --resume)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a Phase 2 checkpoint (required for --phase 3)",
    )
    args = parser.parse_args()

    # Config ----------------------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    # CLI overrides
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    print("Loading ViT backbone...")
    vit = vit = timm.create_model(cfg["backbone"]["name"], pretrained=True, img_size=args.image_size)
    vit.reset_classifier(0)

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

    # ------------------------------------------------------------------
    # Phase 3 only: load pretrained backbone and train detection head
    # ------------------------------------------------------------------
    if args.phase == 3:
        if args.resume is None:
            raise ValueError("--resume <checkpoint> is required for --phase 3")
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        print(f"Loading pretrained backbone from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        # Filter out old detection head keys for backward compatibility
        model_state = ckpt["model"]
        filtered_state = {}
        for k, v in model_state.items():
            if k.startswith("detection_head."):
                continue  # skip — new head has different architecture
            filtered_state[k] = v
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"  Loaded backbone weights (missing: {len(missing)}, unexpected: {len(unexpected)})")

        prototype_state = ckpt.get("prototype_state")
        if prototype_state is None:
            raise ValueError("Checkpoint does not contain prototype_state")
        if isinstance(prototype_state, PrototypeState):
            prototype_state = PrototypeState(
                mean=prototype_state.mean.to(device),
                components=prototype_state.components.to(device),
                centroids=prototype_state.centroids.to(device),
            )
        else:
            # Legacy loading — prototype_state was saved as dict-like
            prototype_state = PrototypeState(
                mean=prototype_state["mean"].to(device),
                components=prototype_state["components"].to(device),
                centroids=prototype_state["centroids"].to(device),
            )

        train_detection_head(model, prototype_state, cfg, args, device)
        return

    # ======================================================================
    # Full pipeline: Phase 1 → Phase 2 → Phase 3
    # ======================================================================

    # ------------------------------------------------------------------
    # Phase 1: SPEM discovery
    # ------------------------------------------------------------------
    ssl_heads = SimSiamHeads(
        embed_dim=cfg["backbone"]["embed_dim"],
        hidden_dim=cfg["ssl"]["hidden_dim"],
        out_dim=cfg["ssl"]["out_dim"],
    ).to(device)

    alignment_head = DomainAlignmentHead(
        embed_dim=cfg["backbone"]["embed_dim"],
        projection_dim=cfg["dapa"]["projection_dim"],
    ).to(device)

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

    # VIZ: SPEM T-SNE
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
    del spem_patch_features

    # ------------------------------------------------------------------
    # Phase 2: SSL pretraining
    # ------------------------------------------------------------------
    print("\n=== Phase 2: Self-Supervised Pretraining ===")

    # --- MoCo prompt-consistency module (optional) ------------------------
    moco_loss: MoCoPromptConsistencyLoss | None = None
    if cfg.get("moco", {}).get("enabled", False):
        moco_loss = MoCoPromptConsistencyLoss(
            prompt_projector,
            queue_size=cfg["moco"].get("queue_size", 4096),
            temperature=cfg["moco"].get("temperature", 0.2),
            momentum=cfg["moco"].get("momentum", 0.999),
        ).to(device)
        print(f"MoCo enabled: queue={moco_loss.queue_size}, "
              f"T={moco_loss.temperature}, m={moco_loss.momentum}")
    # ------------------------------------------------------------------
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
    target_loader = DataLoader(
        target_ssl_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
        collate_fn=detection_collate,
    )

    trainable = list(ssl_heads.parameters()) + list(alignment_head.parameters())
    trainable += list(prompt_projector.parameters())

    optimizer = AdamW(
        trainable,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    # DAPA alignment baseline (before training)
    if enable_viz:
        print("\n[viz] Capturing DAPA alignment baseline (before training)...")
        plot_dapa_alignment(
            model,
            alignment_head,
            source_dataset,
            target_ssl_dataset,
            prototype_state,
            device,
            viz_dir / "dapa_epoch000_before.png",
            num_samples=200,
            batch_size=cfg["data"]["batch_size"],
            image_size=args.image_size,
        )

    use_amp = device.type == "cuda"
    ssl_aug = simsiam_transform(args.image_size)
    history: dict[str, list[float]] = {"loss": [], "ssl": [], "prompt": [], "dapa": []}

    grad_accum = cfg["data"].get("gradient_accumulation", 1)
    print(f"Phase 2: batch_size={cfg['data']['batch_size']}, grad_accum={grad_accum}, "
          f"effective_batch={cfg['data']['batch_size'] * grad_accum}")

    total_epochs = args.epochs if args.epochs is not None else cfg["optim"]["pretrain_epochs"]
    for epoch in range(total_epochs):
        ssl_heads.train()
        alignment_head.train()
        model.train()
        model.backbone.freeze_backbone()
        epoch_losses = {"loss": 0.0, "ssl": 0.0, "prompt": 0.0, "dapa": 0.0}

        optimizer.zero_grad(set_to_none=True)
        for step, (source_batch, (target_imgs, _)) in enumerate(
            zip(source_loader, target_loader)
        ):
            source_images, _ = source_batch
            source_images = source_images.to(device)

            target_view1 = torch.stack([ssl_aug(img) for img in target_imgs]).to(device)
            target_view2 = torch.stack([ssl_aug(img) for img in target_imgs]).to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss, metrics = compute_probe_losses(
                        model,
                        ssl_heads,
                        alignment_head,
                        source_images,
                        target_view1,
                        target_view2,
                        prototype_state,
                        prompt_weight=cfg["spem"]["prompt_weight"],
                        dapa_weight=cfg["dapa"]["weight"],
                        prompt_temperature=cfg["spem"]["prompt_temperature"],
                        moco_loss=moco_loss,
                        prototypes=prototype_state.centroids,
                    )
            else:
                loss, metrics = compute_probe_losses(
                    model,
                    ssl_heads,
                    alignment_head,
                    source_images,
                    target_view1,
                    target_view2,
                    prototype_state,
                    prompt_weight=cfg["spem"]["prompt_weight"],
                    dapa_weight=cfg["dapa"]["weight"],
                    prompt_temperature=cfg["spem"]["prompt_temperature"],
                    moco_loss=moco_loss,
                    prototypes=prototype_state.centroids,
                )

            (loss / grad_accum).backward()

            for k in epoch_losses:
                epoch_losses[k] += metrics[k]

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                print(
                    f"  epoch {epoch:3d} step {step:4d} | "
                    f"loss {metrics['loss']:.4f}  ssl {metrics['ssl']:.4f}  "
                    f"prompt {metrics['prompt']:.4f}  dapa {metrics['dapa']:.4f}"
                )

        n = step + 1
        avg = {k: epoch_losses[k] / n for k in epoch_losses}
        for k, v in avg.items():
            history[k].append(v)
        print(
            f"Epoch {epoch:3d} avg | "
            f"loss {avg['loss']:.4f}  ssl {avg['ssl']:.4f}  "
            f"prompt {avg['prompt']:.4f}  dapa {avg['dapa']:.4f}"
        )

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

    # Final Phase 2 checkpoint
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

    # VIZ after Phase 2
    if enable_viz:
        plot_loss_curves(history, viz_dir / "loss_curves.png")
        print("[viz] Capturing DAPA alignment after training...")
        plot_dapa_alignment(
            model,
            alignment_head,
            source_dataset,
            target_ssl_dataset,
            prototype_state,
            device,
            viz_dir / "dapa_epoch_final_after.png",
            num_samples=200,
            batch_size=cfg["data"]["batch_size"],
            image_size=args.image_size,
        )
        print(f"All visualizations saved to {viz_dir}/")
    else:
        print("Pretraining complete. (visualizations skipped)")

    # ------------------------------------------------------------------
    # Phase 3: Detection head training
    # ------------------------------------------------------------------
    # Clean up SSL heads to free GPU memory
    del ssl_heads, alignment_head, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_detection_head(model, prototype_state, cfg, args, device)


if __name__ == "__main__":
    main()
