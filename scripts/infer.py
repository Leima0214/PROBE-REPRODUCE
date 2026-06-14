"""PROBE inference and evaluation script.

Usage:
    # Single image inference with visualisation
    python scripts/infer.py --config configs/probe_base.yaml \\
        --checkpoint checkpoints/probe_det_best.pt \\
        --image data/images/China_MotorBike_000000.jpg --output result.png

    # Batch evaluation on validation set
    python scripts/infer.py --config configs/probe_base.yaml \\
        --checkpoint checkpoints/probe_det_best.pt --eval --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(Path(_PROJECT_ROOT) / "src"))

import timm
import yaml
from PIL import Image, ImageDraw, ImageFont

from probe.data.road_damage import RoadDamageDataset
from probe.engine.detection import (
    apply_nms,
    collect_detections,
    evaluate_map,
    generate_grid,
)
from probe.models import (
    LightweightDetectionHead,
    PROBEModel,
    PromptEnhancedViT,
    PromptProjector,
    PrototypeState,
    TargetPrototypeDiscovery,
)

# RDD2022 class names
CLASS_NAMES = {
    0: "D00: Longitudinal Crack",
    1: "D10: Transverse Crack",
    2: "D20: Alligator Crack",
    3: "D40: Pothole",
    4: "Repair",
}

CLASS_COLORS = {
    0: (220, 50, 50),    # red
    1: (50, 50, 220),    # blue
    2: (50, 180, 50),    # green
    3: (220, 180, 50),   # orange
    4: (180, 50, 220),   # purple
}


def load_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[PROBEModel, PrototypeState, float, int]:
    """Load PROBE model and prototype state from a detection checkpoint."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Build backbone
    vit = timm.create_model(cfg["backbone"]["name"], pretrained=False)
    vit.reset_classifier(0)

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

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both Phase 2 and Phase 3 checkpoints
    model_state = ckpt.get("model", ckpt)
    if hasattr(model_state, "keys"):
        filtered = {k: v for k, v in model_state.items() if not k.startswith("detection_head.head.")}
        model.load_state_dict(filtered, strict=False)
    else:
        model.load_state_dict(model_state, strict=False)

    # Load prototype state
    ps = ckpt.get("prototype_state")
    if ps is None:
        raise ValueError("Checkpoint missing prototype_state")

    if isinstance(ps, PrototypeState):
        prototype_state = PrototypeState(
            mean=ps.mean.to(device),
            components=ps.components.to(device),
            centroids=ps.centroids.to(device),
        )
    elif isinstance(ps, dict):
        prototype_state = PrototypeState(
            mean=torch.as_tensor(ps["mean"]).to(device),
            components=torch.as_tensor(ps["components"]).to(device),
            centroids=torch.as_tensor(ps["centroids"]).to(device),
        )
    else:
        prototype_state = PrototypeState(
            mean=ps.mean.to(device),
            components=ps.components.to(device),
            centroids=ps.centroids.to(device),
        )

    mAP = ckpt.get("mAP", ckpt.get("best_mAP", 0.0))
    epoch = ckpt.get("epoch", -1)

    model.eval()
    return model, prototype_state, float(mAP), int(epoch), cfg


def draw_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    score_threshold: float = 0.3,
) -> Image.Image:
    """Draw predicted bounding boxes on the image."""
    draw = ImageDraw.Draw(image)
    # Try to get a font — fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        s = score.item()
        if s < score_threshold:
            continue
        c = int(label.item())
        x1, y1, x2, y2 = box.tolist()
        color = CLASS_COLORS.get(c, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = CLASS_NAMES.get(c, f"cls_{c}")
        text = f"{name} {s:.2f}"
        # White background for text
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_bbox, fill=(0, 0, 0, 180))
        draw.text((x1, max(y1 - 14, 0)), text, fill=color, font=font)
    return image


def infer_single(
    model: PROBEModel,
    prototype_state: PrototypeState,
    image_path: str,
    device: torch.device,
    image_size: int = 224,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run detection on a single image.  Returns (pil_image, boxes, scores, labels)."""
    feature_size = image_size // 16
    stride = float(image_size) / feature_size
    locations = generate_grid(feature_size, stride, device)

    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pil_image = Image.open(image_path).convert("RGB")
    tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model.detect(tensor, prototype_state)
        det_boxes, det_scores, det_labels = collect_detections(
            predictions,
            locations,
            stride,
            score_threshold=score_threshold,
            image_size=image_size,
        )
        det_boxes, det_scores, det_labels = apply_nms(
            det_boxes, det_scores, det_labels, iou_threshold=nms_threshold,
        )

    return pil_image, det_boxes.cpu(), det_scores.cpu(), det_labels.cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe_base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image", default=None, help="Path to a single image for inference")
    parser.add_argument("--output", default="output.png", help="Output path for visualised result")
    parser.add_argument("--eval", action="store_true", help="Run mAP evaluation on val set")
    parser.add_argument("--score-threshold", type=float, default=0.3,
                        help="Score threshold for visualisation")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-eval", type=int, default=None,
                        help="Cap on evaluation images (default: all)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, prototype_state, ckpt_map, ckpt_epoch, cfg = load_model(
        args.config, args.checkpoint, device
    )
    print(f"Checkpoint epoch: {ckpt_epoch}, stored mAP: {ckpt_map:.4f}")

    # ------------------------------------------------------------------
    # Evaluation mode
    # ------------------------------------------------------------------
    if args.eval:
        feature_size = args.image_size // 16
        stride = float(args.image_size) / feature_size
        locations = generate_grid(feature_size, stride, device)

        det_cfg = cfg.get("detection_optim", {})
        val_dataset = RoadDamageDataset(
            cfg["data"]["val_manifest"],
            cfg["data"]["image_root"],
        )
        print(f"\nEvaluating mAP@0.5 on {len(val_dataset)} validation samples...")
        metrics = evaluate_map(
            model,
            val_dataset,
            prototype_state,
            device,
            locations,
            stride,
            num_classes=cfg["detection"]["num_classes"],
            iou_threshold=0.5,
            score_threshold=det_cfg.get("score_threshold", 0.05),
            nms_threshold=det_cfg.get("nms_threshold", 0.5),
            image_size=args.image_size,
            max_samples=args.max_eval,
        )
        print(f"\n{'='*50}")
        print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")
        for c in range(cfg["detection"]["num_classes"]):
            print(f"  {CLASS_NAMES.get(c, f'class_{c}')}: {metrics.get(f'AP_cls_{c}', 0.0):.4f}")
        print(f"{'='*50}")
        return

    # ------------------------------------------------------------------
    # Single image inference
    # ------------------------------------------------------------------
    if args.image is None:
        print("Please provide --image for single inference or --eval for evaluation.")
        return

    pil_image, boxes, scores, labels = infer_single(
        model,
        prototype_state,
        args.image,
        device,
        image_size=args.image_size,
        score_threshold=0.05,  # low threshold for collect; visualisation uses --score-threshold
        nms_threshold=args.nms_threshold,
    )

    print(f"\nDetections: {len(boxes)} boxes found")
    for box, score, label in zip(boxes, scores, labels):
        c = int(label.item())
        name = CLASS_NAMES.get(c, f"class_{c}")
        print(f"  {name}: score={score.item():.3f}, box={box.tolist()}")

    result = draw_boxes(pil_image, boxes, scores, labels, score_threshold=args.score_threshold)
    result.save(args.output)
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
