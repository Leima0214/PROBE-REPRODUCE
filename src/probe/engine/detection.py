"""Detection training and inference utilities for PROBE Phase 3.

FCOS-style dense prediction with [l, t, r, b] box encoding, Focal Loss,
GIoU regression loss, centerness BCE, and NMS post-processing.

Grid layout
-----------
ViT-Base/16 on 224×224 images produces 14×14 patch tokens.  Each grid cell
corresponds to a 16×16 px region in the original image.  A location (row i,
col j) maps to image coordinates:

    x_ctr = j * stride + stride / 2    (stride = image_size / feature_size)
    y_ctr = i * stride + stride / 2

Box encoding
------------
For a grid centre (x_ctr, y_ctr) and a GT box [x1, y1, x2, y2]:

    l* = (x_ctr - x1) / stride       t* = (y_ctr - y1) / stride
    r* = (x2 - x_ctr) / stride       b* = (y2 - y_ctr) / stride

A location is *positive* iff the grid centre falls inside the GT box.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms as tv_batched_nms
from torchvision.ops import generalized_box_iou


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def generate_grid(
    feature_size: int,
    stride: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return [feature_size * feature_size, 2] grid of (x_ctr, y_ctr) in image px."""
    half = stride / 2.0
    shifts = torch.arange(0, feature_size, device=device, dtype=dtype) * stride + half
    shift_y, shift_x = torch.meshgrid(shifts, shifts, indexing="ij")
    locations = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=-1)
    return locations  # [H*W, 2]


# ---------------------------------------------------------------------------
# Box encoding / decoding
# ---------------------------------------------------------------------------

def encode_boxes(
    gt_boxes: torch.Tensor,       # [N, 4]  — [x1, y1, x2, y2]
    locations: torch.Tensor,      # [K, 2]  — (x_ctr, y_ctr)
    stride: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode GT boxes as [l, t, r, b] distance targets (normalised by stride).

    Returns
    -------
    targets : [K, 4]   — [l*, t*, r*, b*], 0 for negative locations
    mask :    [K] bool — True if location (x_ctr, y_ctr) falls inside *any* GT box
    assigned_idx : [K] int64 — index of assigned GT box, -1 for negatives
    """
    K = locations.shape[0]
    N = gt_boxes.shape[0]

    if N == 0:
        return (
            torch.zeros(K, 4, device=locations.device),
            torch.zeros(K, dtype=torch.bool, device=locations.device),
            torch.full((K,), -1, dtype=torch.int64, device=locations.device),
        )

    x_ctr = locations[:, 0]  # [K]
    y_ctr = locations[:, 1]  # [K]

    x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    # [K, N] — distance from each location to each box's four sides
    l = x_ctr[:, None] - x1[None, :]   # positive if centre is right of left edge
    t = y_ctr[:, None] - y1[None, :]
    r = x2[None, :] - x_ctr[:, None]
    b = y2[None, :] - y_ctr[:, None]

    # A location is inside a box if all four distances are > 0
    inside = (l > 0.0) & (t > 0.0) & (r > 0.0) & (b > 0.0)  # [K, N]

    # For locations inside multiple boxes, assign the one with the smallest area
    areas = (x2 - x1) * (y2 - y1)  # [N]
    inside_float = inside.float()  # [K, N]
    # Mask out non-inside with a huge area so they aren't selected if no box matches
    huge = areas.max() + 1.0
    masked_areas = inside_float * areas[None, :] + (1.0 - inside_float) * huge
    assigned_idx = masked_areas.argmin(dim=1)  # [K]
    mask = inside_float.sum(dim=1) > 0  # [K] — has at least one matching box

    # Invalidate assignments for negative locations
    assigned_idx[~mask] = -1

    # Gather the assigned box for each location
    idx_safe = assigned_idx.clamp(min=0)  # dummy index 0 for negatives
    assigned_boxes = gt_boxes[idx_safe]    # [K, 4]

    # Compute targets
    lt = torch.stack(
        [
            x_ctr - assigned_boxes[:, 0],
            y_ctr - assigned_boxes[:, 1],
            assigned_boxes[:, 2] - x_ctr,
            assigned_boxes[:, 3] - y_ctr,
        ],
        dim=-1,
    )  # [K, 4]
    targets = lt / stride  # normalise
    targets[~mask] = 0.0

    return targets, mask, assigned_idx


def decode_boxes(
    box_preds: torch.Tensor,    # [K, 4] — [l, t, r, b] normalised
    locations: torch.Tensor,    # [K, 2] — (x_ctr, y_ctr) in px
    stride: float,
    max_size: float = 224.0,
) -> torch.Tensor:
    """Decode normalised [l, t, r, b] predictions to [x1, y1, x2, y2] in px."""
    lt_rb = box_preds * stride  # denormalise
    x1 = locations[:, 0] - lt_rb[:, 0]
    y1 = locations[:, 1] - lt_rb[:, 1]
    x2 = locations[:, 0] + lt_rb[:, 2]
    y2 = locations[:, 1] + lt_rb[:, 3]
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    # Clamp to image bounds (optional but helps)
    boxes[:, 0].clamp_(min=0.0, max=max_size)
    boxes[:, 1].clamp_(min=0.0, max=max_size)
    boxes[:, 2].clamp_(min=0.0, max=max_size)
    boxes[:, 3].clamp_(min=0.0, max=max_size)
    return boxes


def compute_centerness_targets(
    targets: torch.Tensor,  # [K, 4] — [l*, t*, r*, b*] normalised
    mask: torch.Tensor,     # [K] bool
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute centerness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))."""
    lt = targets[:, :2]
    rb = targets[:, 2:]
    lr = torch.cat(
        [lt[:, 0:1], rb[:, 0:1]], dim=-1
    )  # [K, 2] — left, right
    tb = torch.cat(
        [lt[:, 1:2], rb[:, 1:2]], dim=-1
    )  # [K, 2] — top, bottom

    left_right = lr.min(dim=-1).values / (lr.max(dim=-1).values + eps)
    top_bottom = tb.min(dim=-1).values / (tb.max(dim=-1).values + eps)
    centerness = torch.sqrt(left_right * top_bottom)  # [K]
    centerness[~mask] = 0.0
    return centerness


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sigmoid focal loss for multi-label / one-hot classification.

    Args
    ----
    inputs:  [N, C]  logits
    targets: [N, C]  one-hot labels (all zeros = background / ignore)
    alpha:   class-balancing weight for positive class
    gamma:   focusing parameter
    """
    p = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        # Only average over positive locations to avoid background overwhelming
        num_pos = targets.sum()
        if num_pos > 0:
            return loss.sum() / num_pos
        return loss.sum() * 0.0  # zero gradient
    elif reduction == "sum":
        return loss.sum()
    return loss


def giou_loss(
    pred_boxes: torch.Tensor,  # [M, 4] decoded [x1,y1,x2,y2]
    gt_boxes: torch.Tensor,    # [M, 4]
) -> torch.Tensor:
    """Generalised IoU loss (1 - GIoU)."""
    giou = generalized_box_iou(pred_boxes, gt_boxes)  # [M]
    return (1.0 - giou).mean()


# ---------------------------------------------------------------------------
# Detection training step
# ---------------------------------------------------------------------------

def detection_loss(
    predictions: dict[str, torch.Tensor],
    targets: list[dict],
    locations: torch.Tensor,
    stride: float,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    box_weight: float = 1.0,
    ctr_weight: float = 1.0,
) -> dict[str, float]:
    """Compute detection losses for a batch.

    Parameters
    ----------
    predictions : dict with keys "class_logits" [B, C, H, W], "boxes" [B, 4, H, W],
                  "centerness" [B, 1, H, W].
    targets :     list of per-image dicts with "boxes" [N_i, 4] (pixel coords)
                  and "labels" [N_i] (0-indexed class ids).
    locations :   [H*W, 2] grid centres in pixel coords.
    stride :      feature-map stride (16 for ViT-B/16 @ 224).

    Returns
    -------
    metrics : dict with "det_cls", "det_box", "det_ctr", "det_total" float values.
    """
    cls_logits = predictions["class_logits"]   # [B, C, H, W]
    box_preds = predictions["boxes"]           # [B, 4, H, W]
    ctr_logits = predictions["centerness"]     # [B, 1, H, W]

    B, C, H, W = cls_logits.shape
    K = H * W  # number of locations

    device = cls_logits.device
    cls_losses = []
    box_losses = []
    ctr_losses = []
    total_pos = 0

    for b in range(B):
        gt_boxes = targets[b]["boxes"].to(device)      # [N_i, 4]
        gt_labels = targets[b]["labels"].to(device)    # [N_i]

        # Encode targets ----------------------------------------------------
        reg_target, pos_mask, assigned_idx = encode_boxes(
            gt_boxes, locations, stride
        )  # reg_target [K,4], pos_mask [K], assigned_idx [K]

        # Classification targets: one-hot at positive locations ---------------
        cls_target = torch.zeros(K, C, device=device)
        pos_idx = torch.where(pos_mask)[0]
        if pos_idx.numel() > 0:
            assigned_labels = gt_labels[assigned_idx[pos_idx]]
            cls_target[pos_idx, assigned_labels] = 1.0

        # Flatten predictions ------------------------------------------------
        cls_pred = cls_logits[b].permute(1, 2, 0).reshape(K, C)   # [K, C]
        box_pred = box_preds[b].permute(1, 2, 0).reshape(K, 4)    # [K, 4]
        ctr_pred = ctr_logits[b].permute(1, 2, 0).reshape(K)      # [K]

        # Classification loss ------------------------------------------------
        cls_loss = sigmoid_focal_loss(
            cls_pred, cls_target, alpha=focal_alpha, gamma=focal_gamma, reduction="mean"
        )
        cls_losses.append(cls_loss)

        # Regression & centerness (positive locations only) ------------------
        n_pos = pos_idx.numel()
        if n_pos > 0:
            total_pos += n_pos

            # Box loss (GIoU)
            pred_boxes_decoded = decode_boxes(
                box_pred[pos_idx], locations[pos_idx], stride
            )
            gt_boxes_assigned = gt_boxes[assigned_idx[pos_idx]]
            box_loss = giou_loss(pred_boxes_decoded, gt_boxes_assigned)
            box_losses.append(box_loss)

            # Centerness loss (BCE)
            ctr_target = compute_centerness_targets(reg_target, pos_mask)
            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr_pred[pos_idx], ctr_target[pos_idx], reduction="mean"
            )
            ctr_losses.append(ctr_loss)

    # Aggregate -------------------------------------------------------------
    loss_cls = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=device)
    loss_box = torch.stack(box_losses).mean() if box_losses else torch.tensor(0.0, device=device)
    loss_ctr = torch.stack(ctr_losses).mean() if ctr_losses else torch.tensor(0.0, device=device)

    total = loss_cls + box_weight * loss_box + ctr_weight * loss_ctr

    return {
        "det_cls": float(loss_cls.detach().cpu()),
        "det_box": float(loss_box.detach().cpu()),
        "det_ctr": float(loss_ctr.detach().cpu()),
        "det_total": float(total.detach().cpu()),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def collect_detections(
    predictions: dict[str, torch.Tensor],
    locations: torch.Tensor,
    stride: float,
    score_threshold: float = 0.05,
    max_detections: int = 196,
    image_size: float = 224.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode raw predictions into [x1,y1,x2,y2] boxes with scores and labels.

    Parameters
    ----------
    predictions : dict with "class_logits" [1, C, H, W], "boxes" [1, 4, H, W],
                  "centerness" [1, 1, H, W] (single image).
    locations :  [K, 2]
    stride :     float
    score_threshold : only keep detections above this score.
    max_detections : maximum number of detections to return.

    Returns
    -------
    boxes  : [D, 4]  — [x1, y1, x2, y2] pixel coords
    scores : [D]     — final score (cls_prob × centerness)
    labels : [D]     — long, class ids
    """
    cls_logits = predictions["class_logits"]  # [1, C, H, W]
    box_preds = predictions["boxes"]          # [1, 4, H, W]
    ctr_logits = predictions["centerness"]    # [1, 1, H, W]

    C = cls_logits.shape[1]
    K = locations.shape[0]

    # Flatten
    cls_logits = cls_logits[0].permute(1, 2, 0).reshape(K, C)   # [K, C]
    box_preds = box_preds[0].permute(1, 2, 0).reshape(K, 4)     # [K, 4]
    ctr_preds = ctr_logits[0].permute(1, 2, 0).reshape(K, 1)    # [K, 1]

    cls_probs = cls_logits.sigmoid()
    ctr_probs = ctr_preds.sigmoid()

    # Final score per location per class: √(cls × centerness)
    scores = cls_probs * ctr_probs           # [K, C]
    max_scores, max_labels = scores.max(dim=1)  # [K], [K]

    # Filter by score
    keep = max_scores > score_threshold
    if not keep.any():
        return (
            torch.zeros(0, 4, device=cls_logits.device),
            torch.zeros(0, device=cls_logits.device),
            torch.zeros(0, dtype=torch.long, device=cls_logits.device),
        )

    scores = max_scores[keep]
    labels = max_labels[keep]
    box_preds = box_preds[keep]
    locs = locations[keep]

    # Decode boxes
    boxes = decode_boxes(box_preds, locs, stride, max_size=image_size)

    # Keep top-k
    if scores.numel() > max_detections:
        topk = scores.topk(max_detections).indices
        boxes = boxes[topk]
        scores = scores[topk]
        labels = labels[topk]

    # Filter invalid boxes (x2 <= x1 or y2 <= y1)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if valid.any():
        boxes = boxes[valid]
        scores = scores[valid]
        labels = labels[valid]

    return boxes, scores, labels


def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class NMS, then keep top-k overall."""
    if boxes.numel() == 0:
        return boxes, scores, labels
    keep = tv_batched_nms(boxes, scores, labels, iou_threshold)
    if len(keep) > max_detections:
        keep = keep[scores[keep].topk(max_detections).indices]
    return boxes[keep], scores[keep], labels[keep]


# ---------------------------------------------------------------------------
# mAP computation (VOC-style 11-point interpolation)
# ---------------------------------------------------------------------------

def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """Compute IoU between two boxes (both [x1,y1,x2,y2])."""
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]).item() * (box1[3] - box1[1]).item()
    area2 = (box2[2] - box2[0]).item() * (box2[3] - box2[1]).item()
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_voc_ap(
    recalls: list[float],
    precisions: list[float],
) -> float:
    """11-point interpolated average precision (Pascal VOC 2012).

    Sorts by recall, then interpolates precision at 11 recall levels.
    """
    recalls = [0.0] + recalls + [1.0]
    precisions = [0.0] + precisions + [0.0]

    # Make precision monotonically decreasing from right
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Sample at 11 points: 0, 0.1, ..., 1.0
    ap = 0.0
    for t in torch.linspace(0, 1, 11).tolist():
        # Find highest precision at recall >= t
        p_max = 0.0
        for r, p in zip(recalls, precisions):
            if r >= t:
                p_max = max(p_max, p)
        ap += p_max / 11.0

    return ap


@torch.no_grad()
def evaluate_map(
    model,
    dataset,
    prototype_state,
    device: torch.device,
    locations: torch.Tensor,
    stride: float,
    num_classes: int = 5,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    image_size: int = 224,
    max_samples: Optional[int] = None,
) -> dict[str, float]:
    """Compute VOC-style mAP@0.5 on a labeled dataset.

    Parameters
    ----------
    model :        PROBEModel in eval mode.
    dataset :      RoadDamageDataset with labels.
    prototype_state : PrototypeState.
    device :       torch device.
    locations :    [H*W, 2] grid.
    stride :       float.
    num_classes :  number of damage classes.
    iou_threshold : IoU threshold for a correct detection.
    score_threshold : minimum score to consider a detection.
    nms_threshold :  IoU threshold for NMS.
    image_size :    input image size.
    max_samples :   cap on number of evaluation images (None = all).

    Returns
    -------
    metrics : dict with "mAP@0.5", and per-class "AP_cls_{c}" entries.
    """
    import torchvision.transforms as T

    model.eval()

    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Collect all ground truth and detections
    all_gt: dict[int, list[dict]] = {c: [] for c in range(num_classes)}
    all_det: dict[int, list[dict]] = {c: [] for c in range(num_classes)}
    # Per-class: list of dicts {image_id, confidence, box}

    indices = range(len(dataset))
    if max_samples is not None:
        indices = range(min(len(dataset), max_samples))

    for idx in indices:
        img, target = dataset[idx]
        gt_boxes = target["boxes"]   # [N, 4]
        gt_labels = target["labels"]  # [N]

        # Store GT per class
        for box, label in zip(gt_boxes, gt_labels):
            c = int(label.item())
            if c < num_classes:
                all_gt[c].append({"image_id": idx, "box": box, "matched": False})

        # Run detection
        tensor = transform(img).unsqueeze(0).to(device)
        predictions = model.detect(tensor, prototype_state)
        det_boxes, det_scores, det_labels = collect_detections(
            predictions, locations, stride,
            score_threshold=score_threshold, image_size=image_size,
        )
        det_boxes, det_scores, det_labels = apply_nms(
            det_boxes, det_scores, det_labels, iou_threshold=nms_threshold,
        )

        for box, score, label in zip(det_boxes, det_scores, det_labels):
            c = int(label.item())
            if c < num_classes:
                all_det[c].append({"image_id": idx, "confidence": score.item(), "box": box.cpu()})

    # Compute per-class AP
    aps = {}
    for c in range(num_classes):
        dets = all_det[c]
        gts = all_gt[c]

        # Sort detections by confidence descending
        dets.sort(key=lambda x: x["confidence"], reverse=True)

        # Reset matched flag
        for gt in gts:
            gt["matched"] = False

        tp = []
        fp = []
        total_gt = len(gts)

        for det in dets:
            # Find best-matching unmatched GT (same image, same class, highest IoU)
            best_iou = 0.0
            best_gt = None
            for gt in gts:
                if gt["image_id"] != det["image_id"]:
                    continue
                if gt["matched"]:
                    continue
                iou = compute_iou(det["box"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_iou >= iou_threshold and best_gt is not None:
                tp.append(1)
                fp.append(0)
                best_gt["matched"] = True
            else:
                tp.append(0)
                fp.append(1)

        if total_gt == 0 and len(dets) == 0:
            aps[f"AP_cls_{c}"] = 0.0
            continue

        # Cumulative precision / recall
        tp_cum = torch.tensor(tp).cumsum(dim=0).tolist() if tp else []
        fp_cum = torch.tensor(fp).cumsum(dim=0).tolist() if fp else []

        recalls = [t / max(total_gt, 1) for t in tp_cum]
        precisions = [
            tp_cum[i] / max(tp_cum[i] + fp_cum[i], 1)
            for i in range(len(tp_cum))
        ]

        aps[f"AP_cls_{c}"] = compute_voc_ap(recalls, precisions)

    aps["mAP@0.5"] = sum(aps[f"AP_cls_{c}"] for c in range(num_classes)) / num_classes
    return aps
