from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .prompts import PromptInjector, PromptProjector, PrototypeState


@dataclass
class DetectionBatch:
    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor | None = None


class PromptEnhancedViT(nn.Module):
    """Frozen ViT wrapper with SPEM prompt injection.

    The wrapped ViT is expected to expose standard ViT components:
    ``patch_embed``, ``cls_token``, ``pos_embed``, ``blocks`` and ``norm``.
    This keeps the method code independent of a specific timm/torchvision
    backbone while documenting the exact insertion points used by PROBE.
    """

    def __init__(
        self,
        vit: nn.Module,
        prompt_projector: PromptProjector,
        injection_layers: tuple[int, ...] = (0, 6),
    ) -> None:
        super().__init__()
        self.vit = vit
        self.prompt_projector = prompt_projector
        self.injector = PromptInjector(injection_layers)
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.vit.parameters():
            parameter.requires_grad = False
        self.vit.eval()

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.vit.patch_embed(images)
        if patch_tokens.ndim == 4:
            patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        cls = self.vit.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=1)
        return tokens + self.vit.pos_embed[:, : tokens.shape[1]]

    def forward_tokens(
        self,
        images: torch.Tensor,
        prototype_state: PrototypeState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patchify(images)
        prompts = self.prompt_projector(
            prototype_state.centroids.to(tokens.device),
            batch_size=images.shape[0],
        )

        for layer_id, block in enumerate(self.vit.blocks):
            if self.injector.should_inject(layer_id):
                tokens = self.injector.insert(tokens, prompts)
                tokens = block(tokens)
                tokens = self.injector.remove(tokens, prompts.shape[1])
            else:
                tokens = block(tokens)

        return self.vit.norm(tokens), prompts

    def forward(
        self,
        images: torch.Tensor,
        prototype_state: PrototypeState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, prompts = self.forward_tokens(images, prototype_state)
        image_features = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        return image_features, patch_tokens, prompts


class LightweightDetectionHead(nn.Module):
    """FCOS-style dense detection head with separate cls / box / centerness branches.

    Reshapes ViT patch tokens into a square feature map (14×14 for ViT-B/16 @ 224),
    then applies a shared stem + three lightweight prediction branches:

      - Classification: C sigmoid logits per location  (Focal Loss)
      - Box regression: 4 [l, t, r, b] distances per location  (GIoU Loss)
      - Centerness:     1 score per location  (BCE Loss)

    The box branch predicts [l, t, r, b] distances from the grid centre to the
    four box boundaries, normalized by the feature stride (16 px for ViT-B/16).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 384,
        neck_dim: int = 128,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Shared stem
        self.stem = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # Three lightweight prediction branches
        self.cls_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, neck_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(neck_dim, num_classes, kernel_size=1),
        )
        self.box_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, neck_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(neck_dim, 4, kernel_size=1),
        )
        self.ctr_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, neck_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(neck_dim, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.cls_branch, self.box_branch, self.ctr_branch]:
            last_conv = module[-1]
            if isinstance(last_conv, nn.Conv2d):
                nn.init.normal_(last_conv.weight, mean=0.0, std=0.01)
                if last_conv.bias is not None:
                    nn.init.constant_(last_conv.bias, 0.0)
        # Bias towards background at init helps early training stability
        # log(0.076) ≈ -2.5 — prior: ~7.6% of locations are positive
        # Reduced from -4.595 to avoid multi-epoch dead detector (mAP=0)
        last_cls_conv = self.cls_branch[-1]
        if isinstance(last_cls_conv, nn.Conv2d) and last_cls_conv.bias is not None:
            nn.init.constant_(last_cls_conv.bias, -2.5)

    def forward(self, patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return per-location class logits, [l,t,r,b] boxes, and centerness logits."""
        batch, num_patches, dim = patch_tokens.shape
        side = int(num_patches ** 0.5)
        if side * side != num_patches:
            raise ValueError("Patch tokens must form a square feature map.")
        feature_map = patch_tokens.transpose(1, 2).reshape(batch, dim, side, side)

        shared = self.stem(feature_map)
        return {
            "class_logits": self.cls_branch(shared),   # [B, C, H, W]
            "boxes": self.box_branch(shared),           # [B, 4, H, W] — [l, t, r, b]
            "centerness": self.ctr_branch(shared),      # [B, 1, H, W] — logit
        }


class PROBEModel(nn.Module):
    """End-to-end PROBE scaffold: SPEM-enhanced ViT plus detection head."""

    def __init__(
        self,
        backbone: PromptEnhancedViT,
        detection_head: LightweightDetectionHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.detection_head = detection_head

    def encode(
        self,
        images: torch.Tensor,
        prototype_state: PrototypeState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.backbone(images, prototype_state)

    def detect(
        self,
        images: torch.Tensor,
        prototype_state: PrototypeState,
    ) -> dict[str, torch.Tensor]:
        _, patch_tokens, _ = self.encode(images, prototype_state)
        return self.detection_head(patch_tokens)
