from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from probe.models.prompts import PromptConsistencyLoss, PrototypeState


def negative_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return -(pred * target).sum(dim=-1).mean()


class SimSiamHeads(nn.Module):
    """Projector/predictor heads for the PROBE SSL objective."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 2048, out_dim: int = 2048) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, out_dim),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projection = self.projector(features)
        prediction = self.predictor(projection)
        return projection, prediction


class DomainAlignmentHead(nn.Module):
    """Small projection head fp used by DAPA before linear-kernel MMD."""

    def __init__(self, embed_dim: int = 768, projection_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def simsiam_loss(
    heads: SimSiamHeads,
    view1_features: torch.Tensor,
    view2_features: torch.Tensor,
) -> torch.Tensor:
    z1, p1 = heads(view1_features)
    z2, p2 = heads(view2_features)
    return 0.5 * (
        negative_cosine_similarity(p1, z2)
        + negative_cosine_similarity(p2, z1)
    )


def linear_mmd_loss(
    alignment_head: DomainAlignmentHead,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    source_projected = alignment_head(source_features)
    target_projected = alignment_head(target_features)
    return (source_projected.mean(dim=0) - target_projected.mean(dim=0)).pow(2).sum()


def probe_pretrain_step(
    model,
    ssl_heads: SimSiamHeads,
    alignment_head: DomainAlignmentHead,
    source_images: torch.Tensor,
    target_view1: torch.Tensor,
    target_view2: torch.Tensor,
    prototype_state: PrototypeState,
    optimizer,
    prompt_weight: float = 1.0,
    dapa_weight: float = 0.5,
    prompt_temperature: float = 0.2,
) -> dict[str, float]:
    """One paper-aligned PROBE pre-training step.

    The frozen ViT produces prompt-conditioned features. Trainable pieces are
    the SPEM prompt projector, SimSiam heads and DAPA projection head.
    """

    source_features, _, _ = model.encode(source_images, prototype_state)
    target_features1, _, target_prompts1 = model.encode(target_view1, prototype_state)
    target_features2, _, _ = model.encode(target_view2, prototype_state)

    loss_ssl = simsiam_loss(ssl_heads, target_features1, target_features2)
    loss_prompt = PromptConsistencyLoss(prompt_temperature)(
        target_features1,
        target_prompts1,
    )
    loss_dapa = linear_mmd_loss(alignment_head, source_features, target_features1)
    loss = loss_ssl + prompt_weight * loss_prompt + dapa_weight * loss_dapa

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "ssl": float(loss_ssl.detach().cpu()),
        "prompt": float(loss_prompt.detach().cpu()),
        "dapa": float(loss_dapa.detach().cpu()),
    }
