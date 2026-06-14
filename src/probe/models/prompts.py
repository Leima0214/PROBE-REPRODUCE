from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PrototypeState:
    """Target-domain prototype state used by SPEM.

    ``mean`` and ``components`` define the PCA projection from ViT patch
    embeddings to the low-dimensional discovery space; ``centroids`` stores
    the K-means visual prototypes in that PCA space.
    """

    mean: torch.Tensor
    components: torch.Tensor
    centroids: torch.Tensor


def pca_reduce(features: torch.Tensor, out_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project features to their leading principal components."""

    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor of shape [num_tokens, dim].")
    if not 1 <= out_dim <= features.shape[1]:
        raise ValueError("out_dim must be in [1, feature_dim].")

    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    _, _, components = torch.pca_lowrank(centered, q=out_dim)
    reduced = centered @ components
    return reduced, mean.squeeze(0), components


def kmeans(
    features: torch.Tensor,
    num_clusters: int,
    num_iters: int = 25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Small torch K-means used to discover target visual prototypes."""

    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor.")
    if not 1 <= num_clusters <= features.shape[0]:
        raise ValueError("num_clusters must be in [1, num_features].")

    # Deterministic spread initialization keeps the scaffold reproducible.
    init_ids = torch.linspace(
        0,
        features.shape[0] - 1,
        steps=num_clusters,
        device=features.device,
    ).long()
    centroids = features[init_ids].clone()

    for _ in range(num_iters):
        distances = torch.cdist(features, centroids)
        assignments = distances.argmin(dim=1)
        next_centroids = []
        for cluster_id in range(num_clusters):
            mask = assignments == cluster_id
            if mask.any():
                next_centroids.append(features[mask].mean(dim=0))
            else:
                next_centroids.append(centroids[cluster_id])
        updated = torch.stack(next_centroids, dim=0)
        if torch.norm(updated - centroids) < eps:
            centroids = updated
            break
        centroids = updated
    return centroids


class TargetPrototypeDiscovery:
    """PCA + K-means prototype discovery over unlabeled target patches."""

    def __init__(
        self,
        pca_dim: int = 50,
        num_prototypes: int = 10,
        kmeans_iters: int = 25,
    ) -> None:
        self.pca_dim = pca_dim
        self.num_prototypes = num_prototypes
        self.kmeans_iters = kmeans_iters

    @torch.no_grad()
    def fit(self, patch_features: torch.Tensor) -> PrototypeState:
        reduced, mean, components = pca_reduce(patch_features, self.pca_dim)
        centroids = kmeans(reduced, self.num_prototypes, self.kmeans_iters)
        return PrototypeState(mean=mean, components=components, centroids=centroids)

    @staticmethod
    def transform(patch_features: torch.Tensor, state: PrototypeState) -> torch.Tensor:
        return (patch_features - state.mean.unsqueeze(0)) @ state.components


class PromptProjector(nn.Module):
    """Two-layer MLP that maps PCA prototypes back to ViT prompt tokens."""

    def __init__(
        self,
        pca_dim: int = 50,
        embed_dim: int = 768,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pca_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, prototypes: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        prompts = self.net(prototypes)
        if batch_size is None:
            return prompts
        return prompts.unsqueeze(0).expand(batch_size, -1, -1)


class PromptInjector(nn.Module):
    """Inject SPEM prompts by concatenating them before image tokens."""

    def __init__(self, injection_layers: tuple[int, ...] = (0, 6)) -> None:
        super().__init__()
        self.injection_layers = set(injection_layers)

    def should_inject(self, layer_id: int) -> bool:
        return layer_id in self.injection_layers

    def insert(self, tokens: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
        return torch.cat([prompts, tokens], dim=1)

    def remove(self, tokens: torch.Tensor, prompt_count: int) -> torch.Tensor:
        return tokens[:, prompt_count:, :]


class PromptConsistencyLoss(nn.Module):
    """InfoNCE-style loss between image features and their prompt means."""

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features: torch.Tensor, prompt_tokens: torch.Tensor) -> torch.Tensor:
        prompt_means = prompt_tokens.mean(dim=1)
        image_features = F.normalize(image_features, dim=-1)
        prompt_means = F.normalize(prompt_means, dim=-1)
        logits = image_features @ prompt_means.t()
        logits = logits / self.temperature
        labels = torch.arange(image_features.shape[0], device=image_features.device)
        return F.cross_entropy(logits, labels)
