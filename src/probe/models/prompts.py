from __future__ import annotations

from copy import deepcopy
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


class MoCoPromptConsistencyLoss(nn.Module):
    """MoCo-style InfoNCE with momentum encoder and negative-sample queue.

    Unlike ``PromptConsistencyLoss`` which can only contrast negatives within
    the current micro-batch, this maintains a FIFO queue of past prompt means
    encoded by a momentum-updated projector — providing thousands of
    consistent negatives even when ``batch_size`` is small (e.g. 4).

    Parameters
    ----------
    projector:
        Live ``PromptProjector`` whose weights are tracked by the momentum copy.
    queue_size:
        Capacity of the negative-sample FIFO queue.
    temperature:
        InfoNCE temperature (paper default 0.2).
    momentum:
        EMA coefficient for the momentum projector: ``θ_k ← m·θ_k + (1-m)·θ_q``.
    """

    def __init__(
        self,
        projector: PromptProjector,
        queue_size: int = 4096,
        temperature: float = 0.2,
        momentum: float = 0.999,
    ) -> None:
        super().__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.momentum = momentum

        embed_dim = projector.net[-1].out_features

        # Momentum projector — frozen, updated via EMA each step
        self.momentum_projector = deepcopy(projector)
        for p in self.momentum_projector.parameters():
            p.requires_grad = False

        self.register_buffer("queue", torch.zeros(queue_size, embed_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor) -> None:
        """Push *keys* into the FIFO queue (circular buffer)."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        if ptr + batch_size > self.queue_size:
            tail = self.queue_size - ptr
            self.queue[ptr:] = keys[:tail]
            self.queue[: batch_size - tail] = keys[tail:]
        else:
            self.queue[ptr : ptr + batch_size] = keys
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    @torch.no_grad()
    def _momentum_update(self, projector: PromptProjector) -> None:
        """θ_k ← m·θ_k + (1-m)·θ_q"""
        for p_m, p_q in zip(
            self.momentum_projector.parameters(), projector.parameters()
        ):
            p_m.data = self.momentum * p_m.data + (1.0 - self.momentum) * p_q.data

    def forward(
        self,
        image_features: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prototypes: torch.Tensor,
        projector: PromptProjector,
    ) -> torch.Tensor:
        """Compute MoCo InfoNCE loss.

        Parameters
        ----------
        image_features:
            [B, D] CLS-token features from the current ViT (query).
        prompt_tokens:
            [B, K, D] injected prompt tokens from the current projector.
        prototypes:
            [P, d] PCA centroids — input to the momentum projector.
        projector:
            Live ``PromptProjector`` whose weights will be momentum-updated
            after this step.
        """
        B = image_features.shape[0]

        # Query — current prompt means, normalized
        query = F.normalize(prompt_tokens.mean(dim=1), dim=-1)  # [B, D]

        # Key — momentum-encoded prompt means (no gradient)
        with torch.no_grad():
            proto = prototypes.to(prompt_tokens.device)
            key_tokens = self.momentum_projector(proto, B)    # [B, K, D]
            key = F.normalize(key_tokens.mean(dim=1), dim=-1)  # [B, D]

        # Positive logits — query·key for each sample
        pos = (query * key).sum(dim=-1, keepdim=True) / self.temperature  # [B, 1]

        # Negative logits — query vs queue
        neg_pool = F.normalize(self.queue[: self.queue_size], dim=-1)  # [Q, D]
        neg = query @ neg_pool.t() / self.temperature                  # [B, Q]

        # Cross-entropy: positive is column 0
        logits = torch.cat([pos, neg], dim=1)  # [B, 1+Q]
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # Maintain queue and momentum
        self._enqueue(key)
        self._momentum_update(projector)

        return loss
