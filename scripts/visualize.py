"""Visualization tools for PROBE training.

Provides:
  - plot_spem_tsne:   T-SNE of target patch features + discovered prototypes
  - plot_loss_curves: Training loss curves (total, ssl, prompt, dapa)
  - plot_dapa_alignment: Source vs target feature T-SNE after DAPA projection
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)

CHECKPOINT_DIR = Path("checkpoints")


# ---------------------------------------------------------------------------
# SPEM prototype T-SNE
# ---------------------------------------------------------------------------
def plot_spem_tsne(
    patch_features: torch.Tensor,
    centroids: torch.Tensor,
    components: torch.Tensor,
    mean: torch.Tensor,
    save_path: str | Path,
    max_patches: int = 3000,
) -> None:
    """T-SNE of target-domain patch embeddings in PCA space, with prototype overlay.

    Args:
        patch_features: [N, D] raw ViT patch embeddings (before PCA).
        centroids: [K, d0] K-means centroids in PCA space.
        components: [D, d0] PCA projection matrix.
        mean: [D] PCA mean vector.
        save_path: output PNG path.
        max_patches: cap on patches passed to T-SNE (keeps runtime manageable).
    """
    patch_features = patch_features.detach().cpu()
    centroids = centroids.detach().cpu()
    components = components.detach().cpu()
    mean = mean.detach().cpu()

    # Project patches into PCA space
    pca_patches = (patch_features - mean.unsqueeze(0)) @ components

    # Subsample patches for T-SNE speed
    n = pca_patches.shape[0]
    if n > max_patches:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_patches, replace=False)
        pca_patches = pca_patches[idx]

    # Assign each patch to nearest prototype
    dists = torch.cdist(pca_patches, centroids)  # [P, K]
    assignments = dists.argmin(dim=1).numpy()

    # T-SNE on PCA features + centroids jointly
    combined = torch.cat([pca_patches, centroids], dim=0).numpy()
    tsne = TSNE(n_components=2, perplexity=min(30, len(combined) // 3), random_state=42)
    embedded = tsne.fit_transform(combined)

    patch_xy = embedded[: pca_patches.shape[0]]
    centroid_xy = embedded[pca_patches.shape[0] :]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.tab10
    for k in range(centroids.shape[0]):
        mask = assignments == k
        ax.scatter(
            patch_xy[mask, 0],
            patch_xy[mask, 1],
            s=3,
            alpha=0.4,
            color=cmap(k),
            label=f"Prototype {k}",
        )
    ax.scatter(
        centroid_xy[:, 0],
        centroid_xy[:, 1],
        s=200,
        c="red",
        marker="*",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label="Prototypes",
    )
    ax.set_title("SPEM: Target Patch T-SNE (PCA space) + Discovered Prototypes")
    ax.legend(markerscale=2, fontsize=7, loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] SPEM T-SNE saved → {save_path}")


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------
def plot_loss_curves(
    history: dict[str, list[float]],
    save_path: str | Path,
) -> None:
    """Plot per-epoch averaged training losses.

    Args:
        history: dict with keys "loss", "ssl", "prompt", "dapa", each a list.
        save_path: output PNG path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metrics = [
        ("loss", "Total Loss", "#1f77b4"),
        ("ssl", "SimSiam SSL (neg cosine)", "#ff7f0e"),
        ("prompt", "Prompt Consistency (InfoNCE)", "#2ca02c"),
        ("dapa", "DAPA MMD²", "#d62728"),
    ]
    for (key, title, color), ax in zip(metrics, axes.flat):
        if key not in history or len(history[key]) == 0:
            continue
        ax.plot(history[key], color=color, linewidth=1.2)
        ax.scatter(
            range(len(history[key])),
            history[key],
            s=10,
            color=color,
            alpha=0.5,
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    fig.suptitle("PROBE Pretraining Loss Curves", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Loss curves saved → {save_path}")


# ---------------------------------------------------------------------------
# DAPA domain alignment T-SNE
# ---------------------------------------------------------------------------
@torch.no_grad()
def plot_dapa_alignment(
    model,
    alignment_head: torch.nn.Module,
    source_dataset,
    target_dataset,
    prototype_state,
    device: torch.device,
    save_path: str | Path,
    num_samples: int = 200,
    batch_size: int = 4,
    image_size: int = 224,
) -> None:
    """T-SNE of source vs target CLS features after DAPA projection, before MMD.

    If the features from the two domains overlap well, DAPA is working.
    """
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    model.eval()
    alignment_head.eval()

    def collate(batch: list) -> tuple:
        imgs, tgts = zip(*batch)
        if isinstance(imgs[0], torch.Tensor):
            imgs = torch.stack(imgs, 0)
        return imgs, tgts

    def transform_for_image(img):
        """Convert PIL image to normalized tensor."""
        from PIL import Image
        t = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return t(img)

    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    source_feats, target_feats = [], []

    for batch in source_loader:
        images, _ = batch
        if isinstance(images, (list, tuple)):
            images = torch.stack([transform_for_image(img) for img in images])
        src_feat, _, _ = model.encode(images.to(device), prototype_state)
        proj = alignment_head(src_feat).cpu()
        source_feats.append(proj)
        if sum(f.shape[0] for f in source_feats) >= num_samples:
            break
    source_feats = torch.cat(source_feats, dim=0)[:num_samples]

    for batch in target_loader:
        images, _ = batch
        if isinstance(images, (list, tuple)):
            images = torch.stack([transform_for_image(img) for img in images])
        tgt_feat, _, _ = model.encode(images.to(device), prototype_state)
        proj = alignment_head(tgt_feat).cpu()
        target_feats.append(proj)
        if sum(f.shape[0] for f in target_feats) >= num_samples:
            break
    target_feats = torch.cat(target_feats, dim=0)[:num_samples]

    combined = torch.cat([source_feats, target_feats], dim=0).numpy()
    labels = np.array(
        [0] * source_feats.shape[0] + [1] * target_feats.shape[0]
    )
    tsne = TSNE(n_components=2, perplexity=min(30, len(combined) // 3), random_state=42)
    embedded = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, name, color in [(0, "Source (labeled)", "#1f77b4"), (1, "Target (unlabeled)", "#ff7f0e")]:
        mask = labels == label
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=6,
            alpha=0.6,
            color=color,
            label=name,
        )
    ax.set_title("DAPA Domain Alignment — Source vs Target Features (T-SNE)")
    ax.legend(fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] DAPA alignment saved → {save_path}")
