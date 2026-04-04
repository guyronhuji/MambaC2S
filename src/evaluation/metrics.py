"""Evaluation metrics for CyTOF sequence models.

Functions
---------
compute_loss_perplexity   — self-supervised loss and perplexity on a DataLoader
extract_embeddings        — batch extraction of cell embeddings from a model
compute_embedding_metrics — ARI, NMI, kNN purity from embeddings + labels
compute_umap              — 2-D UMAP projection of embeddings
"""

from __future__ import annotations

import logging
import math
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


# ===========================================================================
# Self-supervised metrics
# ===========================================================================

@torch.no_grad()
def compute_loss_perplexity(
    model: BaseModel,
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
    pad_id: int = 0,
) -> dict[str, float]:
    """Compute average cross-entropy loss and perplexity on a DataLoader.

    Args:
        model: Trained autoregressive model.
        dataloader: Yields ``{input_ids, attention_mask}`` batches.
        device: Compute device.
        pad_id: Token id to ignore in loss computation.

    Returns:
        ``{"loss": float, "perplexity": float}``
    """
    import torch.nn.functional as F

    device = torch.device(device) if isinstance(device, str) else device
    model.eval()
    model.to(device)

    total_loss = 0.0
    n_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        src = input_ids[:, :-1]
        tgt = input_ids[:, 1:]
        src_mask = attn_mask[:, :-1]
        tgt_mask = attn_mask[:, 1:]

        logits = model(src, attention_mask=src_mask)
        B, T, V = logits.shape

        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            tgt.reshape(B * T),
            ignore_index=pad_id,
            reduction="none",
        )
        mask_flat = tgt_mask.reshape(B * T).float()
        total_loss += (loss * mask_flat).sum().item()
        n_tokens += mask_flat.sum().item()

    avg_loss = total_loss / max(n_tokens, 1)
    perplexity = math.exp(min(avg_loss, 100))
    return {"loss": avg_loss, "perplexity": perplexity}


# ===========================================================================
# Embedding extraction
# ===========================================================================

@torch.no_grad()
def extract_embeddings(
    model: BaseModel,
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
    pooling: Literal["mean", "last"] = "mean",
) -> np.ndarray:
    """Extract cell embeddings by running the encoder.

    Args:
        model: Trained model with a working :meth:`~BaseModel.encode` method.
        dataloader: Yields ``{input_ids, attention_mask}`` batches.
        device: Compute device.
        pooling: Pooling strategy (``"mean"`` or ``"last"``).

    Returns:
        Float32 NumPy array of shape ``(n_cells, d_model)``.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model.eval()
    model.to(device)
    embeddings = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        emb = model.encode(input_ids, attention_mask=attn_mask, pooling=pooling)
        embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0).astype(np.float32)


# ===========================================================================
# Downstream embedding quality
# ===========================================================================

def compute_embedding_metrics(
    embeddings: np.ndarray,
    labels: list[str] | np.ndarray,
    knn_k: int = 15,
    n_clusters: Optional[int] = None,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate embedding quality with clustering and kNN metrics.

    Args:
        embeddings: Float array of shape ``(n_cells, d_model)``.
        labels: Ground-truth cell-type labels (strings or ints).
        knn_k: Number of neighbours for kNN purity.
        n_clusters: Number of KMeans clusters. Defaults to the number
            of unique labels.
        seed: Random seed for KMeans.

    Returns:
        ``{"ARI": float, "NMI": float, "knn_purity": float}``
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import LabelEncoder

    labels_arr = np.asarray(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels_arr)
    n_unique = len(le.classes_)

    k = n_clusters if n_clusters is not None else n_unique
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    pred_labels = kmeans.fit_predict(embeddings)

    ari = adjusted_rand_score(labels_int, pred_labels)
    nmi = normalized_mutual_info_score(labels_int, pred_labels, average_method="arithmetic")

    # kNN purity: fraction of neighbours sharing the same label
    nn = NearestNeighbors(n_neighbors=knn_k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    indices = indices[:, 1:]  # exclude self

    purities = []
    for i, nbrs in enumerate(indices):
        nbr_labels = labels_int[nbrs]
        majority = np.bincount(nbr_labels).argmax()
        purities.append(float(nbr_labels == majority).mean())
    knn_purity = float(np.mean(purities))

    logger.info(
        "Embedding metrics — ARI=%.4f  NMI=%.4f  kNN-purity=%.4f",
        ari, nmi, knn_purity,
    )
    return {"ARI": ari, "NMI": nmi, "knn_purity": knn_purity}


# ===========================================================================
# UMAP
# ===========================================================================

def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    metric: str = "euclidean",
) -> np.ndarray:
    """Reduce embeddings to 2-D via UMAP.

    Args:
        embeddings: Float array of shape ``(n_cells, d_model)``.
        n_neighbors: UMAP ``n_neighbors`` parameter.
        min_dist: UMAP ``min_dist`` parameter.
        seed: Random state for reproducibility.
        metric: Distance metric.

    Returns:
        Float32 array of shape ``(n_cells, 2)``.

    Raises:
        ImportError: If ``umap-learn`` is not installed.
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP: pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        metric=metric,
    )
    coords = reducer.fit_transform(embeddings)
    return coords.astype(np.float32)
