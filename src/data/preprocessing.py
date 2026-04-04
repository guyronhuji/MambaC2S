"""CyTOF data preprocessing.

Pipeline:
  1. arcsinh transform (optional, cofactor configurable)
  2. Per-marker normalization: z-score or robust (median/IQR)
  3. Rank within cell (argsort of marker values per row)
  4. Strength binning (e.g. LOW / MED / HIGH per marker per cell)

All decisions are logged to ``preprocessing_log.json``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess(
    df: pd.DataFrame,
    arcsinh: bool = True,
    arcsinh_cofactor: float = 5.0,
    normalization: Literal["zscore", "robust"] = "zscore",
    bin_count: int = 3,
    bins: Optional[list[str]] = None,
    exclude_markers: Optional[list[str]] = None,
    log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Apply the full CyTOF preprocessing pipeline.

    Args:
        df: Input DataFrame from :func:`~src.data.loader.load_levine32`.
            Must have a ``cell_id`` column and at least one marker column.
        arcsinh: Whether to apply arcsinh transformation.
        arcsinh_cofactor: Cofactor for arcsinh: ``arcsinh(x / cofactor)``.
        normalization: ``"zscore"`` (mean/std) or ``"robust"`` (median/IQR).
        bin_count: Number of strength bins.
        bins: Names for the bins from lowest to highest expression.
            Defaults to ``["LOW", "MED", "HIGH"]`` for ``bin_count=3``.
        exclude_markers: Marker columns to skip entirely.
        log_path: If provided, save a JSON log of decisions here.

    Returns:
        Processed DataFrame with the same structure as ``df`` plus
        additional columns:
          - ``{marker}_rank`` (int): rank of each marker within its cell
          - ``{marker}_bin`` (str): strength bin label

        The core marker columns are updated in-place (arcsinh + normalised).
    """
    if bins is None:
        bins = ["LOW", "MED", "HIGH"][:bin_count]
    if len(bins) != bin_count:
        raise ValueError(f"len(bins)={len(bins)} must equal bin_count={bin_count}")

    # Identify marker columns (everything except cell_id and label)
    non_marker = {"cell_id", "label"}
    marker_cols = [
        c for c in df.columns
        if c not in non_marker and not c.endswith("_rank") and not c.endswith("_bin")
    ]
    if exclude_markers:
        marker_cols = [c for c in marker_cols if c not in exclude_markers]

    if not marker_cols:
        raise ValueError("No marker columns found in DataFrame.")

    log: dict = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_cells": len(df),
        "n_markers": len(marker_cols),
        "markers": marker_cols,
        "steps": [],
    }

    result = df.copy()
    X = result[marker_cols].values.astype(np.float64)

    # Step 1: arcsinh transform
    if arcsinh:
        X = np.arcsinh(X / arcsinh_cofactor)
        log["steps"].append({
            "step": "arcsinh",
            "cofactor": arcsinh_cofactor,
            "formula": "arcsinh(x / cofactor)",
        })
        logger.info("Applied arcsinh transform (cofactor=%.1f).", arcsinh_cofactor)

    # Step 2: per-marker normalization
    if normalization == "zscore":
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds[stds == 0] = 1.0  # avoid division by zero
        X = (X - means) / stds
        log["steps"].append({
            "step": "normalization",
            "method": "zscore",
            "note": "per-marker mean subtraction and std division",
        })
        logger.info("Applied z-score normalization per marker.")
    elif normalization == "robust":
        medians = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1.0
        X = (X - medians) / iqr
        log["steps"].append({
            "step": "normalization",
            "method": "robust",
            "note": "per-marker (x - median) / IQR",
        })
        logger.info("Applied robust normalization (median/IQR) per marker.")
    else:
        raise ValueError(f"Unknown normalization method: {normalization!r}")

    # Write normalized values back
    result[marker_cols] = X

    # Step 3: rank within cell (argsort)
    # rank[i, j] = rank of marker j for cell i (0-based, lowest expression = rank 0)
    rank_matrix = np.argsort(np.argsort(X, axis=1), axis=1)  # stable double-argsort for ranks
    for j, col in enumerate(marker_cols):
        result[f"{col}_rank"] = rank_matrix[:, j]

    log["steps"].append({
        "step": "rank_within_cell",
        "note": "0-based argsort rank of each marker's expression within each cell",
    })
    logger.info("Computed within-cell ranks for %d markers.", len(marker_cols))

    # Step 4: strength binning
    # Bin by quantile of each marker (across all cells)
    bin_labels_ordered = bins  # low -> high
    for j, col in enumerate(marker_cols):
        vals = X[:, j]
        bin_edges = np.percentile(vals, np.linspace(0, 100, bin_count + 1))
        # Make upper edge inclusive
        bin_edges[-1] += 1e-9
        bin_indices = np.searchsorted(bin_edges[1:], vals, side="left")
        bin_indices = np.clip(bin_indices, 0, bin_count - 1)
        result[f"{col}_bin"] = [bin_labels_ordered[i] for i in bin_indices]

    log["steps"].append({
        "step": "strength_binning",
        "bin_count": bin_count,
        "bin_labels": bins,
        "method": "equal-frequency quantile bins per marker across all cells",
    })
    logger.info("Binned markers into %d strength bins: %s.", bin_count, bins)

    # Save log
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        logger.info("Preprocessing log saved to %s.", log_path)

    return result


def preprocess_from_config(
    df: pd.DataFrame,
    cfg: dict,
    log_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper: run the preprocessing pipeline from a config dict.

    Unpacks the ``preprocessing`` section of the YAML config and delegates to
    :func:`preprocess`.  Returns both the processed DataFrame and the log dict.

    Args:
        df: Raw DataFrame from the data loader.
        cfg: The ``preprocessing`` sub-dict from the YAML config, with keys
            ``arcsinh``, ``arcsinh_cofactor``, ``normalization``,
            ``bin_count``, ``bins``.
        log_path: Optional path to save the preprocessing log as JSON.

    Returns:
        ``(processed_df, log_dict)`` tuple.
    """
    import copy, datetime

    processed = preprocess(
        df,
        arcsinh=cfg.get("arcsinh", True),
        arcsinh_cofactor=float(cfg.get("arcsinh_cofactor", 5.0)),
        normalization=cfg.get("normalization", "zscore"),
        bin_count=int(cfg.get("bin_count", 3)),
        bins=cfg.get("bins", None),
        log_path=log_path,
    )

    log = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "config": copy.deepcopy(cfg),
        "n_cells_in": len(df),
        "n_cells_out": len(processed),
    }
    return processed, log


def get_marker_columns(df: pd.DataFrame) -> list[str]:
    """Return base marker column names (excludes cell_id, label, _rank, _bin columns)."""
    non_marker = {"cell_id", "label"}
    return [
        c for c in df.columns
        if c not in non_marker
        and not c.endswith("_rank")
        and not c.endswith("_bin")
    ]
