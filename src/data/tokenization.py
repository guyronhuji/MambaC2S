"""Cell tokenization schemes.

Three schemes supported:

A. rank_only
    Markers sorted by within-cell expression rank (lowest first).
    Tokens: marker names only.
    Example: ["BOS", "CD3", "CD4", "CCR7", ..., "EOS"]

B. strength_only
    Markers sorted by within-cell rank, each paired with its strength bin.
    Tokens: marker_BIN pairs.
    Example: ["BOS", "CD3_HIGH", "CD4_HIGH", "CCR7_LOW", ..., "EOS"]

C. hybrid
    Interleaved rank position token and strength token per marker.
    Tokens: marker_R{rank}, marker_BIN, ...
    Example: ["BOS", "CD3_R1", "CD3_HIGH", "CD4_R2", "CD4_HIGH", ..., "EOS"]

All schemes prepend BOS and append EOS.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TokenizationScheme = Literal["rank_only", "strength_only", "hybrid"]

BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"


def tokenize_cells(
    df: pd.DataFrame,
    scheme: TokenizationScheme = "rank_only",
) -> list[list[str]]:
    """Convert preprocessed CyTOF cells to token sequences.

    Args:
        df: Preprocessed DataFrame from :func:`~src.data.preprocessing.preprocess`.
            Must contain ``{marker}_rank`` and ``{marker}_bin`` columns.
        scheme: One of ``"rank_only"``, ``"strength_only"``, or ``"hybrid"``.

    Returns:
        List of token sequences, one per cell. Each sequence starts with
        ``BOS`` and ends with ``EOS``.

    Raises:
        ValueError: If required ``_rank`` or ``_bin`` columns are missing.
        ValueError: If ``scheme`` is not one of the supported schemes.
    """
    marker_cols = _get_base_markers(df)

    if not marker_cols:
        raise ValueError("No marker columns found. Run preprocessing first.")

    # Verify required derived columns exist
    missing_rank = [m for m in marker_cols if f"{m}_rank" not in df.columns]
    missing_bin = [m for m in marker_cols if f"{m}_bin" not in df.columns]
    if missing_rank:
        raise ValueError(
            f"Missing _rank columns for: {missing_rank[:5]}... "
            "Run preprocess() before tokenize_cells()."
        )
    if missing_bin:
        raise ValueError(
            f"Missing _bin columns for: {missing_bin[:5]}... "
            "Run preprocess() before tokenize_cells()."
        )

    logger.info(
        "Tokenizing %d cells with scheme='%s', %d markers.",
        len(df), scheme, len(marker_cols),
    )

    if scheme == "rank_only":
        sequences = _tokenize_rank_only(df, marker_cols)
    elif scheme == "strength_only":
        sequences = _tokenize_strength_only(df, marker_cols)
    elif scheme == "hybrid":
        sequences = _tokenize_hybrid(df, marker_cols)
    else:
        raise ValueError(
            f"Unknown scheme: {scheme!r}. "
            "Choose from 'rank_only', 'strength_only', 'hybrid'."
        )

    return sequences


def tokenize_single_cell(
    row: pd.Series,
    marker_cols: list[str],
    scheme: TokenizationScheme = "rank_only",
) -> list[str]:
    """Tokenize a single cell (row of the preprocessed DataFrame).

    Args:
        row: A row from the preprocessed DataFrame.
        marker_cols: Base marker column names.
        scheme: Tokenization scheme.

    Returns:
        Token sequence including BOS and EOS.
    """
    ranks = np.array([row[f"{m}_rank"] for m in marker_cols])
    sorted_indices = np.argsort(ranks)
    sorted_markers = [marker_cols[i] for i in sorted_indices]

    if scheme == "rank_only":
        tokens = [BOS] + sorted_markers + [EOS]

    elif scheme == "strength_only":
        tokens = [BOS]
        for m in sorted_markers:
            tokens.append(f"{m}_{row[f'{m}_bin']}")
        tokens.append(EOS)

    elif scheme == "hybrid":
        tokens = [BOS]
        for rank_pos, m in enumerate(sorted_markers):
            tokens.append(f"{m}_R{rank_pos + 1}")
            tokens.append(f"{m}_{row[f'{m}_bin']}")
        tokens.append(EOS)

    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    return tokens


# ---------------------------------------------------------------------------
# Vectorised implementations (for speed)
# ---------------------------------------------------------------------------

def _get_base_markers(df: pd.DataFrame) -> list[str]:
    """Return base marker names from a preprocessed DataFrame."""
    non_marker = {"cell_id", "label"}
    return [
        c for c in df.columns
        if c not in non_marker
        and not c.endswith("_rank")
        and not c.endswith("_bin")
    ]


def _tokenize_rank_only(df: pd.DataFrame, marker_cols: list[str]) -> list[list[str]]:
    """Scheme A: markers sorted by rank, no strength info."""
    rank_cols = [f"{m}_rank" for m in marker_cols]
    rank_matrix = df[rank_cols].values  # shape (n_cells, n_markers)
    sorted_indices = np.argsort(rank_matrix, axis=1)

    sequences: list[list[str]] = []
    marker_arr = np.array(marker_cols)
    for i in range(len(df)):
        sorted_markers = marker_arr[sorted_indices[i]].tolist()
        sequences.append([BOS] + sorted_markers + [EOS])
    return sequences


def _tokenize_strength_only(df: pd.DataFrame, marker_cols: list[str]) -> list[list[str]]:
    """Scheme B: markers sorted by rank, each shown as marker_BIN."""
    rank_cols = [f"{m}_rank" for m in marker_cols]
    bin_cols = [f"{m}_bin" for m in marker_cols]
    rank_matrix = df[rank_cols].values
    bin_matrix = df[bin_cols].values  # str dtype
    sorted_indices = np.argsort(rank_matrix, axis=1)

    marker_arr = np.array(marker_cols)
    sequences: list[list[str]] = []
    for i in range(len(df)):
        idx = sorted_indices[i]
        tokens = [BOS]
        for j in idx:
            tokens.append(f"{marker_arr[j]}_{bin_matrix[i, j]}")
        tokens.append(EOS)
        sequences.append(tokens)
    return sequences


def _tokenize_hybrid(df: pd.DataFrame, marker_cols: list[str]) -> list[list[str]]:
    """Scheme C: interleaved rank-position and strength tokens per marker."""
    rank_cols = [f"{m}_rank" for m in marker_cols]
    bin_cols = [f"{m}_bin" for m in marker_cols]
    rank_matrix = df[rank_cols].values
    bin_matrix = df[bin_cols].values
    sorted_indices = np.argsort(rank_matrix, axis=1)

    marker_arr = np.array(marker_cols)
    sequences: list[list[str]] = []
    for i in range(len(df)):
        idx = sorted_indices[i]
        tokens = [BOS]
        for rank_pos, j in enumerate(idx):
            tokens.append(f"{marker_arr[j]}_R{rank_pos + 1}")
            tokens.append(f"{marker_arr[j]}_{bin_matrix[i, j]}")
        tokens.append(EOS)
        sequences.append(tokens)
    return sequences


# ---------------------------------------------------------------------------
# Convenience aliases (used by scripts and notebooks)
# ---------------------------------------------------------------------------

def tokenize_dataframe(
    df: pd.DataFrame,
    scheme: TokenizationScheme = "rank_only",
    bins: list[str] | None = None,  # ignored; bins are embedded in _bin columns
) -> list[list[str]]:
    """Alias for :func:`tokenize_cells` with an optional (ignored) ``bins`` arg.

    The ``bins`` parameter is accepted for API consistency with callers that
    pass it, but it has no effect — bin labels are already baked into the
    ``{marker}_bin`` columns by the preprocessing step.
    """
    return tokenize_cells(df, scheme=scheme)


def tokenize_cell(
    row: pd.Series,
    marker_cols: list[str],
    scheme: TokenizationScheme = "rank_only",
) -> list[str]:
    """Alias for :func:`tokenize_single_cell`."""
    return tokenize_single_cell(row, marker_cols, scheme=scheme)
