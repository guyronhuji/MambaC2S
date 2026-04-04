"""Levine32 CyTOF dataset loader.

Supports loading from:
- .fcs files (via fcsparser)
- .h5ad files (via anndata)
- .csv files (via pandas)

Returns a unified DataFrame with columns:
  cell_id, <marker_1>, ..., <marker_N>, label (str or None)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NumPy 2.0 compatibility shim for fcsparser
# fcsparser calls ndarray.newbyteorder() which was removed in NumPy 2.0.
# Patch it back so PyCytoData works on Colab and other NumPy 2.x environments.
# ---------------------------------------------------------------------------
if not hasattr(np.ndarray, "newbyteorder"):
    def _ndarray_newbyteorder(self, order="S"):
        return self.view(self.dtype.newbyteorder(order))
    np.ndarray.newbyteorder = _ndarray_newbyteorder  # type: ignore[attr-defined]

# Known Levine32 marker names (used for auto-detection when column names are ambiguous)
LEVINE32_MARKERS = [
    "CD45RA", "CD133", "CD19", "CD22", "CD11b", "CD4", "CD8",
    "CD34", "Flt3", "CD20", "CXCR4", "CD235ab", "CD45", "CD123",
    "CD321", "CD14", "CD33", "CD47", "CD11c", "CD7", "CD15",
    "CD16", "CD44", "CD38", "CD13", "CD3", "CD61", "CD117",
    "CD49d", "HLA-DR", "CD64", "CD41",
]


def _clean_channel_name(name: str) -> str:
    """Strip CyTOF isotope annotation from a channel name.

    Converts e.g. ``"CD45RA(La139)Di"`` → ``"CD45RA"``.
    Names without parentheses are returned unchanged.
    """
    import re
    return re.sub(r"\([^)]*\)Di?$", "", str(name)).strip()


def load_levine32(
    data_dir: str | Path | None = None,
    label_col: str = "label",
    markers: Optional[list[str]] = None,
    exclude_markers: Optional[list[str]] = None,
    use_pycytodata: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load the Levine32 CyTOF dataset.

    Loading priority:
    1. **PyCytoData** (``use_pycytodata=True``, default) — downloads once and
       caches locally; no ``data_dir`` required.
    2. Fallback to file-based loading from ``data_dir`` in this order:
       ``levine32_processed.h5ad`` → ``*.h5ad`` → ``*.csv`` → ``*.fcs``

    The Levine32 dataset contains 265,627 cells × 32 lineage markers.
    Channel names arrive as ``"CD45RA(La139)Di"``; isotope tags are stripped
    automatically to give clean names such as ``"CD45RA"``.
    Cell-type labels (where available) are stored in the ``"label"`` column;
    unassigned cells are set to ``NaN``.

    Args:
        data_dir: Directory for file-based loading (only used if PyCytoData is
            unavailable or ``use_pycytodata=False``).
        label_col: Name of the label column in file-based data sources.
        markers: Explicit marker list to retain. Auto-detected if ``None``.
        exclude_markers: Markers to drop.
        use_pycytodata: Whether to attempt loading via PyCytoData first.
        force_download: Passed to ``DataLoader.load_dataset`` when using
            PyCytoData; re-downloads even if already cached.

    Returns:
        DataFrame with columns ``cell_id``, marker columns, and ``label``.
        The ``label`` column is ``NaN`` for unlabeled/unassigned cells.

    Raises:
        FileNotFoundError: If no data source succeeds.
    """
    df = None

    # --- Attempt 1: PyCytoData ---
    if use_pycytodata:
        df = _try_load_pycytodata(force_download=force_download)

    # --- Attempt 2: file-based fallback ---
    if df is None:
        if data_dir is None:
            raise FileNotFoundError(
                "PyCytoData loading failed and no data_dir provided. "
                "Install PyCytoData (pip install PyCytoData) or set data_dir."
            )
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
        df = _try_load(data_dir, label_col)

    if df is None:
        raise FileNotFoundError(
            f"No supported data file found in {data_dir}. "
            "Expected: *.h5ad, *.csv, or *.fcs"
        )

    # Resolve marker columns
    non_marker_cols = {"cell_id", label_col, "label"}
    if markers:
        # Use explicitly provided marker list; keep only those present
        available = [m for m in markers if m in df.columns]
        missing = [m for m in markers if m not in df.columns]
        if missing:
            logger.warning("Requested markers not found in data: %s", missing)
        marker_cols = available
    else:
        # Auto-detect: drop known non-marker columns
        marker_cols = [c for c in df.columns if c not in non_marker_cols]
        # If no labels, 'label' may not exist yet — that's fine
        marker_cols = [c for c in marker_cols if c != "label"]

    if exclude_markers:
        marker_cols = [c for c in marker_cols if c not in exclude_markers]

    logger.info("Loaded %d cells with %d markers.", len(df), len(marker_cols))

    # Ensure cell_id exists
    if "cell_id" not in df.columns:
        df.insert(0, "cell_id", [f"cell_{i}" for i in range(len(df))])

    # Ensure label column exists (NaN for unlabeled)
    if "label" not in df.columns:
        if label_col in df.columns and label_col != "label":
            df = df.rename(columns={label_col: "label"})
        else:
            df["label"] = np.nan

    # Reorder columns: cell_id, markers, label
    keep = ["cell_id"] + marker_cols + ["label"]
    df = df[[c for c in keep if c in df.columns]].copy()

    n_labeled = df["label"].notna().sum()
    n_unlabeled = df["label"].isna().sum()
    logger.info(
        "Cells: %d labeled, %d unlabeled (%d total).",
        n_labeled, n_unlabeled, len(df),
    )

    return df


def load_processed(
    path: str | Path,
    label_col: str = "label",
) -> pd.DataFrame:
    """Load a previously preprocessed dataset from an .h5ad or .csv file.

    Args:
        path: Path to the processed ``.h5ad`` or ``.csv`` file.
        label_col: Name of the label column.

    Returns:
        DataFrame with ``cell_id``, marker columns, and ``label``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        # Try .csv fallback
        csv_p = p.with_suffix(".csv")
        if csv_p.exists():
            logger.info("h5ad not found; loading CSV fallback: %s", csv_p)
            return _load_csv(csv_p, label_col)
        raise FileNotFoundError(
            f"Processed data not found at {p} (also tried {csv_p}). "
            "Run scripts/prepare_data.py first."
        )
    if p.suffix in (".h5ad",):
        return _load_h5ad(p, label_col)
    if p.suffix in (".csv",):
        return _load_csv(p, label_col)
    raise ValueError(f"Unsupported file format: {p.suffix!r}. Expected .h5ad or .csv.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_load_pycytodata(force_download: bool = False) -> Optional[pd.DataFrame]:
    """Try loading Levine32 via the PyCytoData package.

    PyCytoData downloads the dataset once and caches it locally.  Channel
    names (e.g. ``"CD45RA(La139)Di"``) are cleaned to base names
    (``"CD45RA"``).  Cells labelled ``"unassigned"`` are treated as unlabeled
    (``NaN``).

    Returns:
        Cleaned DataFrame or ``None`` if PyCytoData is not installed.
    """
    try:
        from PyCytoData import DataLoader  # type: ignore[import]
    except ImportError:
        logger.info("PyCytoData not installed — falling back to file-based loading.")
        return None

    logger.info("Loading Levine32 via PyCytoData (force_download=%s) ...", force_download)
    # Let exceptions propagate — PyCytoData is installed, so errors here are
    # real problems (network, permissions, etc.) that the caller should see.
    exprs = DataLoader.load_dataset(dataset="levine32", force_download=force_download)

    X = exprs.expression_matrix
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    X = np.asarray(X, dtype=np.float32)

    # Channel names
    raw_channels: list[str] = []
    if hasattr(exprs, "channels") and exprs.channels is not None:
        raw_channels = [str(c) for c in exprs.channels]
    else:
        raw_channels = [f"marker_{i}" for i in range(X.shape[1])]

    # Prefer lineage channels if available
    if (
        hasattr(exprs, "lineage_channels")
        and exprs.lineage_channels is not None
        and len(exprs.lineage_channels) > 0
    ):
        lineage_set = set(str(c) for c in exprs.lineage_channels)
        keep_idx = [i for i, c in enumerate(raw_channels) if c in lineage_set]
        if keep_idx:
            X = X[:, keep_idx]
            raw_channels = [raw_channels[i] for i in keep_idx]

    # Clean channel names: "CD45RA(La139)Di" → "CD45RA"
    clean_channels = [_clean_channel_name(c) for c in raw_channels]

    df = pd.DataFrame(X, columns=clean_channels)
    df.insert(0, "cell_id", [f"cell_{i}" for i in range(len(df))])

    # Cell-type labels
    labels = None
    if hasattr(exprs, "cell_types") and exprs.cell_types is not None:
        labels = pd.Series(exprs.cell_types, dtype=str)
        # "unassigned" → NaN (treated as unlabeled)
        labels = labels.where(labels != "unassigned", other=np.nan)

    df["label"] = labels.to_numpy() if labels is not None else np.nan

    logger.info(
        "PyCytoData loaded Levine32: %d cells × %d markers.",
        len(df), len(clean_channels),
    )
    return df


def _try_load(data_dir: Path, label_col: str) -> Optional[pd.DataFrame]:
    """Try to load data from a directory, returning None if nothing found."""
    # Priority 1: processed h5ad
    processed = data_dir / "levine32_processed.h5ad"
    if processed.exists():
        logger.info("Loading processed h5ad: %s", processed)
        return _load_h5ad(processed, label_col)

    # Priority 2: any h5ad
    h5ad_files = sorted(data_dir.glob("*.h5ad"))
    if h5ad_files:
        logger.info("Loading h5ad: %s", h5ad_files[0])
        return _load_h5ad(h5ad_files[0], label_col)

    # Priority 3: csv
    csv_files = sorted(data_dir.glob("*.csv"))
    if csv_files:
        logger.info("Loading CSV: %s", csv_files[0])
        return _load_csv(csv_files[0], label_col)

    # Priority 4: FCS files
    fcs_files = sorted(data_dir.glob("*.fcs"))
    if fcs_files:
        logger.info("Loading %d FCS file(s) from %s", len(fcs_files), data_dir)
        return _load_fcs_files(fcs_files, label_col)

    return None


def _load_h5ad(path: Path, label_col: str) -> pd.DataFrame:
    """Load an AnnData h5ad file into a DataFrame."""
    try:
        import anndata as ad
    except ImportError as e:
        raise ImportError("anndata is required to load .h5ad files. pip install anndata") from e

    adata = ad.read_h5ad(path)
    df = pd.DataFrame(
        adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray(),
        columns=list(adata.var_names),
    )

    # Restore _rank and _bin columns stored in obs
    for col in adata.obs.columns:
        if col.endswith("_rank") or col.endswith("_bin"):
            df[col] = adata.obs[col].values

    # Pull labels from obs
    if label_col in adata.obs.columns:
        df["label"] = adata.obs[label_col].values
    elif "cell_type" in adata.obs.columns:
        df["label"] = adata.obs["cell_type"].values
        logger.warning("label_col '%s' not found; using 'cell_type'.", label_col)
    else:
        df["label"] = np.nan
        logger.info("No label column found in obs; all cells treated as unlabeled.")

    # Cell IDs
    if len(adata.obs_names) > 0:
        df.insert(0, "cell_id", list(adata.obs_names))
    else:
        df.insert(0, "cell_id", [f"cell_{i}" for i in range(len(df))])

    return df


def _load_csv(path: Path, label_col: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    df = pd.read_csv(path)

    if "cell_id" not in df.columns:
        df.insert(0, "cell_id", [f"cell_{i}" for i in range(len(df))])

    if label_col not in df.columns and label_col != "label":
        logger.warning(
            "label_col '%s' not found in CSV. Cells treated as unlabeled.", label_col
        )
        df["label"] = np.nan
    elif label_col != "label" and label_col in df.columns:
        df = df.rename(columns={label_col: "label"})

    if "label" not in df.columns:
        df["label"] = np.nan

    return df


def _load_fcs_files(fcs_paths: list[Path], label_col: str) -> pd.DataFrame:
    """Load one or more FCS files and concatenate them."""
    try:
        import fcsparser
    except ImportError as e:
        raise ImportError(
            "fcsparser is required to load .fcs files. pip install fcsparser"
        ) from e

    frames: list[pd.DataFrame] = []
    for fcs_path in fcs_paths:
        logger.info("  Parsing FCS: %s", fcs_path.name)
        try:
            _meta, data = fcsparser.parse(str(fcs_path), reformat_meta=True)
        except Exception as exc:
            logger.warning("Could not parse FCS file %s: %s", fcs_path.name, exc)
            continue

        # Clean column names: strip channel suffixes like "(Yb176Di)"
        data.columns = [_clean_fcs_channel(c) for c in data.columns]
        data["source_file"] = fcs_path.stem
        frames.append(data)

    if not frames:
        raise ValueError("All FCS files failed to parse.")

    df = pd.concat(frames, ignore_index=True)

    # FCS files typically have no labels; infer from filename if possible
    df["label"] = _infer_fcs_labels(df, label_col)

    df.insert(0, "cell_id", [f"cell_{i}" for i in range(len(df))])

    return df


def _clean_fcs_channel(channel: str) -> str:
    """Strip mass/channel suffixes from FCS channel names.

    Example: 'CD3(110:114)Di' -> 'CD3'
    """
    import re
    # Remove parenthetical suffixes like "(Yb176Di)" or "(110:114)"
    cleaned = re.sub(r"\s*[\(\[].*?[\)\]].*$", "", channel).strip()
    # Remove trailing "Di", "Dd", etc.
    cleaned = re.sub(r"\s+(Di|Dd|Ei|Nd)\s*$", "", cleaned).strip()
    return cleaned if cleaned else channel


def _infer_fcs_labels(df: pd.DataFrame, label_col: str) -> pd.Series:
    """Attempt to infer labels from FCS metadata or source file names."""
    if label_col in df.columns:
        return df[label_col]
    return pd.Series([np.nan] * len(df), dtype=object)
