"""Dataset splitting for CyTOF experiments.

Split policy:
  Labeled cells:   60% train / 20% val / 20% test  (stratified by label)
  Unlabeled cells: 90% train / 10% val

Training corpus = labeled_train + unlabeled_train (labels ignored during training).
Val corpus      = unlabeled_val (self-supervised) + labeled_val (downstream).
Test corpus     = labeled_test (held-out; use only for final evaluation).

The split manifest (cell IDs per split) is saved to split_manifest.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def make_splits(
    df: pd.DataFrame,
    labeled_train: float = 0.6,
    labeled_val: float = 0.2,
    labeled_test: float = 0.2,
    unlabeled_train: float = 0.9,
    unlabeled_val: float = 0.1,
    seed: int = 42,
    manifest_path: Optional[Path] = None,
) -> dict[str, list[str]]:
    """Create reproducible stratified train/val/test splits.

    Args:
        df: DataFrame with ``cell_id`` and ``label`` columns.
        labeled_train: Fraction of labeled cells for training.
        labeled_val: Fraction of labeled cells for validation.
        labeled_test: Fraction of labeled cells for held-out test.
        unlabeled_train: Fraction of unlabeled cells for training.
        unlabeled_val: Fraction of unlabeled cells for validation.
        seed: Random seed for reproducibility.
        manifest_path: If provided, save the manifest JSON here.

    Returns:
        Dictionary mapping split name to list of ``cell_id`` strings::

            {
                "labeled_train": [...],
                "labeled_val":   [...],
                "labeled_test":  [...],
                "unlabeled_train": [...],
                "unlabeled_val":   [...],
                "train":   [...],   # labeled_train + unlabeled_train
                "val_self_supervised": [...],   # unlabeled_val
                "val_downstream": [...],   # labeled_val
                "test":    [...],   # labeled_test
            }

    Raises:
        ValueError: If split fractions do not sum to 1.0 (within tolerance).
    """
    _validate_fractions(
        labeled_train, labeled_val, labeled_test,
        unlabeled_train, unlabeled_val,
    )

    labeled_mask = df["label"].notna()
    labeled_df = df[labeled_mask].copy()
    unlabeled_df = df[~labeled_mask].copy()

    logger.info(
        "Splitting %d labeled + %d unlabeled cells (seed=%d).",
        len(labeled_df), len(unlabeled_df), seed,
    )

    # --- Labeled split (stratified) ---
    labeled_ids, labeled_labels = (
        labeled_df["cell_id"].tolist(),
        labeled_df["label"].tolist(),
    )

    # First split off test
    test_size = labeled_test
    ids_trainval, ids_test = train_test_split(
        labeled_ids,
        test_size=test_size,
        stratify=labeled_labels,
        random_state=seed,
    )
    labels_trainval = labeled_df.set_index("cell_id").loc[ids_trainval, "label"].tolist()

    # Then split train/val
    val_size = labeled_val / (labeled_train + labeled_val)
    ids_labeled_train, ids_labeled_val = train_test_split(
        ids_trainval,
        test_size=val_size,
        stratify=labels_trainval,
        random_state=seed,
    )

    # --- Unlabeled split ---
    unlabeled_ids = unlabeled_df["cell_id"].tolist()
    if len(unlabeled_ids) > 0:
        ids_unlabeled_train, ids_unlabeled_val = train_test_split(
            unlabeled_ids,
            test_size=unlabeled_val,
            random_state=seed,
        )
    else:
        ids_unlabeled_train = []
        ids_unlabeled_val = []
        logger.warning("No unlabeled cells found; unlabeled splits are empty.")

    # --- Compose combined splits ---
    splits = {
        "labeled_train": sorted(ids_labeled_train),
        "labeled_val": sorted(ids_labeled_val),
        "labeled_test": sorted(ids_test),
        "unlabeled_train": sorted(ids_unlabeled_train),
        "unlabeled_val": sorted(ids_unlabeled_val),
        "train": sorted(ids_labeled_train + ids_unlabeled_train),
        "val_self_supervised": sorted(ids_unlabeled_val),
        "val_downstream": sorted(ids_labeled_val),
        "test": sorted(ids_test),
    }

    _log_split_summary(splits, labeled_df)

    if manifest_path is not None:
        _save_manifest(splits, manifest_path, df)

    return splits


def load_manifest(path: str | Path) -> dict[str, list[str]]:
    """Load a split manifest from JSON.

    Args:
        path: Path to ``split_manifest.json``.

    Returns:
        Dictionary mapping split name to list of cell IDs.
    """
    with open(path) as f:
        data = json.load(f)
    return data["splits"]


def apply_splits(
    df: pd.DataFrame,
    splits: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """Filter a DataFrame according to split manifests.

    Args:
        df: Full DataFrame with ``cell_id`` column.
        splits: Split manifest from :func:`make_splits` or :func:`load_manifest`.

    Returns:
        Dictionary mapping split name to subset DataFrame.
    """
    cell_id_to_idx = {cid: i for i, cid in enumerate(df["cell_id"])}
    result: dict[str, pd.DataFrame] = {}
    for name, ids in splits.items():
        valid_ids = [cid for cid in ids if cid in cell_id_to_idx]
        if len(valid_ids) < len(ids):
            logger.warning(
                "Split '%s': %d cell IDs not found in DataFrame (skipped).",
                name, len(ids) - len(valid_ids),
            )
        result[name] = df.loc[df["cell_id"].isin(valid_ids)].copy()
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_fractions(
    labeled_train: float,
    labeled_val: float,
    labeled_test: float,
    unlabeled_train: float,
    unlabeled_val: float,
) -> None:
    labeled_sum = labeled_train + labeled_val + labeled_test
    if not np.isclose(labeled_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Labeled split fractions must sum to 1.0, got {labeled_sum:.4f}"
        )
    unlabeled_sum = unlabeled_train + unlabeled_val
    if not np.isclose(unlabeled_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Unlabeled split fractions must sum to 1.0, got {unlabeled_sum:.4f}"
        )


def _log_split_summary(
    splits: dict[str, list[str]], labeled_df: pd.DataFrame
) -> None:
    logger.info("Split summary:")
    for name, ids in splits.items():
        logger.info("  %-30s: %d cells", name, len(ids))

    # Show label distribution in train/val/test
    label_index = labeled_df.set_index("cell_id")["label"]
    for split_name in ("labeled_train", "labeled_val", "labeled_test"):
        ids = splits[split_name]
        labels = label_index.loc[[i for i in ids if i in label_index.index]]
        counts = labels.value_counts()
        logger.info("  %s label distribution: %s", split_name, counts.to_dict())


def _save_manifest(
    splits: dict[str, list[str]],
    path: Path,
    df: pd.DataFrame,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Include label lookup for convenience
    label_map = df.set_index("cell_id")["label"].to_dict()

    manifest = {
        "version": 1,
        "n_cells_total": len(df),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "splits": splits,
        "label_map": {
            k: (str(v) if v == v else None)  # NaN -> None
            for k, v in label_map.items()
        },
    }

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Split manifest saved to %s.", path)
