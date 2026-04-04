#!/usr/bin/env python3
"""Prepare raw CyTOF data: load, preprocess, and save as .h5ad.

Usage::

    python scripts/prepare_data.py --config configs/base.yaml
    python scripts/prepare_data.py --config configs/base.yaml \\
        --override preprocessing.arcsinh=false preprocessing.normalization=robust
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_levine32
from src.data.preprocessing import preprocess_from_config
from src.utils.config import load_config, merge_config_overrides
from src.utils.logging import setup_logging
from src.utils.reproducibility import log_environment, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare and preprocess CyTOF data.")
    p.add_argument("--config", default="configs/base.yaml", help="Base config YAML.")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Dot-path config overrides, e.g. preprocessing.arcsinh=false")
    p.add_argument("--output", default=None,
                   help="Output .h5ad path. Defaults to data/{dataset_name}_processed.h5ad")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    if args.override:
        cfg = merge_config_overrides(cfg, args.override)

    set_seed(cfg["seed"])
    log_environment()

    data_dir = Path(cfg["dataset"]["data_dir"])
    dataset_name = cfg["dataset"]["dataset_name"]

    logger.info("Loading dataset '%s' from %s ...", dataset_name, data_dir)
    df = load_levine32(data_dir=data_dir, label_col=cfg["dataset"]["label_col"])
    logger.info("Loaded %d cells with %d columns.", len(df), len(df.columns))

    logger.info("Preprocessing (arcsinh=%s, norm=%s) ...",
                cfg["preprocessing"]["arcsinh"],
                cfg["preprocessing"]["normalization"])
    df_processed, prep_log = preprocess_from_config(df, cfg["preprocessing"])

    # Save preprocessing log
    log_path = data_dir / f"{dataset_name}_preprocessing_log.json"
    with open(log_path, "w") as f:
        json.dump(prep_log, f, indent=2)
    logger.info("Preprocessing log saved to %s.", log_path)

    # Save processed data as .h5ad (via anndata) or .csv fallback
    out_path = Path(args.output) if args.output else data_dir / f"{dataset_name}_processed.h5ad"
    _save_processed(df_processed, out_path)
    logger.info("Processed data saved to %s.", out_path)


def _save_processed(df, path: Path) -> None:
    """Save processed DataFrame as .h5ad (preferred) or .csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import anndata as ad
        import numpy as np
        marker_cols = [c for c in df.columns if c not in ("cell_id", "label")]
        X = df[marker_cols].values.astype(np.float32)
        adata = ad.AnnData(
            X=X,
            obs=df[["cell_id", "label"]].set_index("cell_id"),
        )
        adata.var_names = marker_cols
        adata.write_h5ad(path)
    except ImportError:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logging.getLogger(__name__).warning(
            "anndata not installed — saved as CSV: %s", csv_path
        )


if __name__ == "__main__":
    main()
