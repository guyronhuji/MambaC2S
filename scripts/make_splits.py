#!/usr/bin/env python3
"""Create reproducible train/val/test splits from processed data.

Usage::

    python scripts/make_splits.py --config configs/base.yaml
    python scripts/make_splits.py --config configs/base.yaml --output data/splits/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_processed
from src.data.splits import make_splits
from src.utils.config import load_config, merge_config_overrides
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create dataset splits.")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--output", default=None,
                   help="Directory to write split_manifest.json. Defaults to data/.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    if args.override:
        cfg = merge_config_overrides(cfg, args.override)

    set_seed(cfg["seed"])

    data_dir = Path(cfg["dataset"]["data_dir"])
    dataset_name = cfg["dataset"]["dataset_name"]
    processed_path = data_dir / f"{dataset_name}_processed.h5ad"

    logger.info("Loading processed data from %s ...", processed_path)
    df = load_processed(processed_path)
    logger.info("Loaded %d cells.", len(df))

    out_dir = Path(args.output) if args.output else data_dir
    manifest_path = out_dir / "split_manifest.json"

    sp = cfg["splits"]
    splits = make_splits(
        df=df,
        labeled_train=sp["labeled_train"],
        labeled_val=sp["labeled_val"],
        labeled_test=sp["labeled_test"],
        unlabeled_train=sp["unlabeled_train"],
        unlabeled_val=sp["unlabeled_val"],
        seed=cfg["seed"],
        manifest_path=manifest_path,
    )

    logger.info("Splits created and saved to %s.", manifest_path)
    for name, ids in splits.items():
        logger.info("  %-30s: %d cells", name, len(ids))


if __name__ == "__main__":
    main()
