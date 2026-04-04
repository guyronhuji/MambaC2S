#!/usr/bin/env python3
"""Run a complete experiment end-to-end: prepare → splits → train → evaluate.

This script chains all pipeline steps in the correct order.  It skips steps
whose outputs already exist (idempotent re-runs).

Usage::

    python scripts/run_full_experiment.py --config configs/transformer.yaml
    python scripts/run_full_experiment.py --config configs/mamba.yaml \\
        --exp-id my_mamba_run --force-retrain
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, merge_config_overrides
from src.utils.logging import setup_logging

SCRIPTS_DIR = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full CyTOF experiment pipeline.")
    p.add_argument("--config", required=True, help="Experiment config YAML.")
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--exp-id", default=None)
    p.add_argument("--force-retrain", action="store_true",
                   help="Re-run training even if checkpoint exists.")
    p.add_argument("--skip-perturbation", action="store_true")
    return p.parse_args()


def _run(cmd: list[str], logger: logging.Logger) -> None:
    """Run a subprocess command; raise on failure."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config(args.base_config, args.config)
    if args.override:
        cfg = merge_config_overrides(cfg, args.override)

    data_dir = Path(cfg["dataset"]["data_dir"])
    dataset_name = cfg["dataset"]["dataset_name"]

    # --- Step 1: Prepare data ---
    processed_path = data_dir / f"{dataset_name}_processed.h5ad"
    if not processed_path.exists():
        logger.info("=== Step 1: Preparing data ===")
        cmd = [sys.executable, str(SCRIPTS_DIR / "prepare_data.py"),
               "--config", args.base_config]
        if args.override:
            cmd += ["--override"] + args.override
        _run(cmd, logger)
    else:
        logger.info("Step 1 skipped — processed data already exists at %s.", processed_path)

    # --- Step 2: Make splits ---
    manifest_path = data_dir / "split_manifest.json"
    if not manifest_path.exists():
        logger.info("=== Step 2: Making splits ===")
        cmd = [sys.executable, str(SCRIPTS_DIR / "make_splits.py"),
               "--config", args.base_config]
        if args.override:
            cmd += ["--override"] + args.override
        _run(cmd, logger)
    else:
        logger.info("Step 2 skipped — split manifest already exists at %s.", manifest_path)

    # --- Step 3: Train ---
    logger.info("=== Step 3: Training ===")
    train_cmd = [sys.executable, str(SCRIPTS_DIR / "train_model.py"),
                 "--config", args.config,
                 "--base-config", args.base_config]
    if args.exp_id:
        train_cmd += ["--exp-id", args.exp_id]
    if args.override:
        train_cmd += ["--override"] + args.override
    _run(train_cmd, logger)

    # Discover the newly created experiment dir (latest modified)
    output_dir = Path(cfg["output"]["output_dir"])
    exp_dirs = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not exp_dirs:
        logger.error("No experiment directories found in %s after training.", output_dir)
        sys.exit(1)
    exp_dir = exp_dirs[0]
    checkpoint = exp_dir / "best_checkpoint.pt"
    logger.info("Latest experiment: %s", exp_dir)

    # --- Step 4: Evaluate ---
    logger.info("=== Step 4: Evaluating ===")
    eval_cmd = [sys.executable, str(SCRIPTS_DIR / "evaluate_model.py"),
                "--checkpoint", str(checkpoint),
                "--config", args.config,
                "--base-config", args.base_config]
    if args.skip_perturbation:
        eval_cmd.append("--skip-perturbation")
    if args.override:
        eval_cmd += ["--override"] + args.override
    _run(eval_cmd, logger)

    logger.info("=== Full experiment complete ===  Results in %s", exp_dir)


if __name__ == "__main__":
    main()
