#!/usr/bin/env python3
"""Latent dimension sweep for MLPAutoencoder on Levine32 CyTOF data.

Trains one MLPAutoencoder per latent dim (default: 2 4 8 16 32) sequentially.
Skips a dim if its checkpoint already exists (use --force to retrain).
Checkpoints land in outputs/nb06_latent_sweep/latent_{d:02d}/ — the same
structure expected by notebooks/06_latent_dim_sweep.ipynb.

Usage:
    python scripts/latent_dim_sweep.py
    python scripts/latent_dim_sweep.py --latent-dims 8 16 32
    python scripts/latent_dim_sweep.py --device cuda --mixed-precision \\
        --epochs 100 --patience 15 --batch-size 512 --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_processed
from src.data.splits import load_manifest, apply_splits
from src.models.mlp_autoencoder import MLPAutoencoder
from src.training.trainer import Trainer, CellUnlabeledDataset
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed, log_environment, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLPAutoencoder latent dimension sweep on Levine32."
    )
    p.add_argument("--latent-dims", nargs="+", type=int,
                   default=[2, 4, 8, 16, 32],
                   metavar="D",
                   help="Latent dimensionalities to sweep (default: 2 4 8 16 32).")
    p.add_argument("--hidden-dims", nargs="+", type=int,
                   default=[256, 128],
                   metavar="H",
                   help="Encoder hidden layer sizes (default: 256 128).")
    p.add_argument("--epochs", type=int, default=100,
                   help="Max training epochs per dim (default: 100).")
    p.add_argument("--patience", type=int, default=15,
                   help="Early-stopping patience (default: 15).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mixed-precision", action="store_true",
                   help="Enable AMP (recommended on CUDA, avoid on MPS).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default 0; set 4+ on CUDA).")
    p.add_argument("--device", default=None,
                   help="Training device override (e.g. 'cuda', 'cpu'). "
                        "Default: reads from configs/base.yaml.")
    p.add_argument("--data-dir", default="data",
                   help="Directory containing levine32_processed.h5ad and split_manifest.json.")
    p.add_argument("--output-dir", default="outputs/nb06_latent_sweep",
                   help="Root output directory for all checkpoints.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Retrain even if checkpoint already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    set_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = resolve_device(args.device)
    else:
        base_cfg = load_config("configs/base.yaml")
        device   = resolve_device(base_cfg.get("device", "cpu"))
    logger.info("Device: %s", device)

    is_cuda = str(device).startswith("cuda")
    nw      = args.num_workers if is_cuda else 0

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir       = Path(args.data_dir)
    processed_path = data_dir / "levine32_processed.h5ad"
    manifest_path  = data_dir / "split_manifest.json"

    if not processed_path.exists():
        logger.error("Processed data not found: %s\nRun: python scripts/prepare_data.py",
                     processed_path)
        sys.exit(1)
    if not manifest_path.exists():
        logger.error("Split manifest not found: %s\nRun: python scripts/make_splits.py",
                     manifest_path)
        sys.exit(1)

    logger.info("Loading data from %s …", data_dir)
    df        = load_processed(processed_path)
    splits    = load_manifest(manifest_path)
    split_dfs = apply_splits(df, splits)

    # Marker columns (exclude metadata and derived columns)
    exclude = {"cell_id", "label"}
    marker_cols = [
        c for c in df.columns
        if c not in exclude
        and not c.endswith("_rank")
        and not c.endswith("_bin")
    ]
    n_markers = len(marker_cols)
    logger.info("n_markers=%d  train=%d  val_ss=%d",
                n_markers,
                len(split_dfs["train"]),
                len(split_dfs["val_self_supervised"]))

    # Unsupervised training: all cells (labeled + unlabeled), no labels
    train_arr = split_dfs["train"][marker_cols].values.astype(np.float32)
    val_arr   = split_dfs["val_self_supervised"][marker_cols].values.astype(np.float32)

    train_loader = DataLoader(
        CellUnlabeledDataset(train_arr),
        batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=is_cuda,
    )
    val_loader = DataLoader(
        CellUnlabeledDataset(val_arr),
        batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=is_cuda,
    )

    # ── Sweep ─────────────────────────────────────────────────────────────────
    output_root  = Path(args.output_dir)
    hidden_dims  = tuple(args.hidden_dims)
    all_summaries = []

    logger.info("=" * 60)
    logger.info("Latent dim sweep: %s", args.latent_dims)
    logger.info("Architecture   : n_markers→%s→d→%s→n_markers",
                "→".join(str(h) for h in hidden_dims),
                "→".join(str(h) for h in reversed(hidden_dims)))
    logger.info("Epochs / patience: %d / %d", args.epochs, args.patience)
    logger.info("=" * 60)

    for d in args.latent_dims:
        ckpt_dir  = output_root / f"latent_{d:02d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "best_checkpoint.pt"
        summ_path = ckpt_dir / "training_summary.json"

        if ckpt_path.exists() and summ_path.exists() and not args.force:
            logger.info("d=%2d: checkpoint exists — skipping (use --force to retrain).", d)
            with open(summ_path) as f:
                summary = json.load(f)
            all_summaries.append(summary)
            continue

        logger.info("")
        logger.info("── d_model=%d  (%d→%s→%d→%s→%d) ──",
                    d, n_markers,
                    "→".join(str(h) for h in hidden_dims), d,
                    "→".join(str(h) for h in reversed(hidden_dims)), n_markers)

        model = MLPAutoencoder(
            n_markers=n_markers,
            d_model=d,
            hidden_dims=hidden_dims,
        )
        logger.info("   params=%d", model.count_parameters())

        log_environment(ckpt_dir)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=ckpt_dir,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.epochs,
            patience=args.patience,
            grad_clip=args.grad_clip,
            mixed_precision=args.mixed_precision,
            device=device,
            mode="reconstruction",
        )

        result = trainer.train()
        result.update({
            "d_model":           d,
            "hidden_dims":       list(hidden_dims),
            "n_markers":         n_markers,
            "n_params":          model.count_parameters(),
            "supervision_type":  "unsupervised",
            "input_structure":   "vector",
            "training_objective": "reconstruction",
        })
        all_summaries.append(result)

        with open(summ_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info("   ✓  mse=%.5f  epoch=%d / %d",
                    result["best_val_loss"], result["best_epoch"], result["total_epochs"])

    # ── Print summary table ───────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SWEEP COMPLETE")
    logger.info("%-8s  %-10s  %-8s  %-8s", "d_model", "val_mse", "best_ep", "params")
    logger.info("%-8s  %-10s  %-8s  %-8s", "-------", "-------", "-------", "------")
    for s in sorted(all_summaries, key=lambda x: x["d_model"]):
        logger.info("%-8d  %-10.5f  %-8d  %-8d",
                    s["d_model"], s["best_val_loss"],
                    s["best_epoch"], s.get("n_params", 0))
    logger.info("=" * 60)
    logger.info("Checkpoints → %s", output_root)


if __name__ == "__main__":
    main()
