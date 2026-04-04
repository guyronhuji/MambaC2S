#!/usr/bin/env python3
"""Evaluate a trained model: self-supervised loss, embedding quality, perturbation.

Usage::

    python scripts/evaluate_model.py \\
        --checkpoint outputs/transformer_rank_only_20240101_120000/best_checkpoint.pt \\
        --config configs/transformer.yaml

    # Evaluate only on labeled_test (final holdout — use sparingly)
    python scripts/evaluate_model.py \\
        --checkpoint outputs/.../best_checkpoint.pt \\
        --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_processed
from src.data.splits import load_manifest, apply_splits
from src.data.tokenization import tokenize_dataframe
from src.data.vocab import Vocabulary
from src.evaluation.metrics import (
    compute_loss_perplexity,
    extract_embeddings,
    compute_embedding_metrics,
    compute_umap,
)
from src.evaluation.perturbation import run_perturbation_analysis
from src.models.transformer import TransformerLM
from src.models.mamba_model import build_mamba_model
from src.training.trainer import CellSequenceDataset
from src.utils.config import load_config, merge_config_overrides
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained CyTOF sequence model.")
    p.add_argument("--checkpoint", required=True, help="Path to best_checkpoint.pt.")
    p.add_argument("--config", required=True, help="Experiment config YAML.")
    p.add_argument("--base-config", default="configs/base.yaml")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--split", default="val",
                   choices=["val", "test"],
                   help="'val' evaluates on val splits; 'test' on labeled_test.")
    p.add_argument("--skip-perturbation", action="store_true",
                   help="Skip the perturbation analysis (slow).")
    return p.parse_args()


def _load_model(checkpoint_path: Path, cfg: dict, vocab_size: int) -> torch.nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu")
    model_type = cfg["model"]["type"]
    mc = cfg["model"]

    if model_type == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=mc["d_model"],
            n_layers=mc["n_layers"],
            nhead=mc["nhead"],
            dropout=mc["dropout"],
            max_seq_len=mc["max_seq_len"],
        )
    elif model_type == "mamba":
        model = build_mamba_model(
            vocab_size=vocab_size,
            d_model=mc["d_model"],
            n_layers=mc["n_layers"],
            d_state=mc.get("d_state", 16),
            d_conv=mc.get("d_conv", 4),
            expand=mc.get("expand", 2),
            dropout=mc["dropout"],
            max_seq_len=mc["max_seq_len"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    model.load_state_dict(state["model_state_dict"])
    return model


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config(args.base_config, args.config)
    if args.override:
        cfg = merge_config_overrides(cfg, args.override)

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    checkpoint_path = Path(args.checkpoint)
    exp_dir = checkpoint_path.parent
    metrics_dir = exp_dir / "metrics"
    embeddings_dir = exp_dir / "embeddings"
    plots_dir = exp_dir / "plots"
    for d in (metrics_dir, embeddings_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(cfg["dataset"]["data_dir"])
    dataset_name = cfg["dataset"]["dataset_name"]
    processed_path = data_dir / f"{dataset_name}_processed.h5ad"

    # Try experiment-local manifest first
    local_manifest = exp_dir / "split_manifest.json"
    manifest_path = local_manifest if local_manifest.exists() else data_dir / "split_manifest.json"

    logger.info("Loading data from %s ...", processed_path)
    df = load_processed(processed_path)
    splits = load_manifest(manifest_path)
    split_dfs = apply_splits(df, splits)

    scheme = cfg["tokenization"]["scheme"]
    prep = cfg["preprocessing"]
    bins = prep["bins"]

    # Load vocab
    vocab_path = exp_dir / "vocab.json"
    vocab = Vocabulary.load(vocab_path)
    logger.info("Loaded vocab: %d tokens.", len(vocab))

    # Load model
    model = _load_model(checkpoint_path, cfg, vocab_size=len(vocab))
    model.to(device)
    logger.info("Model loaded: %r", model)

    max_seq = cfg["model"]["max_seq_len"]
    pad_id = vocab.pad_id
    eval_bs = cfg["evaluation"]["batch_size"]
    pooling = cfg["evaluation"]["embedding_pooling"]
    collate = partial(CellSequenceDataset.collate_fn, pad_id=pad_id)

    all_metrics: dict = {}

    # --- Self-supervised evaluation ---
    val_ss_df = split_dfs["val_self_supervised"]
    if len(val_ss_df) > 0:
        val_ss_seqs = tokenize_dataframe(val_ss_df, scheme=scheme, bins=bins)
        val_ss_ids = [vocab.encode(s) for s in val_ss_seqs]
        val_ss_ds = CellSequenceDataset(val_ss_ids, pad_id=pad_id, max_seq_len=max_seq)
        val_ss_loader = DataLoader(val_ss_ds, batch_size=eval_bs,
                                   shuffle=False, collate_fn=collate)
        ss_metrics = compute_loss_perplexity(model, val_ss_loader, device=device, pad_id=pad_id)
        logger.info("Self-supervised — loss=%.4f  perplexity=%.2f",
                    ss_metrics["loss"], ss_metrics["perplexity"])
        all_metrics["self_supervised"] = ss_metrics
        with open(metrics_dir / "self_supervised.json", "w") as f:
            json.dump(ss_metrics, f, indent=2)

    # --- Downstream embedding evaluation ---
    ds_split = "test" if args.split == "test" else "val_downstream"
    downstream_df = split_dfs.get(ds_split, split_dfs.get("labeled_val"))

    if downstream_df is not None and len(downstream_df) > 0:
        ds_seqs = tokenize_dataframe(downstream_df, scheme=scheme, bins=bins)
        ds_ids = [vocab.encode(s) for s in ds_seqs]
        ds_ds = CellSequenceDataset(ds_ids, pad_id=pad_id, max_seq_len=max_seq)
        ds_loader = DataLoader(ds_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate)

        logger.info("Extracting embeddings (%s, pooling=%s) ...", ds_split, pooling)
        embs = extract_embeddings(model, ds_loader, device=device, pooling=pooling)
        np.save(embeddings_dir / f"{ds_split}_embeddings.npy", embs)

        labels = downstream_df[cfg["dataset"]["label_col"]].tolist()
        emb_metrics = compute_embedding_metrics(
            embs, labels,
            knn_k=cfg["evaluation"]["knn_k"],
            seed=cfg["seed"],
        )
        all_metrics["downstream"] = emb_metrics
        with open(metrics_dir / "downstream.json", "w") as f:
            json.dump(emb_metrics, f, indent=2)

        # UMAP
        try:
            logger.info("Computing UMAP ...")
            coords = compute_umap(
                embs,
                n_neighbors=cfg["evaluation"]["n_umap_neighbors"],
                min_dist=cfg["evaluation"]["umap_min_dist"],
                seed=cfg["seed"],
            )
            np.save(embeddings_dir / f"{ds_split}_umap.npy", coords)
            _save_umap_plot(coords, labels, plots_dir / f"{ds_split}_umap.png")
        except Exception as exc:
            logger.warning("UMAP failed: %s", exc)

    # --- Perturbation analysis ---
    if not args.skip_perturbation:
        logger.info("Running perturbation analysis ...")
        all_seqs_raw = tokenize_dataframe(df, scheme=scheme, bins=bins)
        all_ids = [vocab.encode(s) for s in all_seqs_raw]
        pert_results = run_perturbation_analysis(
            model=model,
            sequences=all_ids,
            vocab=vocab.token2id,
            bins=bins,
            n_cells=200,
            n_perturbations_per_cell=5,
            device=device,
            pooling=pooling,
            seed=cfg["seed"],
            output_path=metrics_dir / "perturbation.json",
        )
        all_metrics["perturbation"] = {
            k: v for k, v in pert_results.items() if k != "per_cell"
        }

    # Write summary
    with open(metrics_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Evaluation complete. Metrics saved to %s.", metrics_dir)


def _save_umap_plot(coords: np.ndarray, labels: list, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        label_ids = le.fit_transform(labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=label_ids, s=2,
                             alpha=0.6, cmap="tab20")
        ax.set_title("UMAP of cell embeddings")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.colorbar(scatter, ax=ax, label="Cell type")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        logging.getLogger(__name__).warning("Plot failed: %s", exc)


if __name__ == "__main__":
    main()
