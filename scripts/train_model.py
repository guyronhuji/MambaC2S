#!/usr/bin/env python3
"""Train a Transformer or Mamba model on CyTOF token sequences.

Usage::

    python scripts/train_model.py --config configs/transformer.yaml
    python scripts/train_model.py --config configs/mamba.yaml \\
        --override model.d_model=64 training.max_epochs=50
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import pickle
import sys
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_processed
from src.data.splits import load_manifest, apply_splits
from src.data.tokenization import tokenize_dataframe
from src.data.vocab import Vocabulary
from src.models.transformer import TransformerLM
from src.models.mamba_model import build_mamba_model
from src.training.trainer import Trainer, CellSequenceDataset
from src.utils.config import load_config, merge_config_overrides, save_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed, log_environment, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train sequence model on CyTOF data.")
    p.add_argument("--config", required=True, help="Experiment config YAML.")
    p.add_argument("--base-config", default="configs/base.yaml",
                   help="Base config YAML (merged before experiment config).")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--exp-id", default=None,
                   help="Experiment ID. Auto-generated if not provided.")
    return p.parse_args()


def _compute_cache_key(cfg: dict, manifest_path: Path) -> str:
    """Hash the data/tokenization/preprocessing config + manifest for cache invalidation."""
    h = hashlib.sha256()
    payload = {
        "dataset": cfg["dataset"],
        "tokenization": cfg["tokenization"],
        "preprocessing": cfg["preprocessing"],
    }
    h.update(json.dumps(payload, sort_keys=True).encode())
    if manifest_path.exists():
        h.update(manifest_path.read_bytes())
    return h.hexdigest()[:16]


def _make_exp_id(cfg: dict) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = cfg["model"]["type"]
    scheme = cfg["tokenization"]["scheme"]
    return f"{model_type}_{scheme}_{ts}"


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config(args.base_config, args.config)
    if args.override:
        cfg = merge_config_overrides(cfg, args.override)

    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "cpu"))

    exp_id = args.exp_id or cfg["output"].get("exp_id") or _make_exp_id(cfg)
    output_dir = Path(cfg["output"]["output_dir"]) / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment ID: %s  →  %s", exp_id, output_dir)

    # Save resolved config
    save_config(cfg, output_dir / "config_resolved.yaml")
    log_environment(output_dir)

    # Load data (with caching to skip expensive reprocessing on repeated runs)
    data_dir = Path(cfg["dataset"]["data_dir"])
    dataset_name = cfg["dataset"]["dataset_name"]
    processed_path = data_dir / f"{dataset_name}_processed.h5ad"
    manifest_path = data_dir / "split_manifest.json"

    cache_key = _compute_cache_key(cfg, manifest_path)
    cache_dir = data_dir / ".cache" / cache_key
    cache_train = cache_dir / "train_ids.pkl"
    cache_val = cache_dir / "val_ids.pkl"
    cache_vocab = cache_dir / "vocab.json"

    if cache_train.exists() and cache_val.exists() and cache_vocab.exists():
        logger.info("Cache hit (key=%s) — loading tokenised data from %s.", cache_key, cache_dir)
        with open(cache_train, "rb") as f:
            train_ids = pickle.load(f)
        with open(cache_val, "rb") as f:
            val_ids = pickle.load(f)
        vocab = Vocabulary.load(cache_vocab)
        logger.info("Loaded %d train / %d val sequences, vocab=%d tokens (from cache).",
                    len(train_ids), len(val_ids), len(vocab))
    else:
        logger.info("Cache miss (key=%s) — loading and tokenising data ...", cache_key)
        df = load_processed(processed_path)
        splits = load_manifest(manifest_path)
        split_dfs = apply_splits(df, splits)

        train_df = split_dfs["train"]
        val_df = split_dfs["val_self_supervised"]

        scheme = cfg["tokenization"]["scheme"]
        prep = cfg["preprocessing"]
        logger.info("Tokenising with scheme '%s' ...", scheme)
        train_seqs = tokenize_dataframe(train_df, scheme=scheme, bins=prep["bins"])
        val_seqs = tokenize_dataframe(val_df, scheme=scheme, bins=prep["bins"])

        vocab = Vocabulary()
        vocab.build(train_seqs)
        logger.info("Vocabulary built: %d tokens.", len(vocab))

        train_ids = [vocab.encode(s) for s in train_seqs]
        val_ids = [vocab.encode(s) for s in val_seqs]

        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_train, "wb") as f:
            pickle.dump(train_ids, f)
        with open(cache_val, "wb") as f:
            pickle.dump(val_ids, f)
        vocab.save(cache_vocab)
        logger.info("Tokenised data cached to %s.", cache_dir)

    # Save vocab to experiment output dir and copy manifest
    vocab_path = output_dir / "vocab.json"
    vocab.save(vocab_path)

    import shutil
    shutil.copy(manifest_path, output_dir / "split_manifest.json")

    # Datasets & loaders
    max_seq = cfg["model"]["max_seq_len"]
    pad_id = vocab.pad_id

    train_ds = CellSequenceDataset(train_ids, pad_id=pad_id, max_seq_len=max_seq)
    val_ds = CellSequenceDataset(val_ids, pad_id=pad_id, max_seq_len=max_seq)

    collate = partial(CellSequenceDataset.collate_fn, pad_id=pad_id)
    bs = cfg["training"]["batch_size"]

    # num_workers > 0 causes fork issues on macOS (MPS); pin_memory only helps CUDA
    device_type = cfg["training"].get("device", "auto")
    if device_type == "auto":
        import torch as _torch
        device_type = "cuda" if _torch.cuda.is_available() else (
            "mps" if _torch.backends.mps.is_available() else "cpu"
        )
    is_cuda = device_type == "cuda"
    nw = cfg["training"].get("num_workers", 0) if is_cuda else 0

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate, num_workers=nw, pin_memory=is_cuda)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            collate_fn=collate, num_workers=nw, pin_memory=is_cuda)

    # Build model
    model_cfg = cfg["model"]
    vocab_size = len(vocab)
    model_type = model_cfg["type"]
    logger.info("Building %s model (vocab_size=%d) ...", model_type, vocab_size)

    if model_type == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=model_cfg["d_model"],
            n_layers=model_cfg["n_layers"],
            nhead=model_cfg["nhead"],
            dropout=model_cfg["dropout"],
            max_seq_len=model_cfg["max_seq_len"],
        )
    elif model_type == "mamba":
        model = build_mamba_model(
            vocab_size=vocab_size,
            d_model=model_cfg["d_model"],
            n_layers=model_cfg["n_layers"],
            d_state=model_cfg.get("d_state", 16),
            d_conv=model_cfg.get("d_conv", 4),
            expand=model_cfg.get("expand", 2),
            dropout=model_cfg["dropout"],
            max_seq_len=model_cfg["max_seq_len"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    logger.info("Model: %r", model)

    # Train
    tr_cfg = cfg["training"]
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
        max_epochs=tr_cfg["max_epochs"],
        patience=tr_cfg["patience"],
        grad_clip=tr_cfg["grad_clip"],
        mixed_precision=tr_cfg.get("mixed_precision", False),
        device=device,
        pad_id=pad_id,
    )

    results = trainer.train()
    logger.info("Training complete. Best val_loss=%.4f at epoch %d.",
                results["best_val_loss"], results["best_epoch"])

    # Save training summary
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Write per-experiment README
    _write_experiment_readme(output_dir, exp_id, cfg, results)
    logger.info("Experiment saved to %s.", output_dir)


def _write_experiment_readme(
    output_dir: Path,
    exp_id: str,
    cfg: dict,
    results: dict,
) -> None:
    readme = f"""# Experiment: {exp_id}

## Config Summary

| Key | Value |
|-----|-------|
| model_type | {cfg["model"]["type"]} |
| tokenization | {cfg["tokenization"]["scheme"]} |
| d_model | {cfg["model"]["d_model"]} |
| n_layers | {cfg["model"]["n_layers"]} |
| lr | {cfg["training"]["lr"]} |
| batch_size | {cfg["training"]["batch_size"]} |

## Training Results

| Metric | Value |
|--------|-------|
| best_val_loss | {results["best_val_loss"]:.4f} |
| best_epoch | {results["best_epoch"]} |
| total_epochs | {results["total_epochs"]} |

## Files

- `config_resolved.yaml` — full resolved config
- `best_checkpoint.pt` — best model weights
- `training_log.csv` — per-epoch loss/perplexity
- `metrics/` — evaluation results
- `embeddings/` — numpy cell embeddings
- `plots/` — UMAP and other visualisations
"""
    (output_dir / "README_experiment.md").write_text(readme)


if __name__ == "__main__":
    main()
