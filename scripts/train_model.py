#!/usr/bin/env python3
"""Train a model on CyTOF data.

Sequence models (Transformer, LSTM, GRU):
    python scripts/train_model.py --config configs/transformer_hybrid.yaml
    python scripts/train_model.py --config configs/lstm_rank_only.yaml

Vector models (MLP, DeepSets):
    python scripts/train_model.py --config configs/mlp_raw.yaml
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import pickle
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
from src.models import build_model, build_vector_model
from src.training.trainer import Trainer, CellSequenceDataset, CellVectorDataset, CellUnlabeledDataset
from src.utils.config import load_config, merge_config_overrides, save_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed, log_environment, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a model on CyTOF data.")
    p.add_argument("--config", required=True, help="Experiment config YAML.")
    p.add_argument("--base-config", default="configs/base.yaml",
                   help="Base config YAML (merged before experiment config).")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--exp-id", default=None,
                   help="Experiment ID. Auto-generated if not provided.")
    return p.parse_args()


def _compute_cache_key(cfg: dict, manifest_path: Path) -> str:
    """Hash data/tokenization/preprocessing config + manifest for cache invalidation."""
    h = hashlib.sha256()
    payload = {
        "dataset": cfg.get("dataset", {}),
        "tokenization": cfg.get("tokenization", {}),
        "preprocessing": cfg.get("preprocessing", {}),
    }
    h.update(json.dumps(payload, sort_keys=True).encode())
    if manifest_path.exists():
        h.update(manifest_path.read_bytes())
    return h.hexdigest()[:16]


def _make_exp_id(cfg: dict) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = cfg["model"]["type"]
    data_mode = cfg["model"].get("data_mode", "sequence")
    if data_mode in ("vector", "reconstruction"):
        return f"{model_type}_raw_{ts}"
    scheme = cfg.get("tokenization", {}).get("scheme", "raw")
    return f"{model_type}_{scheme}_{ts}"


def _get_metadata(cfg: dict, data_mode: str) -> dict:
    """Return supervision/structure/objective metadata for training_summary.json."""
    model_type = cfg["model"]["type"]
    if data_mode == "sequence":
        return {
            "supervision_type": "unsupervised",
            "input_structure": "sequence",
            "training_objective": "next_token",
        }
    elif data_mode == "vector":
        return {
            "supervision_type": "supervised",
            "input_structure": "vector",
            "training_objective": "classification",
        }
    else:  # reconstruction
        input_structure = "set" if "deepsets" in model_type else "vector"
        return {
            "supervision_type": "unsupervised",
            "input_structure": input_structure,
            "training_objective": "reconstruction",
        }


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

    save_config(cfg, output_dir / "config_resolved.yaml")
    log_environment(output_dir)

    data_dir = Path(cfg["dataset"]["data_dir"])
    dataset_name = cfg["dataset"]["dataset_name"]
    processed_path = data_dir / f"{dataset_name}_processed.h5ad"
    manifest_path = data_dir / "split_manifest.json"

    data_mode = cfg["model"].get("data_mode", "sequence")
    model_type = cfg["model"]["type"]
    bs = cfg["training"]["batch_size"]

    # device type for DataLoader flags
    device_str = str(device)
    is_cuda = device_str.startswith("cuda")
    nw = cfg["training"].get("num_workers", 0) if is_cuda else 0

    # ------------------------------------------------------------------ #
    # Sequence mode — tokenised autoregressive training                    #
    # ------------------------------------------------------------------ #
    if data_mode == "sequence":
        cache_key = _compute_cache_key(cfg, manifest_path)
        cache_dir = data_dir / ".cache" / cache_key
        cache_train = cache_dir / "train_ids.pkl"
        cache_val = cache_dir / "val_ids.pkl"
        cache_vocab = cache_dir / "vocab.json"

        if cache_train.exists() and cache_val.exists() and cache_vocab.exists():
            logger.info("Cache hit (key=%s) — loading from %s.", cache_key, cache_dir)
            with open(cache_train, "rb") as f:
                train_ids = pickle.load(f)
            with open(cache_val, "rb") as f:
                val_ids = pickle.load(f)
            vocab = Vocabulary.load(cache_vocab)
            logger.info("Loaded %d train / %d val sequences, vocab=%d (cache).",
                        len(train_ids), len(val_ids), len(vocab))
        else:
            logger.info("Cache miss (key=%s) — tokenising data ...", cache_key)
            df = load_processed(processed_path)
            splits = load_manifest(manifest_path)
            split_dfs = apply_splits(df, splits)

            scheme = cfg["tokenization"]["scheme"]
            prep = cfg["preprocessing"]
            logger.info("Tokenising with scheme '%s' ...", scheme)
            train_seqs = tokenize_dataframe(split_dfs["train"], scheme=scheme, bins=prep["bins"])
            val_seqs = tokenize_dataframe(split_dfs["val_self_supervised"], scheme=scheme, bins=prep["bins"])

            vocab = Vocabulary()
            vocab.build(train_seqs)
            logger.info("Vocabulary built: %d tokens.", len(vocab))

            train_ids = [vocab.encode(s) for s in train_seqs]
            val_ids = [vocab.encode(s) for s in val_seqs]

            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_train, "wb") as f:
                pickle.dump(train_ids, f)
            with open(cache_val, "wb") as f:
                pickle.dump(val_ids, f)
            vocab.save(cache_vocab)
            logger.info("Tokenised data cached to %s.", cache_dir)

        vocab.save(output_dir / "vocab.json")

        import shutil
        shutil.copy(manifest_path, output_dir / "split_manifest.json")

        max_seq = cfg["model"]["max_seq_len"]
        pad_id = vocab.pad_id

        train_ds = CellSequenceDataset(train_ids, pad_id=pad_id, max_seq_len=max_seq)
        val_ds = CellSequenceDataset(val_ids, pad_id=pad_id, max_seq_len=max_seq)
        collate = partial(CellSequenceDataset.collate_fn, pad_id=pad_id)

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  collate_fn=collate, num_workers=nw, pin_memory=is_cuda)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                collate_fn=collate, num_workers=nw, pin_memory=is_cuda)

        vocab_size = len(vocab)
        logger.info("Building %s model (vocab_size=%d) ...", model_type, vocab_size)
        model = build_model(cfg, vocab_size)
        trainer_kwargs = dict(pad_id=pad_id, mode="sequence")

    # ------------------------------------------------------------------ #
    # Vector mode — supervised classification on raw marker values         #
    # ------------------------------------------------------------------ #
    elif data_mode == "vector":
        logger.info("Vector mode: loading labeled data for supervised training ...")
        df = load_processed(processed_path)
        splits = load_manifest(manifest_path)
        split_dfs = apply_splits(df, splits)

        train_df = split_dfs["labeled_train"]
        val_df = split_dfs["val_downstream"]

        # Marker columns: all except cell_id, label, *_rank, *_bin
        exclude = {"cell_id", "label"}
        marker_cols = [
            c for c in df.columns
            if c not in exclude and not c.endswith("_rank") and not c.endswith("_bin")
        ]
        n_markers = len(marker_cols)
        logger.info("Marker columns (%d): %s", n_markers, marker_cols[:5])

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        train_labels = le.fit_transform(train_df["label"].values)
        val_labels = le.transform(val_df["label"].values)
        n_classes = len(le.classes_)
        logger.info("n_classes=%d  train=%d  val=%d", n_classes, len(train_labels), len(val_labels))

        # Save label encoder mapping for downstream use
        label_map = {int(i): str(cls) for i, cls in enumerate(le.classes_)}
        with open(output_dir / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

        train_markers = train_df[marker_cols].values.astype(np.float32)
        val_markers = val_df[marker_cols].values.astype(np.float32)

        train_ds = CellVectorDataset(train_markers, train_labels)
        val_ds = CellVectorDataset(val_markers, val_labels)

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=is_cuda)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=is_cuda)

        logger.info("Building %s model (n_markers=%d, n_classes=%d) ...",
                    model_type, n_markers, n_classes)
        model = build_vector_model(cfg, n_markers=n_markers, n_classes=n_classes)
        trainer_kwargs = dict(pad_id=0, mode="vector")

    # ------------------------------------------------------------------ #
    # Reconstruction mode — unsupervised autoencoder on all training cells  #
    # ------------------------------------------------------------------ #
    elif data_mode == "reconstruction":
        logger.info("Reconstruction mode: loading all training cells (labeled + unlabeled) ...")
        df = load_processed(processed_path)
        splits = load_manifest(manifest_path)
        split_dfs = apply_splits(df, splits)

        train_df = split_dfs["train"]          # labeled_train + unlabeled_train
        val_df = split_dfs["val_self_supervised"]

        # Marker columns: all except cell_id, label, *_rank, *_bin
        exclude = {"cell_id", "label"}
        marker_cols = [
            c for c in df.columns
            if c not in exclude and not c.endswith("_rank") and not c.endswith("_bin")
        ]
        n_markers = len(marker_cols)
        logger.info("Marker columns (%d): %s", n_markers, marker_cols[:5])
        logger.info("train=%d  val=%d", len(train_df), len(val_df))

        train_markers = train_df[marker_cols].values.astype(np.float32)
        val_markers = val_df[marker_cols].values.astype(np.float32)

        train_ds = CellUnlabeledDataset(train_markers)
        val_ds = CellUnlabeledDataset(val_markers)

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=is_cuda)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=is_cuda)

        logger.info("Building %s model (n_markers=%d) ...", model_type, n_markers)
        model = build_vector_model(cfg, n_markers=n_markers, n_classes=0)
        trainer_kwargs = dict(pad_id=0, mode="reconstruction")

    logger.info("Model: %r", model)

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #
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
        **trainer_kwargs,
    )

    results = trainer.train()
    logger.info("Training complete. Best val_loss=%.4f at epoch %d.",
                results["best_val_loss"], results["best_epoch"])

    results.update(_get_metadata(cfg, data_mode))

    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    _write_experiment_readme(output_dir, exp_id, cfg, results, data_mode)
    logger.info("Experiment saved to %s.", output_dir)


def _write_experiment_readme(
    output_dir: Path,
    exp_id: str,
    cfg: dict,
    results: dict,
    data_mode: str = "sequence",
) -> None:
    scheme = cfg.get("tokenization", {}).get("scheme", "N/A") if data_mode == "sequence" else "raw (vector)"
    readme = f"""# Experiment: {exp_id}

## Config Summary

| Key | Value |
|-----|-------|
| model_type | {cfg["model"]["type"]} |
| data_mode | {data_mode} |
| tokenization | {scheme} |
| d_model | {cfg["model"].get("d_model", "N/A")} |
| n_layers | {cfg["model"].get("n_layers", "N/A")} |
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
- `training_log.csv` — per-epoch loss/metric
- `metrics/` — evaluation results
- `embeddings/` — numpy cell embeddings
"""
    (output_dir / "README_experiment.md").write_text(readme)


if __name__ == "__main__":
    main()
