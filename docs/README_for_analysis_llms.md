# MambaC2S — Guide for Analysis LLMs

> This document is written **for AI assistants** tasked with analysing or
> interpreting results from this project.  It explains the project purpose,
> file layout, metric meanings, and how to compare experiments safely.

---

## 1. Project Purpose

This project compares two autoregressive sequence-model architectures —
**Transformer** (Cell2Sentence-style) and **Mamba** (Selective State Space
Model) — on CyTOF single-cell immunology data.

Each **cell** is represented as a token sequence (marker names ordered by
expression rank, optionally annotated with strength bins).  A language model
is trained to predict the next token in the sequence.  The resulting
**hidden states** are used as cell embeddings for downstream analysis.

The core scientific question: does **architecture** (Transformer vs Mamba) or
**tokenisation scheme** (rank-only, strength-only, hybrid) matter more for
learning useful cell representations?

---

## 2. File Layout

```
MambaC2S/
├── configs/              # YAML configs: base, transformer, mamba, dataset
├── data/                 # Raw and processed data (NOT committed)
│   ├── levine32_processed.h5ad    # preprocessed anndata
│   └── split_manifest.json        # cell-ID splits
├── src/                  # All source code
│   ├── data/             # loader, preprocessing, tokenisation, vocab, splits
│   ├── models/           # base, transformer, mamba_model
│   ├── training/         # trainer, CellSequenceDataset
│   ├── evaluation/       # metrics, perturbation
│   └── utils/            # config, logging, reproducibility
├── scripts/              # CLI pipeline scripts
├── outputs/              # One subdirectory per experiment
│   └── {EXP_ID}/
│       ├── config_resolved.yaml
│       ├── split_manifest.json
│       ├── vocab.json
│       ├── training_log.csv
│       ├── best_checkpoint.pt
│       ├── training_summary.json
│       ├── environment.json
│       ├── README_experiment.md
│       ├── metrics/
│       │   ├── self_supervised.json
│       │   ├── downstream.json
│       │   ├── perturbation.json
│       │   └── all_metrics.json
│       ├── embeddings/
│       │   ├── val_downstream_embeddings.npy
│       │   ├── val_downstream_umap.npy
│       │   └── test_embeddings.npy   (only after final eval)
│       └── plots/
│           └── val_downstream_umap.png
└── docs/                 # This and other documentation
```

---

## 3. Metric Meanings

### Self-supervised (language modelling)

| Metric | Meaning | Lower is better |
|--------|---------|-----------------|
| `loss` | Average cross-entropy per token on validation sequences | Yes |
| `perplexity` | exp(loss); model uncertainty per token | Yes |

### Downstream embedding quality

| Metric | Meaning | Higher is better |
|--------|---------|-----------------|
| `ARI` | Adjusted Rand Index between KMeans clusters and true labels. Range [−1, 1]; random ≈ 0 | Yes |
| `NMI` | Normalized Mutual Information. Range [0, 1]; 1 = perfect | Yes |
| `knn_purity` | Fraction of k=15 nearest neighbours sharing the same label. Range [0, 1] | Yes |

### Perturbation

| Metric | Meaning |
|--------|---------|
| `mean_cosine_distance` | Average cosine distance between original and perturbed cell embeddings. High value = model is sensitive to local token edits |
| `std_cosine_distance` | Variability of the above |

---

## 4. How to Compare Experiments

1. **Read `outputs/summary.csv`** for a ranked table of all experiments.
2. For a specific experiment, read `outputs/{EXP_ID}/config_resolved.yaml`
   to understand its exact settings.
3. Compare `metrics/downstream.json` across experiments for embedding quality.
4. Compare `metrics/self_supervised.json` for language modelling quality.
5. **Never compare experiments** that used different split manifests —
   check that `split_manifest.json` is identical (or that it was generated
   from the same seed and fractions).

---

## 5. What the Embeddings Represent

- `val_downstream_embeddings.npy`: shape `(n_cells, d_model)`, float32.
  Each row is one cell's embedding from the validation set (labeled_val).
- `val_downstream_umap.npy`: shape `(n_cells, 2)`, 2-D UMAP projection.
  Cell types that cluster together share similar representation in the model.

To load:
```python
import numpy as np
embs = np.load("outputs/EXP_ID/embeddings/val_downstream_embeddings.npy")
```

---

## 6. Key Design Decisions

- Test set (`labeled_test`) is held out.  **Do not use it for model selection.**
- All experiments share the same `split_manifest.json` (generated once from seed=42).
- Vocabularies are experiment-local (inside `outputs/{EXP_ID}/vocab.json`)
  because different tokenisation schemes produce different token sets.
- The Mamba model falls back to a pure-PyTorch SSM if `mamba_ssm` is not installed.
