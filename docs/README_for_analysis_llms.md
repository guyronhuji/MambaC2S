# MambaC2S — Guide for Analysis LLMs

> This document is written **for AI assistants** tasked with analysing or
> interpreting results from this project.  It explains the project purpose,
> file layout, metric meanings, and how to compare experiments safely.

---

## 1. Project Purpose

This project compares multiple architectures on Levine32 CyTOF single-cell
immunology data (265,627 cells × 32 markers, 14 cell types).

Two training paradigms are evaluated side by side:

| Paradigm | Models | Data | Objective |
|----------|--------|------|-----------|
| **Sequence (self-supervised)** | Transformer, LSTM, GRU | Tokenised marker sequences | Next-token prediction |
| **Vector (supervised)** | MLP classifier | Raw float marker vectors | Cross-entropy vs cell labels |
| **Vector (unsupervised)** | MLP autoencoder, DeepSets autoencoder | Raw float marker vectors | MSE reconstruction |

The core scientific question: which architecture and tokenisation scheme produces
the best **cell embeddings** for downstream analysis (ARI, NMI, kNN purity)?

### Fair Comparison Note

Sequence models and vector autoencoders are both **unsupervised** — they use no
cell-type labels during training.  `mlp_raw` (MLPClassifier) is the **supervised
upper bound** and should be compared separately from the unsupervised models.
The metadata field `supervision_type` in `training_summary.json` encodes this:
`"supervised"` vs `"unsupervised"`.

---

## 2. Model Inventory

| Config | Model type | Supervision | Input | Objective |
|--------|-----------|-------------|-------|-----------|
| `transformer_{scheme}.yaml` | TransformerLM | unsupervised | sequence | next_token |
| `lstm_{scheme}.yaml` | LSTMLanguageModel | unsupervised | sequence | next_token |
| `gru_{scheme}.yaml` | GRULanguageModel | unsupervised | sequence | next_token |
| `mlp_raw.yaml` | MLPClassifier | **supervised** | vector | classification |
| `deepsets_raw.yaml` | DeepSetsClassifier | **supervised** | set | classification |
| `mlp_autoencoder_raw.yaml` | MLPAutoencoder | unsupervised | vector | reconstruction |
| `deepsets_autoencoder_raw.yaml` | DeepSetsAutoencoder | unsupervised | set | reconstruction |

Schemes: `rank_only`, `strength_only`, `hybrid`.

---

## 3. File Layout

```
MambaC2S/
├── configs/              # YAML configs (13 experiments total)
├── data/                 # Raw and processed data (NOT committed)
│   ├── levine32_processed.h5ad    # preprocessed anndata
│   └── split_manifest.json        # cell-ID splits
├── src/
│   ├── data/             # loader, preprocessing, tokenisation, vocab, splits
│   ├── models/           # base, transformer, lstm, gru, mlp, deepsets,
│   │                     # mlp_autoencoder, deepsets_autoencoder
│   ├── training/         # Trainer (modes: sequence / vector / reconstruction)
│   ├── evaluation/       # metrics, perturbation
│   └── utils/            # config, logging, reproducibility
├── scripts/              # CLI pipeline scripts
├── outputs/              # One subdirectory per experiment
│   └── {EXP_ID}/
│       ├── config_resolved.yaml
│       ├── training_summary.json    ← includes supervision_type, input_structure, training_objective
│       ├── training_log.csv
│       ├── best_checkpoint.pt
│       ├── environment.json
│       ├── metrics/
│       │   ├── downstream.json      ← ARI, NMI, kNN purity
│       │   └── self_supervised.json ← loss, perplexity (sequence models only)
│       └── embeddings/
│           └── val_downstream_embeddings.npy
└── docs/                 # This and other documentation
```

---

## 4. Metric Meanings

### Self-supervised (sequence models only)

| Metric | Meaning | Lower is better |
|--------|---------|-----------------|
| `loss` | Average cross-entropy per token on validation sequences | Yes |
| `perplexity` | exp(loss); model uncertainty per token | Yes |

### Reconstruction loss (vector autoencoders only)

| Metric | Meaning | Lower is better |
|--------|---------|-----------------|
| `best_val_loss` | Mean squared error on held-out validation cells | Yes |

### Downstream embedding quality (all models)

| Metric | Meaning | Higher is better |
|--------|---------|-----------------|
| `ARI` | Adjusted Rand Index between KMeans clusters and true labels. Range [−1, 1]; random ≈ 0 | Yes |
| `NMI` | Normalized Mutual Information. Range [0, 1]; 1 = perfect | Yes |
| `knn_purity` | Fraction of k=15 nearest neighbours sharing the same label. Range [0, 1] | Yes |

---

## 5. How to Compare Experiments

1. **Read `outputs/summary.csv`** for a ranked table of all experiments.
2. For a specific experiment, read `outputs/{EXP_ID}/training_summary.json` —
   this includes `supervision_type`, `input_structure`, `training_objective`.
3. **Compare only within supervision tier**:
   - Unsupervised: sequence models + vector autoencoders
   - Supervised upper bound: MLPClassifier, DeepSetsClassifier
4. Compare `metrics/downstream.json` across experiments for embedding quality.
5. **Never compare experiments** that used different split manifests.

---

## 6. What the Embeddings Represent

- `val_downstream_embeddings.npy`: shape `(n_cells, d_model)`, float32.
  Each row is one cell's embedding from the labeled validation set.
- All models expose an `encode()` method that returns these embeddings.

To load:
```python
import numpy as np
embs = np.load("outputs/EXP_ID/embeddings/val_downstream_embeddings.npy")
```

---

## 7. Key Design Decisions

- Test set (`labeled_test`) is held out.  **Do not use it for model selection.**
- All experiments share the same `split_manifest.json` (generated once from seed=42).
- Sequence models are trained on `train` (labeled_train + unlabeled_train) and
  validated on `val_self_supervised`.
- Vector classifiers are trained on `labeled_train` and validated on `val_downstream`.
- Vector autoencoders are trained on `train` (all cells, ignoring labels) and
  validated on `val_self_supervised`.
- Embeddings for downstream evaluation are always extracted from `val_downstream`
  (labeled cells only, needed for ARI/NMI/kNN ground truth).
