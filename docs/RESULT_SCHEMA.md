# Result Schema

Defines the exact format of every output file produced by MambaC2S.

---

## Experiment Directory

Path: `outputs/{EXP_ID}/`

`EXP_ID` format: `{model_type}_{tokenization_scheme}_{YYYYMMDD_HHMMSS}`

Example: `transformer_rank_only_20240101_120000`

---

## config_resolved.yaml

Full merged YAML config used for this experiment.  Same schema as `configs/base.yaml`.

Key fields:
```yaml
seed: int
device: str
dataset:
  data_dir: str
  dataset_name: str
  label_col: str
preprocessing:
  arcsinh: bool
  arcsinh_cofactor: float
  normalization: str   # "zscore" | "robust"
  bin_count: int
  bins: list[str]
tokenization:
  scheme: str          # "rank_only" | "strength_only" | "hybrid"
model:
  type: str            # "transformer" | "mamba"
  d_model: int
  n_layers: int
  nhead: int           # transformer only
  dropout: float
  max_seq_len: int
  d_state: int         # mamba only
  d_conv: int          # mamba only
  expand: int          # mamba only
training:
  batch_size: int
  lr: float
  weight_decay: float
  max_epochs: int
  patience: int
  grad_clip: float
  mixed_precision: bool
evaluation:
  knn_k: int
  n_umap_neighbors: int
  umap_min_dist: float
  embedding_pooling: str  # "mean" | "last"
  batch_size: int
output:
  output_dir: str
  exp_id: str | null
```

---

## split_manifest.json

```json
{
  "version": 1,
  "n_cells_total": 200000,
  "split_sizes": {
    "labeled_train": 60000,
    "labeled_val": 20000,
    "labeled_test": 20000,
    "unlabeled_train": 90000,
    "unlabeled_val": 10000,
    "train": 150000,
    "val_self_supervised": 10000,
    "val_downstream": 20000,
    "test": 20000
  },
  "splits": {
    "labeled_train": ["cell_0001", "cell_0002", ...],
    "labeled_val":   [...],
    "labeled_test":  [...],
    "unlabeled_train": [...],
    "unlabeled_val": [...],
    "train": [...],
    "val_self_supervised": [...],
    "val_downstream": [...],
    "test": [...]
  },
  "label_map": {
    "cell_0001": "CD4_T_cell",
    "cell_0002": null,
    ...
  }
}
```

---

## vocab.json

```json
{
  "version": 1,
  "size": 150,
  "special_tokens": {
    "PAD": 0,
    "BOS": 1,
    "EOS": 2,
    "UNK": 3
  },
  "token2id": {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
    "CD3": 4,
    "CD4": 5,
    ...
  }
}
```

---

## training_log.csv

One row per epoch.

| Column | Type | Description |
|--------|------|-------------|
| `epoch` | int | Epoch number (1-indexed) |
| `train_loss` | float | Mean training loss (cross-entropy per token) |
| `val_loss` | float | Mean validation loss |
| `val_perplexity` | float | exp(val_loss) |
| `lr` | float | Learning rate at end of epoch |

---

## training_summary.json

```json
{
  "best_val_loss": 2.134,
  "best_epoch": 23,
  "total_epochs": 33,
  "checkpoint": "outputs/EXP_ID/best_checkpoint.pt"
}
```

---

## best_checkpoint.pt

PyTorch state dict.  Load with:
```python
state = torch.load("best_checkpoint.pt", map_location="cpu")
# Keys: epoch, model_state_dict, optimiser_state_dict, val_loss,
#       model_class, vocab_size, d_model, max_seq_len
```

---

## metrics/self_supervised.json

```json
{
  "loss": 2.134,
  "perplexity": 8.45
}
```

---

## metrics/downstream.json

```json
{
  "ARI": 0.612,
  "NMI": 0.703,
  "knn_purity": 0.841
}
```

---

## metrics/perturbation.json

```json
{
  "n_cells_evaluated": 200,
  "n_perturbations": 987,
  "mean_cosine_distance": 0.0342,
  "std_cosine_distance": 0.0211,
  "max_cosine_distance": 0.3104,
  "per_cell": [
    {
      "cell_idx": 42,
      "n_perturbations": 5,
      "mean_cosine_distance": 0.0310
    },
    ...
  ]
}
```

---

## embeddings/

| File | Shape | dtype | Description |
|------|-------|-------|-------------|
| `val_downstream_embeddings.npy` | (n_val, d_model) | float32 | Mean-pooled hidden states for labeled_val cells |
| `val_downstream_umap.npy` | (n_val, 2) | float32 | 2-D UMAP projection |
| `test_embeddings.npy` | (n_test, d_model) | float32 | Only present after final test evaluation |

Load with `numpy.load(path)`.

Cell ordering matches the order in `split_manifest.json → splits.val_downstream`.

---

## environment.json

```json
{
  "python": "3.11.0 ...",
  "platform": "macOS-...",
  "pytorch": "2.1.0",
  "numpy": "1.26.0",
  "cuda_available": false,
  "anndata": "0.10.0",
  "fcsparser": "0.2.8",
  "umap": "0.5.6",
  "sklearn": "1.4.0"
}
```

---

## outputs/summary.csv

Columns:

| Column | Type | Description |
|--------|------|-------------|
| `exp_id` | str | Experiment directory name |
| `model_type` | str | `transformer` or `mamba` |
| `tokenization` | str | `rank_only`, `strength_only`, or `hybrid` |
| `d_model` | int | Model dimensionality |
| `n_layers` | int | Number of layers |
| `best_epoch` | int | Epoch of best checkpoint |
| `total_epochs` | int | Total epochs trained |
| `best_val_loss` | float | Best validation loss (from training) |
| `val_loss` | float | Validation loss (from evaluation script) |
| `val_perplexity` | float | Validation perplexity |
| `ARI` | float | Adjusted Rand Index |
| `NMI` | float | Normalized Mutual Information |
| `knn_purity` | float | kNN purity (k=15) |
| `pert_mean_cosine_dist` | float | Mean perturbation cosine distance |
| `checkpoint` | str | Absolute path to best_checkpoint.pt |
