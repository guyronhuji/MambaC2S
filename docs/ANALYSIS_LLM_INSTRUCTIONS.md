# Analysis LLM Instructions

> Step-by-step instructions for an AI assistant tasked with analysing
> MambaC2S experiment results.  Follow these exactly to avoid evaluation
> errors, data leakage, and incorrect conclusions.

---

## Step 1 — Orient Yourself

1. Read `docs/README_for_analysis_llms.md` for project overview.
2. Read `docs/RESULT_SCHEMA.md` for exact file formats.
3. List `outputs/` to see which experiments exist.

---

## Step 2 — Load the Summary Table

```python
import pandas as pd
df = pd.read_csv("outputs/summary.csv")
print(df[["exp_id", "model_type", "tokenization", "ARI", "NMI", "knn_purity", "val_perplexity"]])
```

This gives you a quick overview.  Do NOT draw conclusions yet — check
configs first.

---

## Step 3 — Verify Experimental Comparability

For any two experiments you want to compare:

1. Load both `config_resolved.yaml` files.
2. Confirm they share:
   - The same `splits.*` fractions
   - The same `seed`
   - The same `preprocessing.*` settings
3. Check that `split_manifest.json` inside each experiment directory has the
   same `n_cells_total` and `split_sizes`.

If any of the above differ, **do not make direct comparisons** without
noting the confounds.

---

## Step 4 — Read Metrics (Correct Order)

For model selection and reporting, **use only validation metrics**:
- `metrics/self_supervised.json` → `loss`, `perplexity`
- `metrics/downstream.json` → `ARI`, `NMI`, `knn_purity`

**Never use `test` split metrics for model selection.**  If `test_embeddings.npy`
exists, it means final evaluation was already run.  Do not re-evaluate on test.

---

## Step 5 — Interpret Embedding Quality

Primary ranking criterion: **ARI** (most robust to cluster imbalance).
Secondary: **NMI**.  Tertiary: **knn_purity**.

Suggested thresholds for CyTOF (Levine32, ~30 cell types):
| ARI | Interpretation |
|-----|----------------|
| < 0.1 | Poor — near random |
| 0.1 – 0.3 | Weak structure captured |
| 0.3 – 0.6 | Moderate — clearly useful |
| > 0.6 | Strong — cell types well separated |

---

## Step 6 — Compare Architectures

Control for tokenisation by comparing:
- `transformer_rank_only_*` vs `mamba_rank_only_*`
- `transformer_strength_only_*` vs `mamba_strength_only_*`

Control for architecture by comparing:
- `transformer_rank_only_*` vs `transformer_strength_only_*` vs `transformer_hybrid_*`

---

## Step 7 — Interpret Perturbation Results

`metrics/perturbation.json`:
- High `mean_cosine_distance` → model is sensitive to token edits (fine-grained representations).
- Low value → embeddings are robust / insensitive.

Neither is inherently better; compare relative differences between models.

---

## How to Avoid Data Leakage

- **NEVER** load `test_embeddings.npy` for model comparison.
- **NEVER** load cells from `labeled_test` IDs (found in `split_manifest.json`)
  unless explicitly running final test evaluation.
- When computing metrics, always verify the DataFrame was filtered by
  `val_downstream` or `val_self_supervised` cell IDs.

---

## How to Load Embeddings for Custom Analysis

```python
import numpy as np
import json

# Embeddings
embs = np.load("outputs/EXP_ID/embeddings/val_downstream_embeddings.npy")  # (n, d)
umap2d = np.load("outputs/EXP_ID/embeddings/val_downstream_umap.npy")       # (n, 2)

# Labels (from manifest)
with open("outputs/EXP_ID/split_manifest.json") as f:
    manifest = json.load(f)
val_ids = manifest["splits"]["val_downstream"]
label_map = manifest["label_map"]
labels = [label_map[cid] for cid in val_ids]
```

---

## Reporting Template

When summarising results, include:

1. Number of experiments compared
2. Dataset and split configuration (seed, fractions)
3. Table: model × tokenisation → ARI / NMI / perplexity
4. Best model by ARI and by perplexity (may differ)
5. Whether Mamba or Transformer performs better, and under which tokenisation
6. Perturbation sensitivity comparison
7. Caveats (test set not used, hardware differences if any)
