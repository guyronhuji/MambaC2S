# PROJECT: Transformer vs Mamba (SSM) on Levine32 CyTOF

## ROLE

You are a senior ML research engineer and scientific software architect.

Your task is to build a **fully reproducible research codebase** that compares:

1. a **Transformer language model** baseline in the style of Cell2Sentence (C2S),
2. a **Mamba / state space sequence model** baseline,
3. optionally a simple non-neural baseline scaffold for later extension,

for **CyTOF single-cell data** using the **Levine32** dataset.

The goal is to test whether **marker-token sequence models** can learn useful cell representations, and whether **architecture choice (Transformer vs Mamba)** matters, separately from **tokenization choice**.

This is a research codebase, not a toy demo.

---

# HIGH-LEVEL GOAL

Build a project that:

* loads the Levine32 dataset,
* preprocesses cells and markers,
* converts each cell into token sequences using several tokenization schemes,
* trains:

  * a **small autoregressive Transformer**
  * a **small Mamba-style autoregressive model**
* evaluates them fairly on:

  * held-out sequence prediction,
  * embedding quality,
  * downstream clustering / label agreement,
  * perturbation consistency,
* saves all artifacts in a clean, inspectable structure,
* and writes **README and instruction files for future analysis LLMs**.

---

# CORE SCIENTIFIC DESIGN

## DATASET

Use the Levine32 CyTOF dataset:

* ~100k labeled cells
* ~100k unlabeled cells
* ~32 markers

Do not hardcode paths. Make dataset loading configurable.

---

# MAIN SCIENTIFIC QUESTION

Compare:

* Transformer (C2S-style)
* Mamba (SSM)

using identical **marker-based tokenizations**.

---

# TOKENIZATION CONDITIONS

Implement:

## A. Rank-only

```text
CD3 CD4 CCR7 CD45RA ...
```

## B. Strength-only

```text
CD3_HIGH CD4_HIGH CCR7_LOW ...
```

## C. Hybrid rank + strength

```text
CD3_R1 CD3_HIGH CD4_R2 CD4_HIGH ...
```

All schemes must use identical splits.

---

# DATA SPLITS

## Labeled cells

* 60% train
* 20% val
* 20% test (stratified)

## Unlabeled cells

* 90% train
* 10% val

## Training corpus

* labeled_train + unlabeled_train (ignore labels)

## Validation

* unlabeled_val (self-supervised)
* labeled_val (downstream)

## Test

* labeled_test (final only)

Save split manifests.

---

# PREPROCESSING

Support:

* arcsinh transform (optional)
* z-score or robust normalization
* rank within cell
* configurable binning

Log all preprocessing decisions.

---

# MODELS

## Transformer

* small autoregressive model
* causal masking
* 2–4 layers
* d_model 64–128

## Mamba (SSM)

* similar size
* same interface
* dependency optional but preferred

## Interface

* forward()
* encode()
* optional generate()

---

# TRAINING

* autoregressive next-token prediction
* cross-entropy loss
* BOS/EOS tokens
* teacher forcing

---

# EMBEDDINGS

Extract:

* mean pooled hidden state
* final token state

---

# EVALUATION

## 1. Self-supervised

* loss
* perplexity

## 2. Embedding quality

* ARI
* NMI
* kNN purity

## 3. UMAP

* marker space
* transformer
* mamba

## 4. Perturbation

* controlled token changes
* measure embedding/output shifts

## 5. Tokenization ablation

Compare all schemes across models.

---

# CONFIG SYSTEM

Use YAML configs for:

* dataset
* preprocessing
* tokenization
* model
* training
* evaluation

---

# CODE STRUCTURE

```text
project_root/
  README.md
  pyproject.toml
  requirements.txt
  configs/
  data/
  src/
    data/
    models/
    training/
    evaluation/
    utils/
  scripts/
  notebooks/
  outputs/
  docs/
```

---

# ANALYSIS LLM FILES (MANDATORY)

## docs/README_for_analysis_llms.md

Explain:

* project purpose
* file locations
* metrics meaning
* how to compare experiments

## docs/ANALYSIS_LLM_INSTRUCTIONS.md

Explicit instructions:

* what files to read
* how to evaluate results
* how to avoid leakage
* how to compare models safely

## docs/RESULT_SCHEMA.md

Define:

* all output files
* metrics format
* embedding format
* naming conventions

## docs/EXPERIMENT_REGISTRY.md

Track:

* experiment_id
* config
* metrics
* checkpoints

## Per-experiment README

Explain each run.

---

# OUTPUT STRUCTURE

```text
outputs/EXP_ID/
  config_resolved.yaml
  split_manifest.json
  vocab.json
  training_log.csv
  best_checkpoint.pt
  metrics/
  embeddings/
  plots/
  tables/
  README_experiment.md
```

---

# REPRODUCIBILITY

* fixed seeds
* saved configs
* saved splits
* environment logging
* version tracking

---

# TRAINING DEFAULTS

## Transformer

* 2–3 layers
* d_model 64–128
* dropout 0.1

## Mamba

* similar scale

## Optimization

* AdamW
* early stopping

---

# METRICS

Save:

## Self-supervised

* val/test loss
* perplexity

## Downstream

* ARI
* NMI
* purity

## Perturbation

* embedding shifts
* consistency

---

# PERTURBATION DESIGN

* small local token edits
* valid transitions only
* save before/after
* aggregate results

---

# NOTEBOOKS

Provide:

* dataset inspection
* tokenization examples
* embedding visualization

---

# CLI SCRIPTS

* prepare_data.py
* make_splits.py
* train_model.py
* evaluate_model.py
* run_full_experiment.py
* summarize_results.py

---

# FAILURE HANDLING

Handle:

* missing dependencies
* bad data
* invalid configs
* hardware limits

---

# DO NOT

* leak test data
* hide preprocessing
* hardcode assumptions
* rely only on notebooks

---

# FINAL REQUIREMENT

Provide example run:

```bash
python prepare_data.py
python make_splits.py
python train_model.py --config configs/transformer.yaml
python evaluate_model.py --checkpoint path
python summarize_results.py
```

---

# PRIORITY ORDER

1. data loading
2. splits
3. tokenization
4. vocab
5. transformer
6. training
7. evaluation
8. mamba
9. perturbation
10. reports
11. analysis docs
12. notebooks

---

# EXPECTED QUALITY

* clean
* modular
* reproducible
* well documented
* extensible

---

# END OF SPEC

