# MambaC2S: Transformer vs Mamba on Levine32 CyTOF Data

Reproducible research codebase comparing Transformer and Mamba (SSM) architectures on
CyTOF single-cell data using marker-token sequences (Cell2Sentence-style).

## Scientific Question

Does architecture choice (Transformer vs Mamba) matter for learning useful cell
representations from CyTOF marker sequences, independently of tokenization choice?

## Tokenization Schemes

| Scheme | Example |
|--------|---------|
| A. Rank-only | `CD3 CD4 CCR7 CD45RA ...` |
| B. Strength-only | `CD3_HIGH CD4_HIGH CCR7_LOW ...` |
| C. Hybrid | `CD3_R1 CD3_HIGH CD4_R2 CD4_HIGH ...` |

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Place Levine32 data in data/ (as .fcs, .h5ad, or .csv)

# 3. Run full experiment
cd scripts/
python prepare_data.py --config ../configs/base.yaml
python make_splits.py --config ../configs/base.yaml
python train_model.py --config ../configs/transformer.yaml
python evaluate_model.py --checkpoint ../outputs/EXP_ID/best_checkpoint.pt \
       --config ../configs/transformer.yaml
python summarize_results.py --output_dir ../outputs/

# Or run everything at once:
python run_full_experiment.py --config ../configs/transformer.yaml
```

## Project Structure

```
MambaC2S/
  configs/           YAML configuration files
  data/              Raw and processed data (not tracked)
  src/
    data/            Loading, preprocessing, tokenization, vocab, splits
    models/          BaseModel, Transformer, Mamba
    training/        Trainer with early stopping and checkpointing
    evaluation/      Metrics (ARI, NMI, kNN purity), perturbation
    utils/           Config, logging, reproducibility
  scripts/           Entry-point CLI scripts
  notebooks/         Jupyter notebooks for exploration
  outputs/           Experiment artifacts (not tracked)
  docs/              Analysis LLM instructions and schemas
```

## Data

The Levine32 CyTOF dataset (~100k labeled + ~100k unlabeled cells, 32 markers)
can be downloaded from:
- [Cytobank](https://www.cytobank.org/)
- [FlowRepository](http://flowrepository.org/) (accession FR-FCM-ZZPH)

Place the downloaded files in `data/` and configure `dataset.data_dir` in your YAML.

## Outputs

Each experiment writes to `outputs/{EXP_ID}/`:
- `config_resolved.yaml` — full resolved configuration
- `split_manifest.json` — cell IDs per split
- `vocab.json` — token vocabulary
- `training_log.csv` — per-epoch train/val loss
- `best_checkpoint.pt` — best model weights
- `metrics/` — self-supervised and downstream metrics JSON
- `embeddings/` — `.npy` files for train/val/test embeddings
- `plots/` — UMAP and training curve figures
- `tables/` — summary CSV tables

## Reproducibility

All experiments use fixed seeds. Configs, splits, and environment versions are
saved at experiment start. See `src/utils/reproducibility.py`.

## Analysis

See `docs/README_for_analysis_llms.md` for instructions on interpreting results.
See `docs/EXPERIMENT_REGISTRY.md` for a log of all experiments run.
