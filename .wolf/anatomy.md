# anatomy.md

> Auto-maintained by OpenWolf. Last scanned: 2026-04-04T09:04:04.678Z
> Files: 57 tracked | Anatomy hits: 0 | Misses: 0

## ./

- `.DS_Store` (~2186 tok)
- `.gitignore` — Git ignore rules (~154 tok)
- `CLAUDE.md` — OpenWolf (~57 tok)
- `Inst.md` — PROJECT: Transformer vs Mamba (SSM) on Levine32 CyTOF (~1471 tok)
- `pyproject.toml` — Python project configuration (~255 tok)
- `README.md` — Project documentation (~759 tok)
- `requirements.txt` — Python dependencies (~82 tok)

## .claude/

- `settings.json` (~441 tok)

## .claude/rules/

- `openwolf.md` (~313 tok)

## configs/

- `base.yaml` (~360 tok)
- `dataset.yaml` — Dataset-specific overrides (~156 tok)
- `mamba.yaml` — Mamba / SSM model configuration (~134 tok)
- `transformer.yaml` — Transformer model configuration (~98 tok)

## data/

- `.gitkeep` (~0 tok)

## docs/

- `ANALYSIS_LLM_INSTRUCTIONS.md` — Analysis LLM Instructions (~990 tok)
- `EXPERIMENT_REGISTRY.md` — Experiment Registry (~335 tok)
- `README_for_analysis_llms.md` — MambaC2S — Guide for Analysis LLMs (~1207 tok)
- `RESULT_SCHEMA.md` — Result Schema (~1320 tok)

## notebooks/

- `01_dataset_inspection.ipynb` (~83519 tok)
- `02_tokenization_examples.ipynb` (~23125 tok)
- `03_embedding_visualization.ipynb` — Declares label (~26668 tok)
- `04_train_model.ipynb` (~7027 tok)

## notebooks/.ipynb_checkpoints/

- `01_dataset_inspection-checkpoint.ipynb` — Declares distribution (~1366 tok)
- `02_tokenization_examples-checkpoint.ipynb` (~1901 tok)
- `03_embedding_visualization-checkpoint.ipynb` — Declares label (~3007 tok)
- `04_train_model-checkpoint.ipynb` (~4153 tok)

## outputs/

- `.gitkeep` (~0 tok)

## outputs/transformer_rank_only_20260404_115547/

- `config_resolved.yaml` (~228 tok)
- `environment.json` (~94 tok)
- `training_log.csv` (~188 tok)
- `vocab.json` (~212 tok)

## scripts/

- `evaluate_model.py` — Evaluate a trained model: self-supervised loss, embedding quality, perturbation. (~2717 tok)
- `make_splits.py` — Create reproducible train/val/test splits from processed data. (~653 tok)
- `prepare_data.py` — Prepare raw CyTOF data: load, preprocess, and save as .h5ad. (~998 tok)
- `run_full_experiment.py` — Run a complete experiment end-to-end: prepare → splits → train → evaluate. (~1250 tok)
- `summarize_results.py` — Summarize all experiment results into a comparison table. (~1262 tok)
- `train_model.py` — Train a Transformer or Mamba model on CyTOF token sequences. (~2181 tok)

## src/

- `__init__.py` — MambaC2S: Transformer vs Mamba on Levine32 CyTOF data. (~24 tok)

## src/data/

- `__init__.py` — Data loading, preprocessing, tokenization, vocabulary, and splits. (~115 tok)
- `loader.py` — Levine32 CyTOF dataset loader. (~3931 tok)
- `preprocessing.py` — CyTOF data preprocessing. (~2180 tok)
- `splits.py` — Dataset splitting for CyTOF experiments. (~2239 tok)
- `tokenization.py` — Cell tokenization schemes. (~2196 tok)
- `vocab.py` — Vocabulary for cell token sequences. (~2183 tok)

## src/evaluation/

- `__init__.py` — Evaluation: self-supervised metrics, embedding quality, perturbation. (~127 tok)
- `metrics.py` — Evaluation metrics for CyTOF sequence models. (~2009 tok)
- `perturbation.py` — Perturbation analysis for CyTOF sequence models. (~2020 tok)

## src/models/

- `__init__.py` — Model implementations: Transformer and Mamba. (~540 tok)
- `base.py` — Abstract base class for all sequence models. (~1671 tok)
- `mamba_model.py` — Mamba / SSM autoregressive language model for CyTOF token sequences. (~3343 tok)
- `transformer.py` — Small causal Transformer language model for CyTOF token sequences. (~2010 tok)

## src/training/

- `__init__.py` — Training loop for autoregressive sequence models. (~48 tok)
- `trainer.py` — Training loop for autoregressive CyTOF sequence models. (~3346 tok)

## src/utils/

- `__init__.py` — Utility modules: config loading, structured logging, reproducibility. (~118 tok)
- `config.py` — YAML configuration loading and merging. (~992 tok)
- `logging.py` — Structured logging utilities. (~957 tok)
- `reproducibility.py` — Reproducibility helpers: seed setting and environment logging. (~910 tok)
