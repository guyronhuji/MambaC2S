# anatomy.md

> Auto-maintained by OpenWolf. Last scanned: 2026-04-05T07:39:42.300Z
> Files: 192 tracked | Anatomy hits: 0 | Misses: 0

## ../../../../../.claude/projects/-Users-ronguy-Dropbox-Work-CyTOF-Experiments-MambaC2S/memory/

- `feedback_mps_performance.md` — MPS Performance — Confirmed Patterns (~393 tok)
- `feedback_notebook_patterns.md` — Notebook Patterns (MambaC2S) (~399 tok)
- `project_architecture.md` — MambaC2S Architecture (as of 2026-04-05) (~789 tok)

## ./

- `.DS_Store` (~2730 tok)
- `.gitignore` — Git ignore rules (~164 tok)
- `.runpodkey` (~16 tok)
- `AGENTS.md` — OpenWolf (~57 tok)
- `CLAUDE.md` — OpenWolf (~57 tok)
- `Inst.md` — PROJECT: Transformer vs Mamba (SSM) on Levine32 CyTOF (~1471 tok)
- `pyproject.toml` — Python project configuration (~255 tok)
- `README.md` — Project documentation (~1667 tok)
- `requirements.txt` — Python dependencies (~137 tok)

## .claude/

- `settings.json` (~441 tok)
- `settings.local.json` (~459 tok)

## .claude/rules/

- `openwolf.md` (~313 tok)

## azure/

- `create_vm.sh` — Create an Azure GPU VM for MambaC2S training (Option A) (~624 tok)
- `fetch_results.sh` — Fetch experiment outputs from the Azure VM to your local machine (~184 tok)
- `setup_vm.sh` — One-time VM setup: install CUDA drivers, Python deps, clone repo (~648 tok)
- `train.sh` — Run the full experiment matrix on Azure VM (~473 tok)

## configs/

- `base.yaml` (~361 tok)
- `dataset.yaml` — Dataset-specific overrides (~156 tok)
- `deepsets_autoencoder_raw.yaml` (~108 tok)
- `deepsets_raw.yaml` (~84 tok)
- `gru_hybrid.yaml` (~74 tok)
- `gru_rank_only.yaml` (~75 tok)
- `gru_strength_only.yaml` (~76 tok)
- `lstm_hybrid.yaml` (~74 tok)
- `lstm_rank_only.yaml` (~75 tok)
- `lstm_strength_only.yaml` (~76 tok)
- `mlp_autoencoder_raw.yaml` (~95 tok)
- `mlp_raw.yaml` (~86 tok)
- `transformer_hybrid.yaml` (~86 tok)
- `transformer_rank_only.yaml` (~87 tok)
- `transformer_strength_only.yaml` (~88 tok)
- `transformer.yaml` — Transformer model configuration (~98 tok)

## data/

- `.gitkeep` (~0 tok)

## docs/

- `ANALYSIS_LLM_INSTRUCTIONS.md` — Analysis LLM Instructions (~990 tok)
- `EXPERIMENT_REGISTRY.md` — Experiment Registry (~335 tok)
- `README_for_analysis_llms.md` — MambaC2S — Guide for Analysis LLMs (~1523 tok)
- `RESULT_SCHEMA.md` — Result Schema (~1320 tok)

## gcp/

- `create_vm.sh` — Create a GCP GPU VM for MambaC2S training (~1041 tok)
- `fetch_results.sh` — Fetch experiment outputs from the GCP VM to your local machine (~187 tok)
- `setup_vm.sh` — One-time VM setup: CUDA drivers, Python deps, clone repo (~683 tok)
- `train.sh` — Run the full experiment matrix on GCP VM (~444 tok)

## notebooks/

- `.DS_Store` (~1639 tok)
- `01_dataset_inspection.ipynb` (~83519 tok)
- `02_tokenization_examples.ipynb` (~23125 tok)
- `03_embedding_visualization.ipynb` — Declares label (~77437 tok)
- `04_train_model.ipynb` (~9932 tok)
- `05_run_all_experiments.ipynb` — Runs all 13 experiments (sequence + vector classifier + autoencoder). Sections: data prep, tokenise, run, load-from-disk, curves, UMAPs. (~4200 tok)\n- `06_latent_dim_sweep.ipynb` — MLPAutoencoder latent dim sweep d∈{2,4,8,16,32}. Trains, ARI/NMI/kNN/silhouette, per-class MSE heatmap, rare-class analysis, summary CSV. 29 cells (~3500 tok)

## notebooks/.ipynb_checkpoints/

- `01_dataset_inspection-checkpoint.ipynb` — Declares distribution (~1366 tok)
- `02_tokenization_examples-checkpoint.ipynb` (~1901 tok)
- `03_embedding_visualization-checkpoint.ipynb` — Declares label (~3007 tok)
- `04_train_model-checkpoint.ipynb` (~27927 tok)
- `analysis_transformer_vs_mamba_biology-checkpoint.ipynb` (~695 tok)

## outputs/

- `.DS_Store` (~2726 tok)
- `.gitkeep` (~0 tok)
- `nb05_bar_chart.png` — Notebook 05 summary bar chart comparing selected models/runs (~15 tok)
- `nb05_results.csv` — Notebook 05 compact results table for selected sequence and vector models (~120 tok)
- `nb05_sequence_curves.png` — Notebook 05 training/validation loss curves for sequence models (~15 tok)
- `nb05_umaps.png` — Notebook 05 UMAP montage for selected trained models (~20 tok)
- `nb05_vector_curves.png` — Notebook 05 training/validation loss curves for vector models (~15 tok)

## outputs/deepsets_autoencoder_raw_20260405_090252/

- `best_checkpoint.pt` — Best DeepSets autoencoder raw-vector checkpoint (~0 tok)
- `config_resolved.yaml` (~226 tok)
- `training_log.csv` — Per-epoch reconstruction losses for DeepSets autoencoder run (~90 tok)
- `training_summary.json` — Compact experiment summary for DeepSets autoencoder (~140 tok)

## outputs/deepsets_raw_20260405_070942/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~0 tok)

## outputs/deepsets_raw_20260405_081336/

- `best_checkpoint.pt` — Best DeepSets raw-vector checkpoint (~0 tok)
- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `training_log.csv` — Per-epoch training/validation losses for DeepSets raw-vector run (~90 tok)
- `vocab.json` (~0 tok)

## outputs/gru_hybrid_20260405_070934/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~6822 tok)

## outputs/gru_hybrid_20260405_075745/

- `best_checkpoint.pt` — Best GRU hybrid checkpoint (~0 tok)
- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `training_log.csv` — Per-epoch training/validation losses for GRU hybrid run (~60 tok)
- `vocab.json` (~6822 tok)

## outputs/gru_rank_only_20260405_070926/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~0 tok)

## outputs/gru_strength_only_20260405_070930/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~0 tok)

## outputs/lstm_hybrid_20260405_070923/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~6822 tok)

## outputs/lstm_hybrid_20260405_074017/

- `best_checkpoint.pt` — Best LSTM hybrid checkpoint (~0 tok)
- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `training_log.csv` — Per-epoch training/validation losses for LSTM hybrid run (~90 tok)
- `vocab.json` (~6822 tok)

## outputs/lstm_rank_only_20260405_070915/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~0 tok)

## outputs/lstm_strength_only_20260405_070919/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~0 tok)

## outputs/mamba_hybrid_20260404_214014/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~6822 tok)

## outputs/mamba_hybrid_20260404_215232/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~6822 tok)

## outputs/mamba_hybrid_20260404_220345/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `training_log.csv` (~404 tok)
- `vocab.json` (~6822 tok)

## outputs/mlp_autoencoder_raw_20260405_085828/

- `best_checkpoint.pt` — Earlier MLP autoencoder raw-vector checkpoint (~0 tok)
- `config_resolved.yaml` (~226 tok)
- `training_log.csv` — Partial per-epoch reconstruction losses for early MLP autoencoder run (~70 tok)

## outputs/mlp_autoencoder_raw_20260405_090035/

- `best_checkpoint.pt` — Best MLP autoencoder raw-vector checkpoint (~0 tok)
- `config_resolved.yaml` (~226 tok)
- `training_log.csv` — Per-epoch reconstruction losses for MLP autoencoder run (~90 tok)
- `training_summary.json` — Compact experiment summary for MLP autoencoder (~140 tok)

## outputs/mlp_raw_20260405_070938/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~0 tok)

## outputs/mlp_raw_20260405_081307/

- `best_checkpoint.pt` — Best MLP raw-vector checkpoint (~0 tok)
- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `training_log.csv` — Per-epoch training/validation losses for MLP raw-vector run (~90 tok)
- `vocab.json` (~0 tok)

## outputs/transformer_hybrid_20260405_070912/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~6822 tok)

## outputs/transformer_hybrid_20260405_071533/

- `best_checkpoint.pt` — Best Transformer hybrid checkpoint (~0 tok)
- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `training_log.csv` — Per-epoch training/validation losses for Transformer hybrid run (~90 tok)
- `vocab.json` (~6822 tok)

## outputs/transformer_rank_only_20260405_070904/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~0 tok)

## outputs/transformer_rank_only_20260405_071059/

- `best_checkpoint.pt` — Best Transformer rank-only checkpoint (~0 tok)
- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `training_log.csv` — Per-epoch training/validation losses for Transformer rank-only run (~60 tok)
- `vocab.json` (~0 tok)

## outputs/transformer_rank_only_20260405_071401/

- `best_checkpoint.pt` — Early/partial Transformer rank-only retry checkpoint (~0 tok)
- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `training_log.csv` — Short training log for Transformer rank-only retry (~40 tok)
- `vocab.json` (~0 tok)

## outputs/transformer_strength_only_20260405_070908/

- `config_resolved.yaml` (~226 tok)
- `environment.json` (~109 tok)
- `vocab.json` (~0 tok)

## runpod/

- `create_pod.sh` — Create a RunPod GPU pod for MambaC2S training (~1929 tok)
- `fetch_results.sh` — Fetch experiment outputs from a running RunPod pod. (~1325 tok)
- `monitor.py` — Live training monitor — reads per-job status/CSV files and renders (~1124 tok)
- `setup_pod.sh` — One-time pod setup: clone repo and install dependencies (~660 tok)
- `train_hybrid.sh` — ============================================================ (~575 tok)
- `train_latent_sweep.sh` — ============================================================ (~395 tok)
- `train.sh` — ============================================================ (~757 tok)

## scripts/

- `evaluate_model.py` — Evaluate a trained model: self-supervised loss, embedding quality, perturbation. (~2717 tok)
- `latent_dim_sweep.py` — Latent dimension sweep for MLPAutoencoder on Levine32 CyTOF data. (~2576 tok)
- `make_splits.py` — Create reproducible train/val/test splits from processed data. (~653 tok)
- `prepare_data.py` — Prepare raw CyTOF data: load, preprocess, and save as .h5ad. (~1103 tok)
- `run_full_experiment.py` — Run a complete experiment end-to-end: prepare → splits → train → evaluate. (~1250 tok)
- `summarize_results.py` — Summarize all experiment results into a comparison table. (~1262 tok)
- `train_model.py` — Train a model on CyTOF data. (~4146 tok)

## src/

- `__init__.py` — MambaC2S: Transformer vs Mamba on Levine32 CyTOF data. (~24 tok)
- `.DS_Store` (~1639 tok)

## src/data/

- `__init__.py` — Data loading, preprocessing, tokenization, vocabulary, and splits. (~115 tok)
- `loader.py` — Levine32 CyTOF dataset loader. (~4152 tok)
- `preprocessing.py` — CyTOF data preprocessing. (~2180 tok)
- `splits.py` — Dataset splitting for CyTOF experiments. (~2239 tok)
- `tokenization.py` — Cell tokenization schemes. (~2196 tok)
- `vocab.py` — Vocabulary for cell token sequences. (~2183 tok)

## src/evaluation/

- `__init__.py` — Evaluation: self-supervised metrics, embedding quality, perturbation. (~127 tok)
- `metrics.py` — Evaluation metrics for CyTOF sequence models. (~2253 tok)
- `perturbation.py` — Perturbation analysis for CyTOF sequence models. (~2041 tok)

## src/models/

- `__init__.py` — Model implementations: Transformer, LSTM, GRU, MLP, DeepSets. (~1271 tok)
- `base.py` — Abstract base classes for all CyTOF models. (~2202 tok)
- `deepsets_autoencoder.py` — DeepSets autoencoder for unsupervised, permutation-invariant representation learning. (~1016 tok)
- `deepsets_model.py` — DeepSets classifier for raw CyTOF marker vectors. (~787 tok)
- `gru_lm.py` — GRU autoregressive language model for CyTOF token sequences. (~1312 tok)
- `lstm_lm.py` — LSTM autoregressive language model for CyTOF token sequences. (~725 tok)
- `mlp_autoencoder.py` — MLP autoencoder for unsupervised representation learning on CyTOF marker vectors. (~858 tok)
- `mlp_model.py` — MLP supervised classifier for raw CyTOF marker vectors. (~586 tok)
- `transformer.py` — Small causal Transformer language model for CyTOF token sequences. (~2032 tok)

## src/training/

- `__init__.py` — Training loop for autoregressive sequence models. (~48 tok)
- `trainer.py` — Training loop: modes sequence/vector/reconstruction. CellSequenceDataset, CellVectorDataset, CellUnlabeledDataset. (~5500 tok)

## src/utils/

- `__init__.py` — Utility modules: config loading, structured logging, reproducibility. (~118 tok)
- `config.py` — YAML configuration loading and merging. (~992 tok)
- `logging.py` — Structured logging utilities. (~957 tok)
- `reproducibility.py` — Reproducibility helpers: seed setting and environment logging. (~910 tok)

## tools/

- `generate_analysis_notebook.py` — Generate the Transformer vs Mamba biological analysis notebook. (~30892 tok)
