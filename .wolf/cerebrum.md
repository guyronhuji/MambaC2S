# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Do not edit manually unless correcting an error.
> Last updated: 2026-04-04

## User Preferences

- **Device:** Always use MPS (Apple Silicon GPU), never CUDA. Default device in configs is `"mps"`. `resolve_device()` rejects `"cuda"` strings and falls back to CPU if MPS unavailable. Mixed precision on MPS uses `bfloat16` via `torch.amp.autocast`, no GradScaler.
- **Parallel code style:** When writing parallel processing logic, prefer multiprocessing over multithreading.
- **Notebook UX preferences:** Keep plot legends from obscuring data (prefer legends outside axes), include progress bars for long analysis steps, and prioritize runtime speedups when possible.

## Key Learnings

- **Project:** mambac2s (formerly Transformer vs Mamba; now Transformer/LSTM/GRU vs MLP/DeepSets)
- **Description:** Reproducible research codebase comparing Transformer and Mamba (SSM) architectures on Levine32 CyTOF single-cell data using marker-token sequences.
- **Levine32 loading:** Use `PyCytoData` (`from PyCytoData import DataLoader; exprs = DataLoader.load_dataset(dataset="levine32", force_download=True)`). Channel names are `"CD45RA(La139)Di"` format — strip isotope tag to get `"CD45RA"`. Dataset has 265,627 cells × 32 lineage markers. Cell types in `exprs.cell_types`; `"unassigned"` cells → NaN.
- **Preprocessed df structure:** After preprocessing, the df has `cell_id`, `{marker}` (processed value), `{marker}_rank` (int), `{marker}_bin` (str e.g. LOW/MED/HIGH), and `label` columns. tokenize_cells() requires `_rank` and `_bin` columns.
- **Tokenization API:** `tokenize_cells(df, scheme)` for batch; `tokenize_single_cell(row, marker_cols, scheme)` for single cells (needs full preprocessed row with _rank/_bin). Aliases: `tokenize_dataframe()` and `tokenize_cell()`.
- **Notebooks path convention:** All notebooks use `PROJECT_ROOT = Path('..').resolve()` and build absolute paths from there. `sys.path.insert(0, str(PROJECT_ROOT))` for imports.
- **Experiment artifact reality check:** Current local runs mainly store embeddings as `.npy` under `embeddings/val_downstream_embeddings.npy`; optional files like token embeddings, per-cell predictions, and perturbation summaries may be absent and must be handled gracefully in analysis notebooks.
- **Notebook 05 summary artifacts:** `outputs/nb05_results.csv` and `outputs/nb05_*.png` summarize only the runs that had enough artifacts at notebook execution time. The bar chart mixes sequence-model self-supervised loss with vector-model supervised cross-entropy, so compare sequence models only against sequence models and vector models only against vector models. Also, use `exp_id`/`data_mode` rather than the `scheme` field for raw models in that CSV.
- **Cross-family comparison rule:** Do not compare raw validation losses across sequence next-token models, supervised vector classifiers, and reconstruction autoencoders. Use within-family losses for optimization quality, then compare families using downstream embedding geometry on the same labeled split (e.g. ARI/NMI/kNN purity/silhouette on `val_downstream`).
- **Autoencoder result pattern:** On the current Levine32 outputs, `mlp_autoencoder_raw` learns meaningfully biological latents and lands between supervised `mlp_raw` and sequence models, while `deepsets_autoencoder_raw` remains weak and diffuse. Current downstream geometry on `val_downstream`: `mlp_raw` ARI≈0.792, `mlp_autoencoder_raw` ARI≈0.595, `transformer_hybrid` ARI≈0.372, `mamba_hybrid` ARI≈0.333, `deepsets_autoencoder_raw` ARI≈0.112, `deepsets_raw` ARI≈0.105.
- **Hybrid marker-embedding interpretation:** For marker-level biology on hybrid-tokenized models, averaging only the `LOW/MED/HIGH` strength-token embeddings per marker gives a cleaner lineage/program signal than averaging rank and strength tokens together; rank tokens inject ordering mechanics and make marker links noisier.
- **`preprocess()` API:** Takes keyword args (`arcsinh`, `arcsinh_cofactor`, `normalization`, `bin_count`, `bins`), NOT a config dict. Returns a single DataFrame, not a tuple. Use `preprocess_from_config(df, cfg_dict)` when calling from scripts/notebooks — this unpacks the config and returns `(df, log)`.
- **Do NOT call `preprocess(df, cfg['preprocessing'])`** — this is wrong. Always use `preprocess_from_config(df, cfg['preprocessing'])` at call sites that have a config dict.

- **GRU on MPS is slow with `nn.GRU`**: PyTorch's MPS backend has a fused Metal kernel for `nn.LSTM` but not for `nn.GRU`. Using `nn.GRU` on MPS falls back to element-wise per-timestep ops (~16 ms/batch vs ~11 ms/batch with JIT). Fix: use `_gru_scan` (JIT-compiled) + per-layer `nn.Linear` in `gru_lm.py`. Same root cause as SimpleMambaLM's slowness before `@torch.jit.script` was added.
- **`tqdm.auto` in trainer.py**: `from tqdm.auto import tqdm` auto-detects Jupyter and renders notebook-style widget bars. Without this, tqdm in a notebook shows garbled text output. This is now the import in `src/training/trainer.py`.
- **Notebook 05 uses in-process training**: Notebook 05 calls `Trainer` directly (not subprocess) so tqdm bars render live. Subprocess with `stdout=PIPE` buffers all output until process exit — no live bars.
- **Notebook 05 "load from disk" cell**: Section 8 of nb05 scans `outputs/` and merges results from disk with any session `all_results`. This makes sections 9–12 (table, curves, UMAPs) work even if training cell was skipped. Always run section 8 before plotting.
- **val_downstream for UMAP**: Sequence model embeddings for UMAP are extracted from `split_dfs['val_downstream']` (labeled cells), NOT `val_self_supervised`. This is the only split with ground-truth cell types for coloring.

## Do-Not-Repeat

<!-- Mistakes made and corrected. Each entry prevents the same mistake recurring. -->
<!-- Format: [YYYY-MM-DD] Description of what went wrong and what to do instead. -->

- **[2026-04-05] Do NOT use `nn.GRU` directly on MPS.** It falls back to a slow per-timestep path. Use the JIT-compiled `_GRULayer`/`_gru_scan` in `src/models/gru_lm.py` instead.
- **[2026-04-05] Do NOT use `from tqdm import tqdm` in training code.** Use `from tqdm.auto import tqdm` so Jupyter notebook users get proper widget progress bars.
- **[2026-04-05] Do NOT call subprocess with `stdout=PIPE` to show live training progress in notebooks.** Call `Trainer` directly in-process.
- **[2026-04-05] Do NOT put `elif data_mode == "reconstruction":` AFTER an `else:` block.** Python syntax error. The original `else:` (vector mode) must be converted to `elif data_mode == "vector":` before adding a third branch.
- **[2026-04-05] Do NOT compare supervised MLP against self-supervised sequence models.** Use unsupervised autoencoders (MLPAutoencoder, DeepSetsAutoencoder) as the fair baseline. MLPClassifier/DeepSetsClassifier are supervised upper bounds only.

## Decision Log

- **[2026-04-05] Removed Mamba entirely; replaced with LSTM, GRU, MLP, DeepSets.** Mamba CUDA kernels incompatible with MPS; pure-PyTorch SimpleMambaLM was slowest architecture on MPS. LSTM/GRU are faster on MPS and easier to deploy.
- **[2026-04-05] Three data modes.** `sequence` (Transformer/LSTM/GRU, next-token, val_self_supervised), `vector` (MLP/DeepSets classifiers, cross-entropy, labeled_train→val_downstream), `reconstruction` (MLPAutoencoder/DeepSetsAutoencoder, MSE, train+unlabeled→val_self_supervised).
- **[2026-04-05] Fair comparison via unsupervised autoencoders.** MLPClassifier/DeepSetsClassifier are supervised upper bounds. MLPAutoencoder/DeepSetsAutoencoder are unsupervised baselines — same data split as sequence models (all training cells, no labels). training_summary.json includes supervision_type/input_structure/training_objective metadata.
- **[2026-04-05] n_markers and n_classes derived at runtime.** Vector model configs do NOT hardcode these; they are extracted from data in train_model.py and passed to build_vector_model(). Autoencoders receive n_classes=0 (unused).
- **[2026-04-05] 13 config files.** transformer/lstm/gru × rank_only/strength_only/hybrid + mlp_raw + deepsets_raw + mlp_autoencoder_raw + deepsets_autoencoder_raw.
- **[2026-04-05] VectorBaseModel.forward() returns tuple for autoencoders.** MLPAutoencoder and DeepSetsAutoencoder override forward() to return (recon, z) tuple. Python allows this despite abstract signature saying Tensor — trainer reconstruction mode unpacks the tuple. Classifiers still return Tensor as expected.
