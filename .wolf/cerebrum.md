# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Do not edit manually unless correcting an error.
> Last updated: 2026-04-04

## User Preferences

- **Device:** Always use MPS (Apple Silicon GPU), never CUDA. Default device in configs is `"mps"`. `resolve_device()` rejects `"cuda"` strings and falls back to CPU if MPS unavailable. Mixed precision on MPS uses `bfloat16` via `torch.amp.autocast`, no GradScaler.

## Key Learnings

- **Project:** mambac2s
- **Description:** Reproducible research codebase comparing Transformer and Mamba (SSM) architectures on Levine32 CyTOF single-cell data using marker-token sequences.
- **Levine32 loading:** Use `PyCytoData` (`from PyCytoData import DataLoader; exprs = DataLoader.load_dataset(dataset="levine32", force_download=True)`). Channel names are `"CD45RA(La139)Di"` format — strip isotope tag to get `"CD45RA"`. Dataset has 265,627 cells × 32 lineage markers. Cell types in `exprs.cell_types`; `"unassigned"` cells → NaN.
- **Preprocessed df structure:** After preprocessing, the df has `cell_id`, `{marker}` (processed value), `{marker}_rank` (int), `{marker}_bin` (str e.g. LOW/MED/HIGH), and `label` columns. tokenize_cells() requires `_rank` and `_bin` columns.
- **Tokenization API:** `tokenize_cells(df, scheme)` for batch; `tokenize_single_cell(row, marker_cols, scheme)` for single cells (needs full preprocessed row with _rank/_bin). Aliases: `tokenize_dataframe()` and `tokenize_cell()`.
- **Notebooks path convention:** All notebooks use `PROJECT_ROOT = Path('..').resolve()` and build absolute paths from there. `sys.path.insert(0, str(PROJECT_ROOT))` for imports.
- **`preprocess()` API:** Takes keyword args (`arcsinh`, `arcsinh_cofactor`, `normalization`, `bin_count`, `bins`), NOT a config dict. Returns a single DataFrame, not a tuple. Use `preprocess_from_config(df, cfg_dict)` when calling from scripts/notebooks — this unpacks the config and returns `(df, log)`.
- **Do NOT call `preprocess(df, cfg['preprocessing'])`** — this is wrong. Always use `preprocess_from_config(df, cfg['preprocessing'])` at call sites that have a config dict.

## Do-Not-Repeat

<!-- Mistakes made and corrected. Each entry prevents the same mistake recurring. -->
<!-- Format: [YYYY-MM-DD] Description of what went wrong and what to do instead. -->

## Decision Log

<!-- Significant technical decisions with rationale. Why X was chosen over Y. -->
