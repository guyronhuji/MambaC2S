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

---

## Running Experiments

### Option 1 — Google Colab (easiest)

1. Open any notebook at **File → Open notebook → GitHub** → `guyronhuji/MambaC2S`
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Add this as the **first cell** and run it:

```python
import os
if not os.path.exists('/content/MambaC2S'):
    !git clone https://github.com/guyronhuji/MambaC2S.git /content/MambaC2S
%cd /content/MambaC2S
!pip install -q "numpy<2.0"
!pip install -q PyCytoData anndata umap-learn scikit-learn matplotlib seaborn pyyaml tabulate
import sys
sys.path.insert(0, '/content/MambaC2S')
```

4. Change `PROJECT_ROOT = Path('..').resolve()` to `Path('/content/MambaC2S').resolve()` in the notebook
5. Run all cells

---

### Option 2 — RunPod (recommended for full training)

**One-time local setup:**
```bash
brew install runpod/runpodctl/runpodctl

# Add your SSH public key at runpod.io → Settings → SSH Public Keys:
cat ~/.ssh/id_ed25519.pub
# (generate one first if needed: ssh-keygen -t ed25519)
```

**Create a pod** (shows live GPU prices, prompts you to choose):
```bash
./runpod/create_pod.sh
```

**SSH into the pod** — get the exact command from runpod.io → Pods → Connect → SSH.
It looks like:
```bash
ssh <podid>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

**Set up the pod** (run inside the pod — ~2 min, uses `uv` for fast installs):
```bash
bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/runpod/setup_pod.sh)
```

**Run the full experiment matrix** (6 combinations: 2 models × 3 schemes):
```bash
cd /workspace/MambaC2S && bash runpod/train.sh
```
`train.sh` automatically runs `prepare_data.py` and `make_splits.py` if not already done.

**Fetch results back to your local machine** (run this on your Mac, not the pod):
```bash
# Replace the SSH address with yours from the Connect page
bash runpod/fetch_results.sh <podid>-<hash>@ssh.runpod.io
```
Results are saved to `./outputs/runpod/`. The script uses `rsync` so re-running it
only copies new or changed files.

To check what results are available before fetching:
```bash
ssh <podid>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519 \
    "ls /workspace/MambaC2S/outputs/"
```

**Stop the pod when done** (from runpod.io console or):
```bash
runpodctl pod list   # get pod ID
runpodctl stop pod <POD_ID>
```

---

### Option 3 — GCP VM

**Prerequisites:** `brew install google-cloud-sdk && gcloud init`

```bash
./gcp/create_vm.sh          # scans zones, picks best available GPU
gcloud compute ssh mambac2s-vm --zone=<ZONE>
bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/gcp/setup_vm.sh)
bash gcp/train.sh
# Back on local:
bash gcp/fetch_results.sh <ZONE>
gcloud compute instances stop mambac2s-vm --zone=<ZONE>
```

> **Note:** New GCP projects have `GPUS_ALL_REGIONS` quota = 0. Request an increase at
> console.cloud.google.com/iam-admin/quotas before running.

---

### Option 4 — Azure VM

**Prerequisites:** `brew install azure-cli && az login`

```bash
./azure/create_vm.sh        # creates NC4as T4 v3 VM (~$0.50/hr)
ssh azureuser@<VM-IP>
bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/azure/setup_vm.sh)
bash azure/train.sh
# Back on local:
bash azure/fetch_results.sh <VM-IP>
az vm deallocate -g mambaC2S-rg -n mambaC2S-vm
```

---

### Option 5 — Local (Mac with Apple Silicon)

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/make_splits.py
python scripts/train_model.py --config configs/transformer.yaml
python scripts/train_model.py --config configs/mamba.yaml
```

`device: "auto"` in `configs/base.yaml` picks MPS on Apple Silicon automatically.

---

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
  notebooks/         Jupyter notebooks (01–04)
  outputs/           Experiment artifacts (not tracked)
  docs/              Analysis LLM instructions and schemas
  runpod/            RunPod pod scripts
  gcp/               GCP VM scripts
  azure/             Azure VM scripts
```

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

## Reproducibility

All experiments use fixed seeds. Configs, splits, and environment versions are
saved at experiment start. See `src/utils/reproducibility.py`.

## Analysis

See `docs/README_for_analysis_llms.md` for instructions on interpreting results.
See `docs/EXPERIMENT_REGISTRY.md` for a log of all experiments run.
