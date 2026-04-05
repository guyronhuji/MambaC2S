#!/usr/bin/env bash
# ============================================================
# Run the MLPAutoencoder latent dimension sweep on RunPod.
# Trains d ∈ {2, 4, 8, 16, 32} sequentially on GPU.
#
# Usage (inside the pod):
#   bash runpod/train_latent_sweep.sh
#
# To sweep a custom set of dims or epochs:
#   bash runpod/train_latent_sweep.sh --latent-dims 8 16 32 --epochs 150
#
# Results land in:
#   outputs/nb06_latent_sweep/latent_{d:02d}/
#
# Fetch them back with (from your Mac):
#   bash runpod/fetch_results.sh
# Then open notebooks/06_latent_dim_sweep.ipynb for analysis.
# ============================================================

set -euo pipefail

cd /workspace/MambaC2S

# ── Prepare data if not already done ────────────────────────
if [ ! -f data/levine32_processed.h5ad ]; then
    echo "=== Preparing data ==="
    python scripts/prepare_data.py
fi

if [ ! -f data/split_manifest.json ]; then
    echo "=== Creating splits ==="
    python scripts/make_splits.py
fi

# ── Run sweep ───────────────────────────────────────────────
echo "=== Starting latent dim sweep — $(date) ==="

python scripts/latent_dim_sweep.py \
    --device cuda \
    --mixed-precision \
    --epochs 100 \
    --patience 15 \
    --batch-size 512 \
    --num-workers 4 \
    "$@"

echo ""
echo "=== Sweep complete — $(date) ==="
echo ""
echo "Checkpoints saved to: outputs/nb06_latent_sweep/"
echo ""
echo "To fetch results back to your Mac:"
echo "  bash runpod/fetch_results.sh"
