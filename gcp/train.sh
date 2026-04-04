#!/usr/bin/env bash
# ============================================================
# Run the full experiment matrix on GCP VM
# Run this INSIDE the VM:  bash gcp/train.sh
#
# Trains all 6 combinations:
#   models:  transformer, mamba
#   schemes: rank_only, strength_only, hybrid
# ============================================================

set -euo pipefail

cd ~/MambaC2S
source ~/venv/bin/activate

LOG_DIR="outputs/gcp_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

models=(transformer mamba)
schemes=(rank_only strength_only hybrid)

echo "Starting experiment matrix — $(date)"
echo "Logs → $LOG_DIR"
echo ""

for model in "${models[@]}"; do
    for scheme in "${schemes[@]}"; do
        echo "──────────────────────────────────────"
        echo "  Training: $model / $scheme"
        echo "──────────────────────────────────────"
        python scripts/train_model.py \
            --config "configs/${model}.yaml" \
            --override "tokenization.scheme=${scheme}" "training.mixed_precision=true" \
            2>&1 | tee "$LOG_DIR/${model}_${scheme}.log"
        echo ""
    done
done

echo "All runs complete — $(date)"
echo ""
python scripts/summarize_results.py --output-dir outputs/ 2>/dev/null || \
    ls -lt outputs/ | head -10
