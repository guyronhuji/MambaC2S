#!/usr/bin/env bash
# ============================================================
# Run the full experiment matrix on Azure VM
# Run this INSIDE the VM:  bash azure/train.sh
#
# Trains all 6 combinations:
#   models: transformer, mamba
#   schemes: rank_only, strength_only, hybrid
# ============================================================

set -euo pipefail

cd ~/MambaC2S
source ~/venv/bin/activate

LOG_DIR="outputs/azure_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Enable mixed precision for CUDA speed
export EXTRA_ARGS="--mixed-precision"

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
            --model "$model" \
            --scheme "$scheme" \
            --mixed-precision \
            2>&1 | tee "$LOG_DIR/${model}_${scheme}.log"
        echo ""
    done
done

echo "All runs complete — $(date)"
echo ""
echo "Results:"
python scripts/summarize_results.py --output-dir outputs/ 2>/dev/null || \
    ls -lt outputs/ | head -10
