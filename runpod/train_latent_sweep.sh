#!/usr/bin/env bash
# ============================================================
# Run the MLPAutoencoder latent dimension sweep on RunPod.
# Trains d ∈ {2, 4, 8, 16, 32} in parallel — one process per
# dim, each gets its own GPU time-slice.
#
# Usage (inside the pod):
#   bash runpod/train_latent_sweep.sh
#
# Limit concurrency on smaller GPUs (e.g. 16GB):
#   PARALLEL_JOBS=3 bash runpod/train_latent_sweep.sh
#
# Extra args are forwarded to every latent_dim_sweep.py call:
#   bash runpod/train_latent_sweep.sh --epochs 150 --force
#
# Results land in:
#   outputs/nb06_latent_sweep/latent_{d:02d}/
#
# Fetch them back with (from your Mac):
#   bash runpod/fetch_results.sh
# ============================================================

set -euo pipefail

cd /workspace/MambaC2S

LATENT_DIMS=(2 4 8 16 32)
PARALLEL_JOBS="${PARALLEL_JOBS:-5}"   # default: all dims at once

# ── Prepare data if not already done ────────────────────────
if [ ! -f data/levine32_processed.h5ad ]; then
    echo "=== Preparing data ==="
    python scripts/prepare_data.py
fi

if [ ! -f data/split_manifest.json ]; then
    echo "=== Creating splits ==="
    python scripts/make_splits.py
fi

# ── Launch one process per dim ───────────────────────────────
LOG_DIR="outputs/nb06_latent_sweep/logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Latent dim sweep — $(date) ==="
echo "Dims          : ${LATENT_DIMS[*]}"
echo "Parallel jobs : $PARALLEL_JOBS"
echo "Logs          : $LOG_DIR"
echo ""

PIDS=()

for d in "${LATENT_DIMS[@]}"; do
    logfile="$LOG_DIR/latent_${d}.log"

    # Semaphore: wait until a slot is free
    while [ ${#PIDS[@]} -ge "$PARALLEL_JOBS" ]; do
        wait -n 2>/dev/null || true
        new_pids=()
        for pid in "${PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
        done
        PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    done

    echo "▶  d=$d  (log: latent_${d}.log)"
    python scripts/latent_dim_sweep.py \
        --latent-dims "$d" \
        --device cuda \
        --mixed-precision \
        --epochs 100 \
        --patience 15 \
        --batch-size 512 \
        --num-workers 4 \
        "$@" \
        > "$logfile" 2>&1 &
    PIDS+=($!)
done

# Wait for all remaining jobs
wait

echo ""
echo "=== All dims complete — $(date) ==="
echo ""

# ── Print summary from each training_summary.json ────────────
echo "Results:"
printf "%-8s  %-10s  %-8s\n" "d_model" "val_mse" "best_ep"
printf "%-8s  %-10s  %-8s\n" "-------" "-------" "-------"
for d in "${LATENT_DIMS[@]}"; do
    summ="outputs/nb06_latent_sweep/latent_$(printf '%02d' $d)/training_summary.json"
    if [ -f "$summ" ]; then
        python3 -c "
import json
s = json.load(open('$summ'))
print(f'{s[\"d_model\"]:<8}  {s[\"best_val_loss\"]:<10.5f}  {s[\"best_epoch\"]:<8}')
"
    else
        printf "%-8s  %-10s  %-8s\n" "$d" "MISSING" "-"
    fi
done

echo ""
echo "Checkpoints → outputs/nb06_latent_sweep/"
echo ""
echo "To fetch results back to your Mac:"
echo "  bash runpod/fetch_results.sh"
