#!/usr/bin/env bash
# ============================================================
# Run the full experiment matrix on RunPod — parallel edition
# Run this INSIDE the pod:  bash runpod/train.sh
#
# Trains all 6 combinations (2 models × 3 schemes).
# Runs PARALLEL_JOBS experiments simultaneously to saturate the GPU.
# ============================================================

set -euo pipefail

cd /workspace/MambaC2S

# How many experiments to run in parallel.
# 2 works well for 24GB GPUs (RTX 3090/4090).
# Increase to 3 for 48GB+ (A40, A100).
PARALLEL_JOBS=2

# Prep data if needed
if [ ! -f data/levine32_processed.h5ad ]; then
    echo "Processed data not found — running prepare_data.py ..."
    python scripts/prepare_data.py
fi
if [ ! -f data/split_manifest.json ]; then
    echo "Split manifest not found — running make_splits.py ..."
    python scripts/make_splits.py
fi

LOG_DIR="outputs/runpod_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Starting experiment matrix — $(date)"
echo "Parallel jobs : $PARALLEL_JOBS"
echo "Logs → $LOG_DIR"
echo ""

# Build job list
JOBS=()
for model in transformer mamba; do
    for scheme in rank_only strength_only hybrid; do
        JOBS+=("${model}|${scheme}")
    done
done

run_job() {
    local model="$1" scheme="$2" logfile="$3"
    echo "[START] $model / $scheme"
    python scripts/train_model.py \
        --config "configs/${model}.yaml" \
        --override \
            "tokenization.scheme=${scheme}" \
            "training.mixed_precision=true" \
            "training.batch_size=256" \
            "training.num_workers=4" \
        > "$logfile" 2>&1 \
        && echo "[DONE ] $model / $scheme" \
        || echo "[FAIL ] $model / $scheme — see $logfile"
}

export -f run_job
export LOG_DIR

# Run in parallel using a simple slot-based approach
PIDS=()
for job in "${JOBS[@]}"; do
    model="${job%%|*}"
    scheme="${job##*|}"
    logfile="$LOG_DIR/${model}_${scheme}.log"

    # Wait if we've filled all slots
    while [ ${#PIDS[@]} -ge "$PARALLEL_JOBS" ]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                unset 'PIDS[$i]'
                PIDS=("${PIDS[@]}")
            fi
        done
        sleep 2
    done

    run_job "$model" "$scheme" "$logfile" &
    PIDS+=($!)
done

# Wait for remaining jobs
wait

echo ""
echo "All runs complete — $(date)"
echo ""
python scripts/summarize_results.py --output-dir outputs/ 2>/dev/null || \
    ls -lt outputs/ | head -10
