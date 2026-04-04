#!/usr/bin/env bash
# ============================================================
# Run hybrid-scheme experiments only: transformer + mamba
# 2 jobs total — runs them in parallel.
# ============================================================

set -euo pipefail

cd /workspace/MambaC2S

PARALLEL_JOBS=2   # both jobs run simultaneously
REFRESH=5

# Prep data if needed
if [ ! -f data/levine32_processed.h5ad ]; then
    echo "Preparing data ..."
    python scripts/prepare_data.py
fi
if [ ! -f data/split_manifest.json ]; then
    echo "Creating splits ..."
    python scripts/make_splits.py
fi

LOG_DIR="outputs/runpod_hybrid_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Hybrid-scheme experiment run — $(date)"
echo "Logs → $LOG_DIR"
echo ""

ALL_JOBS=(transformer|hybrid mamba|hybrid)

run_job() {
    local model="$1" scheme="$2" logfile="$3" statusfile="$4"
    echo "running" > "$statusfile"
    python scripts/train_model.py \
        --config "configs/${model}.yaml" \
        --override \
            "tokenization.scheme=${scheme}" \
            "training.mixed_precision=true" \
            "training.batch_size=256" \
            "training.num_workers=4" \
        > "$logfile" 2>&1 \
        && echo "done" > "$statusfile" \
        || echo "failed" > "$statusfile"
}
export -f run_job

python runpod/monitor.py --log-dir "$LOG_DIR" --jobs "${#ALL_JOBS[@]}" --refresh "$REFRESH" &
MONITOR_PID=$!

PIDS=()
for job in "${ALL_JOBS[@]}"; do
    model="${job%%|*}"
    scheme="${job##*|}"
    logfile="$LOG_DIR/${model}_${scheme}.log"
    statusfile="$LOG_DIR/${model}_${scheme}.status"
    echo "queued" > "$statusfile"

    while [ ${#PIDS[@]} -ge "$PARALLEL_JOBS" ]; do
        wait -n 2>/dev/null || true
        new_pids=()
        for pid in "${PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
        done
        PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    done

    run_job "$model" "$scheme" "$logfile" "$statusfile" &
    PIDS+=($!)
done

wait
kill "$MONITOR_PID" 2>/dev/null || true

echo ""
echo "All runs complete — $(date)"
echo ""
python scripts/summarize_results.py --output-dir outputs/ 2>/dev/null || \
    ls -lt outputs/ | head -10
