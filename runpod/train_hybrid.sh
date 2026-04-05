#!/usr/bin/env bash
# ============================================================
# Run hybrid-scheme experiments: Transformer, LSTM, GRU
# 3 jobs — runs them in parallel.
# ============================================================

set -euo pipefail

cd /workspace/MambaC2S

PARALLEL_JOBS=3
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

ALL_JOBS=(
    "configs/transformer_hybrid.yaml"
    "configs/lstm_hybrid.yaml"
    "configs/gru_hybrid.yaml"
)

run_job() {
    local config="$1" logfile="$2" statusfile="$3"
    echo "running" > "$statusfile"
    python scripts/train_model.py \
        --config "$config" \
        --override \
            "device=cuda" \
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
for config in "${ALL_JOBS[@]}"; do
    name="${config##*/}"
    name="${name%.yaml}"
    logfile="$LOG_DIR/${name}.log"
    statusfile="$LOG_DIR/${name}.status"
    echo "queued" > "$statusfile"

    while [ ${#PIDS[@]} -ge "$PARALLEL_JOBS" ]; do
        wait -n 2>/dev/null || true
        new_pids=()
        for pid in "${PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
        done
        PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    done

    run_job "$config" "$logfile" "$statusfile" &
    PIDS+=($!)
done

wait
kill "$MONITOR_PID" 2>/dev/null || true

echo ""
echo "All runs complete — $(date)"
echo ""
python scripts/summarize_results.py --output-dir outputs/ 2>/dev/null || \
    ls -lt outputs/ | head -10
