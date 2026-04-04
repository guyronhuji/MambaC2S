#!/usr/bin/env bash
# ============================================================
# Run the full experiment matrix on RunPod — parallel edition
# Runs PARALLEL_JOBS experiments simultaneously with a live
# progress table that refreshes every 5 seconds.
# ============================================================

set -euo pipefail

cd /workspace/MambaC2S

PARALLEL_JOBS=2   # 2 for 24GB (RTX 3090/4090), 3 for 48GB+ (A40/A100)
REFRESH=5         # seconds between display updates

# Prep data if needed
if [ ! -f data/levine32_processed.h5ad ]; then
    echo "Preparing data ..."
    python scripts/prepare_data.py
fi
if [ ! -f data/split_manifest.json ]; then
    echo "Creating splits ..."
    python scripts/make_splits.py
fi

# Resume the most recent run dir if it has done/failed jobs; else start fresh
PREV_DIR=$(ls -dt outputs/runpod_run_* 2>/dev/null | head -1 || true)
if [ -n "$PREV_DIR" ] && ls "$PREV_DIR"/*.status 2>/dev/null | xargs grep -ql "done\|failed" 2>/dev/null; then
    LOG_DIR="$PREV_DIR"
    echo "Resuming previous run: $LOG_DIR"
else
    LOG_DIR="outputs/runpod_run_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

echo "Experiment matrix — $(date)"
echo "Parallel jobs : $PARALLEL_JOBS   Refresh : ${REFRESH}s"
echo "Logs → $LOG_DIR"
echo ""

ALL_JOBS=()
for model in transformer mamba; do
    for scheme in rank_only strength_only hybrid; do
        ALL_JOBS+=("${model}|${scheme}")
    done
done

# Skip jobs already completed or failed
JOBS=()
for job in "${ALL_JOBS[@]}"; do
    model="${job%%|*}"
    scheme="${job##*|}"
    statusfile="$LOG_DIR/${model}_${scheme}.status"
    if [ -f "$statusfile" ] && grep -q "done" "$statusfile"; then
        echo "  skipping ${model}/${scheme} (already done)"
    else
        JOBS+=("$job")
    fi
done

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "All experiments already completed."
    exit 0
fi
echo ""

# ── Job runner (runs in background) ─────────────────────────
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

# Start rich monitor in foreground (it exits when all jobs finish)
python runpod/monitor.py --log-dir "$LOG_DIR" --jobs "${#ALL_JOBS[@]}" --refresh "$REFRESH" &
MONITOR_PID=$!

# ── Parallel job runner ──────────────────────────────────────
PIDS=()
for job in "${JOBS[@]}"; do
    model="${job%%|*}"
    scheme="${job##*|}"
    logfile="$LOG_DIR/${model}_${scheme}.log"
    statusfile="$LOG_DIR/${model}_${scheme}.status"
    echo "queued" > "$statusfile"

    while [ ${#PIDS[@]} -ge "$PARALLEL_JOBS" ]; do
        wait -n 2>/dev/null || true   # reap one finished job (removes zombie)
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
