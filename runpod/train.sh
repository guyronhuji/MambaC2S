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

LOG_DIR="outputs/runpod_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Experiment matrix — $(date)"
echo "Parallel jobs : $PARALLEL_JOBS   Refresh : ${REFRESH}s"
echo "Logs → $LOG_DIR"
echo ""

JOBS=()
for model in transformer mamba; do
    for scheme in rank_only strength_only hybrid; do
        JOBS+=("${model}|${scheme}")
    done
done

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

# ── Live monitor (runs in background) ───────────────────────
monitor() {
    local log_dir="$1"
    # Map experiment name → output dir (most recent match)
    while true; do
        sleep "$REFRESH"
        # Move cursor up to overwrite previous table
        # Count lines: header(3) + one per job entry
        local entries
        entries=$(python3 - "$log_dir" <<'PYEOF'
import sys, os, csv, glob, re

log_dir = sys.argv[1]
rows = []

for logfile in sorted(glob.glob(f"{log_dir}/*.log")):
    name = os.path.basename(logfile).replace(".log", "").replace("_", "/", 1)
    statusfile = logfile.replace(".log", ".status")
    status = open(statusfile).read().strip() if os.path.exists(statusfile) else "queued"

    # Parse output dir from log line: "Experiment ID: xxx  →  outputs/xxx"
    exp_dir = ""
    max_epochs = "?"
    try:
        for line in open(logfile):
            if "Experiment ID:" in line and "→" in line:
                exp_dir = line.split("→")[-1].strip()
            if "max_epochs=" in line:
                m = re.search(r"max_epochs=(\d+)", line)
                if m:
                    max_epochs = m.group(1)
    except Exception:
        pass

    epoch, train_loss, val_loss, val_ppl = "-", "-", "-", "-"
    if exp_dir and os.path.exists(f"{exp_dir}/training_log.csv"):
        try:
            with open(f"{exp_dir}/training_log.csv") as f:
                rows_csv = list(csv.DictReader(f))
            if rows_csv:
                last = rows_csv[-1]
                epoch      = last.get("epoch", "-")
                train_loss = f"{float(last['train_loss']):.3f}" if last.get("train_loss") else "-"
                val_loss   = f"{float(last['val_loss']):.3f}"   if last.get("val_loss")   else "-"
                val_ppl    = f"{float(last['val_perplexity']):.1f}" if last.get("val_perplexity") else "-"
        except Exception:
            pass

    icon = {"running": ">>", "done": "OK", "failed": "!!", "queued": ".."}. get(status, "??")
    ep_str = f"{epoch}/{max_epochs}" if epoch != "-" else "-"
    rows.append((icon, name, ep_str, train_loss, val_loss, val_ppl))

n = len(rows)
print(f"\033[{n+3}A", end="")
print(f"\033[2K{'':2} {'Experiment':<30} {'Epoch':<9} {'TrLoss':<9} {'ValLoss':<9} {'Pplx'}")
print(f"\033[2K{'--'} {'-'*30} {'-'*9} {'-'*9} {'-'*9} {'----'}")
for icon, name, ep, tl, vl, ppl in rows:
    print(f"\033[2K{icon} {name:<30} {ep:<9} {tl:<9} {vl:<9} {ppl}")
PYEOF
        )
        printf "%s\n" "$entries"
    done
}

# Print blank lines for the monitor to overwrite
for job in "${JOBS[@]}"; do echo ""; done
echo ""; echo ""  # header lines

# Start monitor in background
monitor "$LOG_DIR" &
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
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                unset 'PIDS[$i]'
                PIDS=("${PIDS[@]+"${PIDS[@]}"}")
            fi
        done
        sleep 2
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
