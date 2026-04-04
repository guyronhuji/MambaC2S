#!/usr/bin/env bash
# ============================================================
# Create a RunPod GPU pod for MambaC2S training
#
# Prerequisites (one-time):
#   1. brew install runpod/runpodctl/runpodctl
#   2. runpodctl config set --apiKey <YOUR_KEY>
#      (or keep ~/.runpodkey with "apiKey <KEY>")
#   3. Add SSH key at runpod.io → Settings → SSH Public Keys
#
# Usage:
#   chmod +x runpod/create_pod.sh
#   ./runpod/create_pod.sh
# ============================================================

set -euo pipefail

POD_NAME="mambac2s"
IMAGE="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
DISK_GB=50

# Load API key (also writes runpodctl config)
KEYFILE="$(dirname "$0")/../.runpodkey"
if [ -f "$KEYFILE" ]; then
  API_KEY=$(awk '{print $2}' "$KEYFILE")
  mkdir -p ~/.runpod
  printf "apiKey: %s\napiUrl: https://api.runpod.io/graphql\n" "$API_KEY" > ~/.runpod/.runpod.yaml
else
  API_KEY=""
fi

echo "Querying RunPod for available GPUs ..."

# ── 1. runpodctl gpu list: authoritative GPU IDs + cloud types ─
GPU_RAW=$(runpodctl gpu list 2>/dev/null) || { echo "ERROR: runpodctl gpu list failed"; exit 1; }

# ── 2. GraphQL: spot + on-demand prices ───────────────────────
QUERY='{"query":"{ gpuTypes { id displayName lowestPrice(input: {gpuCount: 1}) { minimumBidPrice uninterruptablePrice } } }"}'
PRICE_JSON=$(curl -sf -X POST "https://api.runpod.io/graphql?api_key=${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$QUERY") || PRICE_JSON="{}"

# ── 3. runpodctl probes: actual availability counts ───────────
echo "Probing availability counts ..."
PROBE_1=$(runpodctl get cloud 1 2>/dev/null | grep -v "^warning\|^GPU TYPE\|^$" || true)
PROBE_2=$(runpodctl get cloud 2 2>/dev/null | grep -v "^warning\|^GPU TYPE\|^$" || true)
PROBE_4=$(runpodctl get cloud 4 2>/dev/null | grep -v "^warning\|^GPU TYPE\|^$" || true)
PROBE_8=$(runpodctl get cloud 8 2>/dev/null | grep -v "^warning\|^GPU TYPE\|^$" || true)
PROBE_16=$(runpodctl get cloud 16 2>/dev/null | grep -v "^warning\|^GPU TYPE\|^$" || true)

# ── 4. Merge and display ──────────────────────────────────────
GPU_LIST=$(GPU_RAW="$GPU_RAW" PRICE_JSON="$PRICE_JSON" \
           PROBE_1="$PROBE_1" PROBE_2="$PROBE_2" \
           PROBE_4="$PROBE_4" PROBE_8="$PROBE_8" PROBE_16="$PROBE_16" \
           python3 - <<'PYEOF'
import os, json

gpu_list  = json.loads(os.environ["GPU_RAW"])
price_raw = json.loads(os.environ.get("PRICE_JSON", "{}"))

# Build price lookup: gpuId → {spot, od}
prices = {}
for g in price_raw.get("data", {}).get("gpuTypes", []):
    lp = g.get("lowestPrice") or {}
    spot = lp.get("minimumBidPrice")
    od   = lp.get("uninterruptablePrice")
    prices[g["id"]] = {
        "spot": f"${spot:.2f}" if spot else "-",
        "od":   f"${od:.2f}"  if od   else "-",
    }

probes = {
    16: os.environ["PROBE_16"],
    8:  os.environ["PROBE_8"],
    4:  os.environ["PROBE_4"],
    2:  os.environ["PROBE_2"],
    1:  os.environ["PROBE_1"],
}

def max_avail(gpu_id: str) -> str:
    for n in [16, 8, 4, 2, 1]:
        if gpu_id in probes[n]:
            return f">={n}"
    return "?"

available = []
for g in gpu_list:
    if not g.get("available"):
        continue
    gpu_id = g["gpuId"]
    comm   = g.get("communityCloud", False)
    secure = g.get("secureCloud", False)
    cloud  = "COMMUNITY" if comm else "SECURE"
    cloud_label = "C" if (comm and not secure) else ("S" if (secure and not comm) else "C+S")
    p = prices.get(gpu_id, {"spot": "-", "od": "-"})
    available.append({
        "id":     gpu_id,
        "name":   g["displayName"],
        "vram":   g.get("memoryInGb", "?"),
        "spot":   p["spot"],
        "od":     p["od"],
        "avail":  max_avail(gpu_id),
        "cloud":  cloud,       # used when creating pod
        "cloudl": cloud_label, # display label
    })

# Sort by on-demand price (treat "-" as very expensive)
def od_sort(x):
    try:    return float(x["od"].strip("$"))
    except: return 999
available.sort(key=od_sort)

print(json.dumps(available))
PYEOF
)

if [ -z "$GPU_LIST" ] || [ "$GPU_LIST" = "[]" ]; then
  echo "No GPUs available right now. Try again shortly."
  exit 1
fi

# Display table
RUNPOD_GPU_LIST="$GPU_LIST" python3 - <<'PYEOF'
import os, json
gpus = json.loads(os.environ["RUNPOD_GPU_LIST"])
print(f"\n{'#':<4} {'GPU':<28} {'VRAM':>6} {'SPOT $/hr':>10} {'OD $/hr':>9} {'AVAIL':>7} {'CLOUD':>6}")
print(f"{'---':<4} {'-'*28} {'------':>6} {'----------':>10} {'-'*9:>9} {'-------':>7} {'-----':>6}")
for i, g in enumerate(gpus):
    print(f"{i+1:<4} {g['name']:<28} {str(g['vram'])+'GB':>6} {g['spot']:>10} {g['od']:>9} {g['avail']:>7} {g['cloudl']:>6}")
PYEOF

# Export arrays for bash
eval "$(RUNPOD_GPU_LIST="$GPU_LIST" python3 - <<'PYEOF'
import os, json
gpus = json.loads(os.environ["RUNPOD_GPU_LIST"])
ids_str    = " ".join(f'"{g["id"]}"'    for g in gpus)
names_str  = " ".join(f'"{g["name"]}"'  for g in gpus)
clouds_str = " ".join(f'"{g["cloud"]}"' for g in gpus)
print(f'GPU_IDS=({ids_str})')
print(f'GPU_NAMES=({names_str})')
print(f'GPU_CLOUDS=({clouds_str})')
print(f'TOTAL={len(gpus)}')
PYEOF
)"

echo ""
read -rp "Choose GPU number (1-${TOTAL}): " CHOICE

if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt "$TOTAL" ]; then
  echo "Invalid choice."
  exit 1
fi

IDX=$((CHOICE-1))
SELECTED_ID="${GPU_IDS[$IDX]}"
SELECTED_NAME="${GPU_NAMES[$IDX]}"
SELECTED_CLOUD="${GPU_CLOUDS[$IDX]}"

echo ""
echo "Creating pod: $SELECTED_NAME ($SELECTED_CLOUD cloud) ..."

OUTPUT=$(runpodctl pod create \
  --name "$POD_NAME" \
  --gpu-id "$SELECTED_ID" \
  --image "$IMAGE" \
  --container-disk-in-gb "$DISK_GB" \
  --volume-in-gb 5 \
  --volume-mount-path "/workspace" \
  --cloud-type "$SELECTED_CLOUD" \
  --ssh \
  2>&1) && {
  POD_ID=$(echo "$OUTPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null || \
           echo "$OUTPUT" | grep -oE '"id":"[^"]*"' | head -1 | cut -d'"' -f4 || true)
  echo ""
  echo "============================================================"
  echo "  Pod created!"
  echo "  GPU    : $SELECTED_NAME"
  echo "  Cloud  : $SELECTED_CLOUD"
  [ -n "$POD_ID" ] && echo "  Pod ID : $POD_ID"
  echo ""
  echo "  Wait ~60s, then get your SSH command from:"
  echo "    https://www.runpod.io/console/pods"
  echo "    Click pod → Connect → 'SSH over exposed TCP'"
  echo ""
  echo "  Once inside, run setup:"
  echo "    bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/runpod/setup_pod.sh)"
  echo ""
  echo "  STOP pod when done:"
  [ -n "$POD_ID" ] && echo "    runpodctl pod stop $POD_ID" || echo "    runpodctl pod list  →  stop by ID"
  echo "============================================================"
} || {
  echo "ERROR: $OUTPUT"
  exit 1
}
