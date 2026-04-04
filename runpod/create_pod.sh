#!/usr/bin/env bash
# ============================================================
# Create a RunPod GPU pod for MambaC2S training
# Queries the RunPod API for real-time GPU availability.
#
# Prerequisites (one-time):
#   1. brew install runpod/runpodctl/runpodctl
#   2. Add SSH key at runpod.io → Settings → SSH Public Keys:
#      cat ~/.ssh/id_ed25519.pub
#
# Usage:
#   chmod +x runpod/create_pod.sh
#   ./runpod/create_pod.sh
# ============================================================

set -euo pipefail

POD_NAME="mambac2s"
IMAGE="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
DISK_GB=50

# Load API key
KEYFILE="$(dirname "$0")/../.runpodkey"
if [ -f "$KEYFILE" ]; then
  API_KEY=$(awk '{print $2}' "$KEYFILE")
  mkdir -p ~/.runpod
  printf "apiKey: %s\napiUrl: https://api.runpod.io/graphql\n" "$API_KEY" > ~/.runpod/.runpod.yaml
else
  API_KEY=""
fi

echo "Querying RunPod for available GPUs ..."
echo ""

# Query the GraphQL API for real-time availability
GPU_LIST=$(RUNPOD_API_KEY="$API_KEY" python3 - <<'PYEOF'
import os, json, ssl, urllib.request, urllib.error

api_key = os.environ["RUNPOD_API_KEY"]

# Build SSL context — use certifi if available, otherwise skip verification
try:
    import certifi
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    ssl_ctx = ssl._create_unverified_context()

query = """
query {
  gpuTypes {
    id
    displayName
    memoryInGb
    lowestPrice(input: {gpuCount: 1}) {
      minimumBidPrice
      uninterruptablePrice
      stockStatus
    }
  }
}
"""

url = f"https://api.runpod.io/graphql?api_key={api_key}"
req = urllib.request.Request(
    url,
    data=json.dumps({"query": query}).encode(),
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req, timeout=15, context=ssl_ctx) as r:
        data = json.load(r)
except urllib.error.URLError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)

gpus = data.get("data", {}).get("gpuTypes", [])

# Filter to GPUs with actual stock
available = []
for g in gpus:
    lp = g.get("lowestPrice") or {}
    stock = lp.get("stockStatus") or ""
    spot  = lp.get("minimumBidPrice")
    od    = lp.get("uninterruptablePrice")
    if stock.lower() in ("high", "medium", "low") and od:
        available.append({
            "id":    g["id"],
            "name":  g["displayName"],
            "vram":  g.get("memoryInGb", "?"),
            "spot":  f"${spot:.2f}" if spot else "-",
            "od":    f"${od:.2f}",
            "stock": stock,
        })

# Sort by on-demand price
available.sort(key=lambda x: float(x["od"].strip("$")))
print(json.dumps(available))
PYEOF
)

if [ -z "$GPU_LIST" ] || [ "$GPU_LIST" = "[]" ]; then
  echo "No GPUs available right now. Try again shortly."
  exit 1
fi

# Display table and build selection arrays
eval "$(python3 - "$GPU_LIST" <<'PYEOF'
import sys, json

gpus = json.loads(sys.argv[1])

STOCK_ICON = {"High": "●●●", "Medium": "●●○", "Low": "●○○"}

print(f'printf "\\n%-4s %-34s %8s %10s %14s %8s\\n" "№" "GPU" "VRAM(GB)" "SPOT \\$/hr" "ON-DEMAND \\$/hr" "STOCK"')
print(f'printf "%-4s %-34s %8s %10s %14s %8s\\n" "---" "---------------------------------" "--------" "----------" "--------------" "-----"')

names = []
for i, g in enumerate(gpus):
    icon = STOCK_ICON.get(g["stock"], "?")
    print(f'printf "%-4s %-34s %8s %10s %14s %8s\\n" "{i+1}" "{g["name"]}" "{g["vram"]}" "{g["spot"]}" "{g["od"]}" "{icon}"')
    names.append(g["id"])

# Export arrays for bash
ids_str = " ".join(f'"{g["id"]}"' for g in gpus)
names_str = " ".join(f'"{g["name"]}"' for g in gpus)
print(f'GPU_IDS=({ids_str})')
print(f'GPU_NAMES=({names_str})')
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

echo ""
echo "Creating pod with: $SELECTED_NAME"

OUTPUT=$(runpodctl create pod \
  --name "$POD_NAME" \
  --gpuType "$SELECTED_NAME" \
  --imageName "$IMAGE" \
  --containerDiskSize "$DISK_GB" \
  --volumeSize 5 \
  --volumePath "/workspace" \
  --startSSH \
  2>&1) && {
  POD_ID=$(echo "$OUTPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null || \
           echo "$OUTPUT" | grep -oE '"id":"[^"]*"' | head -1 | cut -d'"' -f4 || true)
  echo ""
  echo "============================================================"
  echo "  Pod created!"
  echo "  GPU    : $SELECTED_NAME"
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
  [ -n "$POD_ID" ] && echo "    runpodctl stop pod $POD_ID" || echo "    runpodctl pod list  →  stop by ID"
  echo "============================================================"
} || {
  echo "ERROR: $OUTPUT"
  exit 1
}
