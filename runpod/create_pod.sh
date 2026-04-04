#!/usr/bin/env bash
# ============================================================
# Create a RunPod GPU pod for MambaC2S training
# Shows available GPUs with prices and lets you choose.
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

# Auto-configure from .runpodkey if present
KEYFILE="$(dirname "$0")/../.runpodkey"
if [ -f "$KEYFILE" ]; then
  API_KEY=$(awk '{print $2}' "$KEYFILE")
  mkdir -p ~/.runpod
  printf "apiKey: %s\napiUrl: https://api.runpod.io/graphql\n" "$API_KEY" > ~/.runpod/.runpod.yaml
fi

echo "Fetching available GPUs ..."
echo ""

# Get cloud list, skip header, skip "Reserved" rows, clean up multi-line names
RAW=$(runpodctl get cloud 2>/dev/null | tail -n +2 | grep -v "^$" | grep -v "Reserved")

# Parse into arrays using Python for robustness
GPU_LIST=$(python3 - <<'PYEOF'
import sys, re

lines = """$RAW""".strip().split("\n")

gpus = []
for line in lines:
    # tab-separated: GPU TYPE  MEM GB  VCPU  SPOT $/HR  ONDEMAND $/HR
    parts = re.split(r'\t', line.strip())
    if len(parts) >= 5:
        name  = parts[0].strip()
        mem   = parts[1].strip()
        vcpu  = parts[2].strip()
        spot  = parts[3].strip()
        od    = parts[4].strip()
        if name and mem and od and od != "Reserved":
            gpus.append(f"{name}|{mem}|{vcpu}|{spot}|{od}")

for g in gpus:
    print(g)
PYEOF
)

# Fallback: use awk if python approach fails
if [ -z "$GPU_LIST" ]; then
  GPU_LIST=$(runpodctl get cloud 2>/dev/null \
    | tail -n +2 \
    | grep -v "^$" \
    | grep -v "Reserved" \
    | awk 'NF>=5 {
        name=""; for(i=1;i<=NF-4;i++) name=name" "$i;
        gsub(/^ /,"",name);
        printf "%s|%s|%s|%s|%s\n", name, $(NF-3), $(NF-2), $(NF-1), $NF
      }')
fi

# Build indexed arrays
declare -a GPU_NAMES GPU_MEM GPU_VCPU GPU_SPOT GPU_OD
i=0
while IFS='|' read -r name mem vcpu spot od; do
  [ -z "$name" ] && continue
  GPU_NAMES[$i]="$name"
  GPU_MEM[$i]="$mem"
  GPU_VCPU[$i]="$vcpu"
  GPU_SPOT[$i]="$spot"
  GPU_OD[$i]="$od"
  i=$((i+1))
done <<< "$GPU_LIST"

if [ ${#GPU_NAMES[@]} -eq 0 ]; then
  echo "ERROR: Could not fetch GPU list. Check your API key."
  exit 1
fi

# Display table
printf "\n%-4s %-34s %8s %6s %12s %14s\n" "№" "GPU" "VRAM(GB)" "vCPU" "SPOT \$/hr" "ON-DEMAND \$/hr"
printf "%-4s %-34s %8s %6s %12s %14s\n" "---" "---------------------------------" "--------" "------" "----------" "--------------"
for j in "${!GPU_NAMES[@]}"; do
  printf "%-4s %-34s %8s %6s %12s %14s\n" \
    "$((j+1))" "${GPU_NAMES[$j]}" "${GPU_MEM[$j]}" "${GPU_VCPU[$j]}" "${GPU_SPOT[$j]}" "${GPU_OD[$j]}"
done

echo ""
read -p "Choose GPU number (1-${#GPU_NAMES[@]}): " CHOICE

if ! [[ "$CHOICE" =~ ^[0-9]+$ ]] || [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt ${#GPU_NAMES[@]} ]; then
  echo "Invalid choice."
  exit 1
fi

IDX=$((CHOICE-1))
SELECTED="${GPU_NAMES[$IDX]}"

echo ""
echo "Selected: $SELECTED  (on-demand: \$${GPU_OD[$IDX]}/hr)"
echo "Creating pod ..."

OUTPUT=$(runpodctl create pod \
  --name "$POD_NAME" \
  --gpuType "$SELECTED" \
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
  echo "  GPU     : $SELECTED"
  [ -n "$POD_ID" ] && echo "  Pod ID  : $POD_ID"
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
  echo ""
  echo "ERROR: $OUTPUT"
  exit 1
}
