#!/usr/bin/env bash
# ============================================================
# Create a RunPod GPU pod for MambaC2S training
#
# Prerequisites (one-time):
#   1. brew install runpod/runpodctl/runpodctl
#   2. Add SSH key at runpod.io → Settings → SSH Public Keys:
#      cat ~/.ssh/id_ed25519.pub
#      (generate if needed: ssh-keygen -t ed25519)
#
# Usage:
#   chmod +x runpod/create_pod.sh
#   ./runpod/create_pod.sh
# ============================================================

set -euo pipefail

POD_NAME="mambac2s"
IMAGE="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
DISK_GB=50

# Available GPU types (non-Reserved, cheapest first)
GPU_TYPES=(
  "NVIDIA GeForce RTX 3090"
  "NVIDIA GeForce RTX 4090"
  "NVIDIA RTX A5000"
  "NVIDIA RTX A4000"
  "NVIDIA GeForce RTX 3090 Ti"
  "NVIDIA A40"
  "NVIDIA A100 80GB PCIe"
)

# Auto-configure from .runpodkey if present
KEYFILE="$(dirname "$0")/../.runpodkey"
if [ -f "$KEYFILE" ]; then
  API_KEY=$(awk '{print $2}' "$KEYFILE")
  mkdir -p ~/.runpod
  printf "apiKey: %s\napiUrl: https://api.runpod.io/graphql\n" "$API_KEY" > ~/.runpod/.runpod.yaml
fi

echo "Creating RunPod pod: $POD_NAME"
echo "Image: $IMAGE"
echo ""

POD_ID=""
USED_GPU=""
for GPU in "${GPU_TYPES[@]}"; do
  echo -n "Trying: $GPU ... "
  OUTPUT=$(runpodctl create pod \
    --name "$POD_NAME" \
    --gpuType "$GPU" \
    --imageName "$IMAGE" \
    --containerDiskSize "$DISK_GB" \
    --volumeSize 5 \
    --volumePath "/workspace" \
    --startSSH \
    2>&1) && {
    USED_GPU="$GPU"
    POD_ID=$(echo "$OUTPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null || \
             echo "$OUTPUT" | grep -oE '"id":"[^"]*"' | head -1 | cut -d'"' -f4 || true)
    echo "OK"
    break
  } || {
    ERR=$(echo "$OUTPUT" | head -1)
    echo "unavailable — $ERR"
  }
done

if [ -z "$USED_GPU" ]; then
  echo ""
  echo "ERROR: No GPU available. Check 'runpodctl get cloud' for current availability."
  exit 1
fi

echo ""
echo "============================================================"
echo "  Pod created!"
echo "  GPU     : $USED_GPU"
[ -n "$POD_ID" ] && echo "  Pod ID  : $POD_ID"
echo ""
echo "  Wait ~60s, then get your SSH command from:"
echo "    https://www.runpod.io/console/pods"
echo "    Click pod → Connect → 'SSH over exposed TCP'"
echo ""
echo "  It will look like:"
echo "    ssh root@ssh.runpod.io -p <PORT> -i ~/.ssh/id_ed25519"
echo ""
echo "  Once inside, run setup:"
echo "    bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/runpod/setup_pod.sh)"
echo ""
echo "  STOP pod when done (to avoid charges):"
[ -n "$POD_ID" ] && echo "    runpodctl stop pod $POD_ID" || echo "    runpodctl pod list  →  then stop by ID"
echo "============================================================"
