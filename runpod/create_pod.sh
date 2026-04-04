#!/usr/bin/env bash
# ============================================================
# Create a RunPod GPU pod for MambaC2S training
#
# Prerequisites (one-time):
#   1. Install CLI:  brew install runpod/runpodctl/runpodctl
#   2. Get API key:  https://www.runpod.io/console/user/settings  (API Keys tab)
#   3. Configure:    runpodctl config --apiKey <YOUR_KEY>
#   4. Add SSH key:  https://www.runpod.io/console/user/settings  (SSH Public Keys tab)
#      Paste the output of: cat ~/.ssh/id_ed25519.pub
#      (generate one first if needed: ssh-keygen -t ed25519)
#
# Usage:
#   chmod +x runpod/create_pod.sh
#   ./runpod/create_pod.sh
#
# Cost estimates (on-demand):
#   RTX 3090  ≈ $0.22/hr    RTX 4090  ≈ $0.44/hr
#   A40       ≈ $0.40/hr    A100 80GB ≈ $1.89/hr
# ============================================================

set -euo pipefail

# ── Edit if needed ───────────────────────────────────────────
POD_NAME="mambac2s"
# GPU preference order — script picks the first type with availability
GPU_TYPES=(
  "NVIDIA GeForce RTX 4090"
  "NVIDIA GeForce RTX 3090"
  "NVIDIA A40"
  "NVIDIA RTX A5000"
  "NVIDIA A100 80GB PCIe"
)
IMAGE="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
DISK_GB=50
# ─────────────────────────────────────────────────────────────

# Verify CLI is configured
if ! runpodctl config list 2>/dev/null | grep -q "apiKey"; then
  echo "ERROR: runpodctl not configured. Run:"
  echo "  runpodctl config --apiKey <YOUR_KEY>"
  exit 1
fi

echo "Creating RunPod pod: $POD_NAME"
echo "Image: $IMAGE"
echo ""

POD_ID=""
USED_GPU=""
for GPU in "${GPU_TYPES[@]}"; do
  echo -n "Trying GPU: $GPU ... "
  RESULT=$(runpodctl create pod \
    --name "$POD_NAME" \
    --gpuType "$GPU" \
    --imageName "$IMAGE" \
    --containerDiskInGb "$DISK_GB" \
    --ports "22/tcp" \
    2>&1) && {
    POD_ID=$(echo "$RESULT" | grep -oE 'pod [a-z0-9]+' | awk '{print $2}' || true)
    # Fallback: grab any hex/alphanum id-like string
    if [ -z "$POD_ID" ]; then
      POD_ID=$(echo "$RESULT" | grep -oE '[a-z0-9]{8,}' | head -1 || true)
    fi
    USED_GPU="$GPU"
    echo "OK"
    break
  } || {
    MSG=$(echo "$RESULT" | head -1)
    echo "unavailable ($MSG)"
  }
done

if [ -z "$USED_GPU" ]; then
  echo ""
  echo "ERROR: No GPU type available. Try again later or add more GPU types to the list."
  exit 1
fi

echo ""
echo "============================================================"
echo "  Pod created!"
echo "  GPU     : $USED_GPU"
if [ -n "$POD_ID" ]; then
  echo "  Pod ID  : $POD_ID"
fi
echo ""
echo "  Wait ~60s for the pod to start, then:"
echo ""
echo "  1. Get the SSH command from:"
echo "     https://www.runpod.io/console/pods"
echo "     Click your pod → Connect → SSH"
echo ""
echo "  2. SSH in and run setup:"
echo "     bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/runpod/setup_pod.sh)"
echo ""
echo "  3. Run training:"
echo "     bash runpod/train.sh"
echo ""
echo "  STOP pod when done (from the RunPod console or):"
if [ -n "$POD_ID" ]; then
  echo "     runpodctl stop pod $POD_ID"
fi
echo "============================================================"
