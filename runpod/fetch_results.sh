#!/usr/bin/env bash
# ============================================================
# Fetch experiment outputs from a RunPod pod to your local machine
# Run this from your LOCAL machine.
#
# Usage:
#   bash runpod/fetch_results.sh <ssh-host> <ssh-port>
#
# Get ssh-host and ssh-port from:
#   https://www.runpod.io/console/pods → Connect → SSH
#   It looks like: ssh root@ssh.runpod.io -p 12345
#   → host = ssh.runpod.io   port = 12345
# ============================================================

set -euo pipefail

SSH_HOST="${1:?Usage: bash runpod/fetch_results.sh <ssh-host> <ssh-port>}"
SSH_PORT="${2:?Usage: bash runpod/fetch_results.sh <ssh-host> <ssh-port>}"
LOCAL_DEST="./outputs/runpod"

mkdir -p "$LOCAL_DEST"

echo "Fetching outputs from $SSH_HOST:$SSH_PORT ..."
rsync -avz --progress \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "root@$SSH_HOST:/workspace/MambaC2S/outputs/" \
    "$LOCAL_DEST/"

echo ""
echo "Done — results saved to $LOCAL_DEST"
