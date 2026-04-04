#!/usr/bin/env bash
# ============================================================
# Fetch experiment outputs from a RunPod pod to your local machine
# Run this from your LOCAL machine.
#
# Usage:
#   bash runpod/fetch_results.sh <ssh-user@host> [port]
#
# Examples:
#   bash runpod/fetch_results.sh yk23p2p92l9t8c-64411c9e@ssh.runpod.io
#   bash runpod/fetch_results.sh root@ssh.runpod.io 12345
# ============================================================

set -euo pipefail

TARGET="${1:?Usage: bash runpod/fetch_results.sh <user@host> [port]}"
PORT="${2:-}"
LOCAL_DEST="./outputs/runpod"

mkdir -p "$LOCAL_DEST"

if [ -n "$PORT" ]; then
  SSH_OPT="-e ssh -p $PORT -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"
else
  SSH_OPT="-e ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"
fi

echo "Fetching from $TARGET ..."
rsync -avz --progress $SSH_OPT \
    "$TARGET:/workspace/MambaC2S/outputs/" \
    "$LOCAL_DEST/"

echo ""
echo "Done — results saved to $LOCAL_DEST"
