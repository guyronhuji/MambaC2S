#!/usr/bin/env bash
# ============================================================
# Fetch experiment outputs from the GCP VM to your local machine
# Run this from your LOCAL machine:
#
#   bash gcp/fetch_results.sh [zone]
#
# zone defaults to us-central1-a
# ============================================================

set -euo pipefail

VM_NAME="mambac2s-vm"
ZONE="${1:-us-central1-a}"
LOCAL_DEST="./outputs/gcp"

mkdir -p "$LOCAL_DEST"

echo "Fetching outputs from $VM_NAME ($ZONE) ..."
gcloud compute scp --recurse \
    "$VM_NAME:~/MambaC2S/outputs/" \
    "$LOCAL_DEST/" \
    --zone="$ZONE"

echo ""
echo "Done — results saved to $LOCAL_DEST"
