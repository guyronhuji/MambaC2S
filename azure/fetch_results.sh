#!/usr/bin/env bash
# ============================================================
# Fetch experiment outputs from the Azure VM to your local machine
# Run this from your LOCAL machine:
#
#   bash azure/fetch_results.sh <vm-public-ip>
# ============================================================

set -euo pipefail

VM_IP="${1:?Usage: bash azure/fetch_results.sh <vm-public-ip>}"
ADMIN_USER="azureuser"
LOCAL_DEST="./outputs/azure"

mkdir -p "$LOCAL_DEST"

echo "Fetching outputs from $VM_IP ..."
rsync -avz --progress \
    "$ADMIN_USER@$VM_IP:~/MambaC2S/outputs/" \
    "$LOCAL_DEST/"

echo ""
echo "Done — results saved to $LOCAL_DEST"
