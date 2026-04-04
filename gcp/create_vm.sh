#!/usr/bin/env bash
# ============================================================
# Create a GCP GPU VM for MambaC2S training
# Run this ONCE from your local machine after: gcloud init
#
# Usage:
#   chmod +x gcp/create_vm.sh
#   ./gcp/create_vm.sh
#
# Cost estimate: n1-standard-4 + T4 ≈ $0.35/hr
# (remember to stop the VM when done!)
# ============================================================

set -euo pipefail

# ── Edit these if needed ─────────────────────────────────────
PROJECT=$(gcloud config get-value project)
ZONE="us-central1-a"
VM_NAME="mambac2s-vm"
MACHINE_TYPE="n1-standard-4"    # 4 vCPUs, 15 GB RAM
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
DISK_SIZE="50GB"
# ─────────────────────────────────────────────────────────────

echo "Project : $PROJECT"
echo "Zone    : $ZONE"
echo "VM      : $VM_NAME ($MACHINE_TYPE + ${GPU_COUNT}x $GPU_TYPE)"
echo ""

# Enable required APIs (safe to run even if already enabled)
echo "Enabling Compute API ..."
gcloud services enable compute.googleapis.com --quiet

echo "Creating VM ..."
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size="$DISK_SIZE" \
  --boot-disk-type=pd-balanced \
  --metadata=install-nvidia-driver=True

echo ""
echo "============================================================"
echo "  VM created: $VM_NAME"
echo ""
echo "  SSH in:"
echo "    gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "  Once inside, run setup:"
echo "    bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/gcp/setup_vm.sh)"
echo ""
echo "  STOP VM when done (saves cost):"
echo "    gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo ""
echo "  START VM again:"
echo "    gcloud compute instances start $VM_NAME --zone=$ZONE"
echo "============================================================"
