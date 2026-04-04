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
VM_NAME="mambac2s-vm"
MACHINE_TYPE="n1-standard-4"    # 4 vCPUs, 15 GB RAM
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
DISK_SIZE="50GB"
# ─────────────────────────────────────────────────────────────

# T4 zones to try in order (europe-west4 first, then us, then asia)
ZONES=(
    europe-west4-a
    europe-west4-b
    europe-west4-c
    us-central1-a
    us-central1-b
    us-central1-c
    us-central1-f
    us-east1-c
    us-east1-d
    us-west1-b
    asia-east1-a
    asia-east1-b
)

echo "Project : $PROJECT"
echo "VM      : $VM_NAME ($MACHINE_TYPE + ${GPU_COUNT}x $GPU_TYPE)"
echo ""

gcloud services enable compute.googleapis.com --quiet

USED_ZONE=""
for ZONE in "${ZONES[@]}"; do
    echo -n "Trying zone $ZONE ... "
    if gcloud compute instances create "$VM_NAME" \
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
        --metadata=install-nvidia-driver=True \
        --quiet 2>/dev/null; then
        USED_ZONE="$ZONE"
        echo "OK"
        break
    else
        echo "no capacity, trying next zone ..."
    fi
done

if [ -z "$USED_ZONE" ]; then
    echo ""
    echo "ERROR: No T4 capacity found in any zone. Try again later or use a different GPU type."
    exit 1
fi

echo ""
echo "============================================================"
echo "  VM created: $VM_NAME  (zone: $USED_ZONE)"
echo ""
echo "  SSH in:"
echo "    gcloud compute ssh $VM_NAME --zone=$USED_ZONE"
echo ""
echo "  Once inside, run setup:"
echo "    bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/gcp/setup_vm.sh)"
echo ""
echo "  STOP VM when done (saves cost):"
echo "    gcloud compute instances stop $VM_NAME --zone=$USED_ZONE"
echo ""
echo "  START VM again:"
echo "    gcloud compute instances start $VM_NAME --zone=$USED_ZONE"
echo ""
echo "  Fetch results:"
echo "    bash gcp/fetch_results.sh $USED_ZONE"
echo "============================================================"
