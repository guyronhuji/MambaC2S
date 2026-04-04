#!/usr/bin/env bash
# ============================================================
# Create a GCP GPU VM for MambaC2S training
# Tries zones in order until one has capacity.
# Compatible with bash 3.2 (macOS default).
#
# Usage:
#   chmod +x gcp/create_vm.sh
#   ./gcp/create_vm.sh
#
# Cost estimates (on-demand, approximate):
#   T4  (n1-standard-4)  ≈ $0.35/hr
#   L4  (g2-standard-4)  ≈ $0.70/hr
# ============================================================

set -euo pipefail

PROJECT=$(gcloud config get-value project)
VM_NAME="mambac2s-vm"
DISK_SIZE="50GB"

# ── Preferred zones (edit to reorder) ────────────────────────
# Format: "zone:machine-type:accelerator"
CANDIDATES=(
  "europe-west4-a:n1-standard-4:nvidia-tesla-t4"
  "europe-west4-b:n1-standard-4:nvidia-tesla-t4"
  "europe-west4-c:n1-standard-4:nvidia-tesla-t4"
  "europe-west4-a:g2-standard-4:nvidia-l4"
  "europe-west4-b:g2-standard-4:nvidia-l4"
  "us-central1-a:n1-standard-4:nvidia-tesla-t4"
  "us-central1-b:n1-standard-4:nvidia-tesla-t4"
  "us-central1-c:n1-standard-4:nvidia-tesla-t4"
  "us-central1-f:n1-standard-4:nvidia-tesla-t4"
  "us-east1-c:n1-standard-4:nvidia-tesla-t4"
  "us-east1-d:n1-standard-4:nvidia-tesla-t4"
  "us-west1-b:n1-standard-4:nvidia-tesla-t4"
)
# ─────────────────────────────────────────────────────────────

echo "Project : $PROJECT"
echo ""

USED_ZONE=""
USED_MACHINE=""
USED_ACCEL=""

for candidate in "${CANDIDATES[@]}"; do
  ZONE=$(echo "$candidate"    | cut -d: -f1)
  MACHINE=$(echo "$candidate" | cut -d: -f2)
  ACCEL=$(echo "$candidate"   | cut -d: -f3)

  echo -n "Trying $ZONE  ($ACCEL) ... "
  if gcloud compute instances create "$VM_NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --machine-type="$MACHINE" \
      --accelerator="type=$ACCEL,count=1" \
      --maintenance-policy=TERMINATE \
      --restart-on-failure \
      --image-family=ubuntu-2204-lts \
      --image-project=ubuntu-os-cloud \
      --boot-disk-size="$DISK_SIZE" \
      --boot-disk-type=pd-balanced \
      --metadata=install-nvidia-driver=True \
      --quiet 2>/dev/null; then
    USED_ZONE="$ZONE"
    USED_MACHINE="$MACHINE"
    USED_ACCEL="$ACCEL"
    echo "OK"
    break
  else
    echo "no capacity"
  fi
done

if [ -z "$USED_ZONE" ]; then
  echo ""
  echo "ERROR: No capacity found in any candidate zone."
  echo "Try again later, or add more zones to CANDIDATES in gcp/create_vm.sh"
  exit 1
fi

echo ""
echo "============================================================"
echo "  VM ready: $VM_NAME"
echo "  Zone    : $USED_ZONE"
echo "  GPU     : $USED_ACCEL ($USED_MACHINE)"
echo ""
echo "  SSH in:"
echo "    gcloud compute ssh $VM_NAME --zone=$USED_ZONE"
echo ""
echo "  Once inside, run setup:"
echo "    bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/gcp/setup_vm.sh)"
echo ""
echo "  STOP VM when done:"
echo "    gcloud compute instances stop $VM_NAME --zone=$USED_ZONE"
echo ""
echo "  START VM again:"
echo "    gcloud compute instances start $VM_NAME --zone=$USED_ZONE"
echo ""
echo "  Fetch results:"
echo "    bash gcp/fetch_results.sh $USED_ZONE"
echo "============================================================"
