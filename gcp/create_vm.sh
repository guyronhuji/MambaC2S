#!/usr/bin/env bash
# ============================================================
# Create a GCP GPU VM for MambaC2S training
# Queries live quota to find the best available GPU and zone.
#
# Usage:
#   chmod +x gcp/create_vm.sh
#   ./gcp/create_vm.sh
#
# Cost estimates (on-demand, approximate):
#   T4  (n1-standard-4)  ≈ $0.35/hr
#   L4  (g2-standard-4)  ≈ $0.70/hr
#   A100(a2-highgpu-1g)  ≈ $3.67/hr
# ============================================================

set -euo pipefail

PROJECT=$(gcloud config get-value project)
VM_NAME="mambac2s-vm"
DISK_SIZE="50GB"

# GPU priority: best first
declare -A GPU_MACHINE
GPU_MACHINE["NVIDIA_A100_80GB_GPUS"]="a2-ultragpu-1g"
GPU_MACHINE["NVIDIA_A100_GPUS"]="a2-highgpu-1g"
GPU_MACHINE["NVIDIA_L4_GPUS"]="g2-standard-4"
GPU_MACHINE["NVIDIA_T4_GPUS"]="n1-standard-4"

declare -A GPU_ACCEL
GPU_ACCEL["NVIDIA_A100_80GB_GPUS"]="nvidia-a100-80gb"
GPU_ACCEL["NVIDIA_A100_GPUS"]="nvidia-tesla-a100"
GPU_ACCEL["NVIDIA_L4_GPUS"]="nvidia-l4"
GPU_ACCEL["NVIDIA_T4_GPUS"]="nvidia-tesla-t4"

GPU_PRIORITY=(NVIDIA_A100_80GB_GPUS NVIDIA_A100_GPUS NVIDIA_L4_GPUS NVIDIA_T4_GPUS)

echo "Project : $PROJECT"
echo ""
echo "Querying GPU quota across all regions ..."
echo "(this takes ~30 seconds)"
echo ""

# Build quota table: region  GPU_TYPE  limit  usage  available
QUOTA_TABLE=$(
  for r in $(gcloud compute regions list --format="value(name)"); do
    gcloud compute regions describe "$r" \
      --flatten="quotas[]" \
      --format="csv[no-heading](name,quotas.metric,quotas.limit,quotas.usage)" 2>/dev/null
  done | awk -F, '
    $2=="NVIDIA_T4_GPUS" || $2=="NVIDIA_L4_GPUS" || $2=="NVIDIA_A100_GPUS" || $2=="NVIDIA_A100_80GB_GPUS" {
      avail = ($3+0) - ($4+0)
      if (avail > 0)
        printf "%s %s %.0f %.0f %.0f\n", $1, $2, $3+0, $4+0, avail
    }
  '
)

if [ -z "$QUOTA_TABLE" ]; then
  echo "ERROR: No GPU quota available in any region. Request a quota increase at:"
  echo "  https://console.cloud.google.com/iam-admin/quotas"
  exit 1
fi

echo "Regions with available GPU quota:"
printf "%-20s %-26s %8s %8s %8s\n" "REGION" "GPU" "LIMIT" "USED" "FREE"
printf "%-20s %-26s %8s %8s %8s\n" "------" "---" "-----" "----" "----"
echo "$QUOTA_TABLE" | while read -r region gpu limit used avail; do
  printf "%-20s %-26s %8.0f %8.0f %8.0f\n" "$region" "$gpu" "$limit" "$used" "$avail"
done
echo ""

# Pick the best GPU type available (highest priority first)
CHOSEN_GPU=""
CHOSEN_REGION=""
for gpu in "${GPU_PRIORITY[@]}"; do
  match=$(echo "$QUOTA_TABLE" | awk -v g="$gpu" '$2==g {print $1; exit}')
  if [ -n "$match" ]; then
    CHOSEN_GPU="$gpu"
    CHOSEN_REGION="$match"
    break
  fi
done

if [ -z "$CHOSEN_GPU" ]; then
  echo "ERROR: Could not match any GPU type."
  exit 1
fi

MACHINE_TYPE="${GPU_MACHINE[$CHOSEN_GPU]}"
ACCEL="${GPU_ACCEL[$CHOSEN_GPU]}"

echo "Selected: $CHOSEN_GPU in $CHOSEN_REGION → $MACHINE_TYPE + $ACCEL"
echo ""

# Try all zones in the chosen region
USED_ZONE=""
for ZONE in "${CHOSEN_REGION}-a" "${CHOSEN_REGION}-b" "${CHOSEN_REGION}-c" "${CHOSEN_REGION}-d"; do
  echo -n "Trying $ZONE ... "
  if gcloud compute instances create "$VM_NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --machine-type="$MACHINE_TYPE" \
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
    echo "OK"
    break
  else
    echo "no capacity"
  fi
done

if [ -z "$USED_ZONE" ]; then
  echo "ERROR: Quota available but no zone had capacity. Try running again shortly."
  exit 1
fi

echo ""
echo "============================================================"
echo "  VM ready: $VM_NAME"
echo "  Zone    : $USED_ZONE"
echo "  GPU     : $ACCEL ($MACHINE_TYPE)"
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
