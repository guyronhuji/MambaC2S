#!/usr/bin/env bash
# ============================================================
# Create an Azure GPU VM for MambaC2S training (Option A)
# Run this ONCE from your local machine after: az login
#
# Usage:
#   chmod +x azure/create_vm.sh
#   ./azure/create_vm.sh
#
# Cost estimate: NC4as T4 v3 ≈ $0.50/hr  (remember to deallocate when done!)
# ============================================================

set -euo pipefail

# ── Edit these if needed ─────────────────────────────────────
RESOURCE_GROUP="mambaC2S-rg"
LOCATION="eastus"
VM_NAME="mambaC2S-vm"
VM_SIZE="Standard_NC4as_T4_v3"   # 1x T4 GPU, 4 vCPUs, 28 GB RAM
ADMIN_USER="azureuser"
# ─────────────────────────────────────────────────────────────

echo "Creating resource group: $RESOURCE_GROUP in $LOCATION ..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo "Creating VM: $VM_NAME ($VM_SIZE) ..."
az vm create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --size "$VM_SIZE" \
  --image "Canonical:ubuntu-24_04-lts:server:latest" \
  --admin-username "$ADMIN_USER" \
  --generate-ssh-keys \
  --output table

echo ""
echo "Opening SSH port ..."
az vm open-port --resource-group "$RESOURCE_GROUP" --name "$VM_NAME" --port 22 --output none

echo ""
echo "VM created. Getting public IP ..."
PUBLIC_IP=$(az vm show \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --show-details \
  --query publicIps \
  --output tsv)

echo ""
echo "============================================================"
echo "  VM ready: $PUBLIC_IP"
echo ""
echo "  1. SSH in:"
echo "     ssh $ADMIN_USER@$PUBLIC_IP"
echo ""
echo "  2. Once inside, run the VM setup script:"
echo "     bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/azure/setup_vm.sh)"
echo ""
echo "  STOP VM when not training (saves cost):"
echo "     az vm deallocate -g $RESOURCE_GROUP -n $VM_NAME"
echo ""
echo "  START VM again:"
echo "     az vm start -g $RESOURCE_GROUP -n $VM_NAME"
echo "     az vm show -g $RESOURCE_GROUP -n $VM_NAME --show-details --query publicIps -o tsv"
echo "============================================================"
