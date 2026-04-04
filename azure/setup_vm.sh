#!/usr/bin/env bash
# ============================================================
# One-time VM setup: install CUDA drivers, Python deps, clone repo
# Run this INSIDE the VM after SSH'ing in.
#
# Usage:
#   bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/azure/setup_vm.sh)
# ============================================================

set -euo pipefail

echo "=== [1/5] System packages ==="
sudo apt-get update -q
sudo apt-get install -y -q git python3-pip python3-venv nvtop htop

echo ""
echo "=== [2/5] NVIDIA drivers + CUDA ==="
# Install the Ubuntu driver utilities and let them fetch the right driver version
sudo apt-get install -y -q ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu 2>/dev/null || true

# Install CUDA toolkit (12.x)
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -q
sudo apt-get install -y -q cuda-toolkit-12-4
rm cuda-keyring_1.1-1_all.deb

echo ""
echo "=== [3/5] Python virtual environment ==="
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip -q

echo ""
echo "=== [4/5] Clone repo and install dependencies ==="
if [ ! -d ~/MambaC2S ]; then
    git clone https://github.com/guyronhuji/MambaC2S.git ~/MambaC2S
else
    echo "Repo already present — pulling latest ..."
    git -C ~/MambaC2S pull
fi

cd ~/MambaC2S
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q -r requirements.txt

echo ""
echo "=== [5/5] Verify GPU ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  To activate the environment next time you SSH in:"
echo "    source ~/venv/bin/activate && cd ~/MambaC2S"
echo ""
echo "  Then run training:"
echo "    bash azure/train.sh"
echo "============================================================"

# Persist venv activation in .bashrc
grep -q "source ~/venv/bin/activate" ~/.bashrc || echo "source ~/venv/bin/activate && cd ~/MambaC2S" >> ~/.bashrc
