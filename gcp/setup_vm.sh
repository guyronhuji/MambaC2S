#!/usr/bin/env bash
# ============================================================
# One-time VM setup: CUDA drivers, Python deps, clone repo
# Run this INSIDE the VM after SSH'ing in.
#
# Usage:
#   bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/gcp/setup_vm.sh)
# ============================================================

set -euo pipefail

echo "=== [1/5] System packages ==="
sudo apt-get update -q
sudo apt-get install -y -q git python3-pip python3-venv nvtop htop

echo ""
echo "=== [2/5] NVIDIA drivers + CUDA ==="
# Install CUDA 12.4 keyring
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -q
sudo apt-get install -y -q cuda-toolkit-12-4 nvidia-driver-550
rm cuda-keyring_1.1-1_all.deb

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

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
python3 -c "
import torch
print('CUDA available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU            :', torch.cuda.get_device_name(0))
    print('VRAM           :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

# Persist venv + path in .bashrc
grep -q "source ~/venv/bin/activate" ~/.bashrc || \
    echo "source ~/venv/bin/activate && cd ~/MambaC2S" >> ~/.bashrc

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Each time you SSH in, your venv activates automatically."
echo "  Run training with:  bash gcp/train.sh"
echo "============================================================"
