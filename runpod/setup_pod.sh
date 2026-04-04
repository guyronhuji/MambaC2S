#!/usr/bin/env bash
# ============================================================
# One-time pod setup: clone repo and install dependencies
# Run this INSIDE the RunPod pod.
#
# The RunPod PyTorch image already has CUDA + PyTorch installed,
# so this is much faster than GCP/Azure setup (~2 min).
#
# Usage:
#   bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/runpod/setup_pod.sh)
# ============================================================

set -euo pipefail

echo "=== [1/4] Install uv (fast package installer) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

echo ""
echo "=== [2/4] Clone repo ==="
if [ ! -d /workspace/MambaC2S ]; then
    git clone https://github.com/guyronhuji/MambaC2S.git /workspace/MambaC2S
else
    echo "Repo already present — pulling latest ..."
    git -C /workspace/MambaC2S pull
fi
cd /workspace/MambaC2S

echo ""
echo "=== [3/4] Install dependencies ==="
# PyTorch already installed in the image — skip it from requirements
uv pip install --system PyCytoData anndata umap-learn scikit-learn matplotlib seaborn pyyaml tabulate

echo ""
echo "=== [4/4] Verify GPU ==="
python3 -c "
import torch
print('CUDA available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU            :', torch.cuda.get_device_name(0))
    print('VRAM           :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
print('PyTorch        :', torch.__version__)
"

# Persist working directory in .bashrc
grep -q "cd /workspace/MambaC2S" ~/.bashrc 2>/dev/null || \
    echo "cd /workspace/MambaC2S" >> ~/.bashrc

echo ""
echo "============================================================"
echo "  Setup complete!  Run training with:"
echo "    bash runpod/train.sh"
echo "============================================================"
