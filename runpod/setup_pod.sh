#!/usr/bin/env bash
# ============================================================
# One-time pod setup: clone repo and install dependencies
# Run this INSIDE the RunPod pod.
#
# Usage:
#   bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/runpod/setup_pod.sh)
# ============================================================

set -euo pipefail

echo "=== [1/4] Installing uv ==="
curl -Lf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
echo "uv installed: $(uv --version)"

echo ""
echo "=== [2/4] Cloning repo ==="
if [ ! -d /workspace/MambaC2S ]; then
    git clone https://github.com/guyronhuji/MambaC2S.git /workspace/MambaC2S
else
    echo "Already present — pulling latest ..."
    git -C /workspace/MambaC2S pull
fi
cd /workspace/MambaC2S
echo "Repo ready at $(pwd)"

echo ""
echo "=== [3/4] Installing dependencies ==="
uv pip install --system --verbose \
    PyCytoData anndata umap-learn scikit-learn \
    matplotlib seaborn pyyaml tabulate rich

echo ""
echo "=== [4/4] Verifying GPU ==="
python3 -c "
import torch
print('CUDA available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU            :', torch.cuda.get_device_name(0))
    print('VRAM           :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
print('PyTorch        :', torch.__version__)
"

grep -q "cd /workspace/MambaC2S" ~/.bashrc 2>/dev/null || \
    echo "cd /workspace/MambaC2S" >> ~/.bashrc

echo ""
echo "============================================================"
echo "  Setup complete!  Run training with:"
echo "    bash runpod/train.sh"
echo "============================================================"
