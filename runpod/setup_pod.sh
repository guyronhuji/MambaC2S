#!/usr/bin/env bash
# ============================================================
# One-time pod setup: clone repo and install dependencies
# Run this INSIDE the RunPod pod.
#
# Usage:
#   bash <(curl -s https://raw.githubusercontent.com/guyronhuji/MambaC2S/main/runpod/setup_pod.sh)
# ============================================================

set -euo pipefail

echo "=== [1/4] Cloning repo ==="
if [ ! -d /workspace/MambaC2S ]; then
    git clone https://github.com/guyronhuji/MambaC2S.git /workspace/MambaC2S
else
    echo "Already present — pulling latest ..."
    git -C /workspace/MambaC2S pull
fi
cd /workspace/MambaC2S
echo "Repo ready at $(pwd)"

echo ""
echo "=== [2/4] Installing dependencies ==="
pip install \
    PyCytoData anndata umap-learn scikit-learn \
    matplotlib seaborn pyyaml tabulate rich

echo ""
echo "=== [3/4] Installing mamba-ssm (CUDA kernels) ==="
# mamba-ssm compiles C++/CUDA extensions — must use pip (not uv) and PyTorch
# must match the system CUDA toolkit. Force-reinstall PyTorch for cu121 first.
if nvcc --version &>/dev/null; then
    SYS_CUDA=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
    CU_TAG="cu$(echo "$SYS_CUDA" | tr -d '.')"
    echo "System CUDA: ${SYS_CUDA} → installing PyTorch for ${CU_TAG}"
    pip install --force-reinstall torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${CU_TAG}"
    echo "PyTorch CUDA after reinstall: $(python3 -c 'import torch; print(torch.version.cuda)')"
    pip install causal-conv1d mamba-ssm
else
    echo "nvcc not found — skipping mamba-ssm (pure-PyTorch fallback will be used)."
fi

echo ""
echo "=== [4/4] Verifying installation ==="
python3 -c "
import torch
print('CUDA available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU            :', torch.cuda.get_device_name(0))
    print('VRAM           :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
print('PyTorch        :', torch.__version__)
try:
    import mamba_ssm; print('mamba-ssm      : installed ✓')
except ImportError:
    print('mamba-ssm      : NOT installed (pure-PyTorch fallback)')
"

grep -q "cd /workspace/MambaC2S" ~/.bashrc 2>/dev/null || \
    echo "cd /workspace/MambaC2S" >> ~/.bashrc

echo ""
echo "============================================================"
echo "  Setup complete!  Run training with:"
echo "    bash runpod/train.sh"
echo "============================================================"
