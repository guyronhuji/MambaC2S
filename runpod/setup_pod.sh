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
echo "=== [3b/4] Installing mamba-ssm (CUDA kernels) ==="
# mamba-ssm compiles C++/CUDA extensions at install time — must use pip, not uv.
# PyTorch and the system CUDA toolkit must match versions; reinstall if they don't.
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    SYS_CUDA=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
    TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "")
    echo "System CUDA: ${SYS_CUDA}   PyTorch CUDA: ${TORCH_CUDA}"

    if [ -n "$SYS_CUDA" ] && [ "$SYS_CUDA" != "$TORCH_CUDA" ]; then
        echo "Version mismatch — reinstalling PyTorch for CUDA ${SYS_CUDA} ..."
        # Convert e.g. "12.1" -> "cu121"
        CU_TAG="cu$(echo "$SYS_CUDA" | tr -d '.')"
        pip install --upgrade torch torchvision torchaudio \
            --index-url "https://download.pytorch.org/whl/${CU_TAG}"
    fi

    pip install causal-conv1d mamba-ssm
else
    echo "No CUDA — skipping mamba-ssm (pure-PyTorch fallback will be used)."
fi

echo ""
echo "=== [4/4] Verifying GPU ==="
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
