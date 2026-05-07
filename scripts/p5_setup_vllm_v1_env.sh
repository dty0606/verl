#!/usr/bin/env bash
# Build an isolated latest-VERL + vLLM V1 environment for Tau3 smoke tests.
#
# Prefer the official project image once available. This helper is for P5 bring-up
# when we need a reproducible local conda env and want to avoid torch/vLLM ABI
# drift from ad-hoc pip installs.

set -euo pipefail

ENV_NAME="${ENV_NAME:-sdpo-vllm20-v1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.20.1}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
VLLM_EXTRA_INDEX_URL="${VLLM_EXTRA_INDEX_URL:-https://wheels.vllm.ai/${VLLM_VERSION}/${TORCH_BACKEND}}"
INSTALL_CUDA_TOOLS="${INSTALL_CUDA_TOOLS:-1}"
CUDA_TOOLS_CHANNEL="${CUDA_TOOLS_CHANNEL:-nvidia/label/cuda-12.9.1}"
CUDA_NVCC_TOOLS_VERSION="${CUDA_NVCC_TOOLS_VERSION:-12.9.86}"
CUDA_NVVM_TOOLS_VERSION="${CUDA_NVVM_TOOLS_VERSION:-12.9.86}"
CUDA_CUDART_DEV_VERSION="${CUDA_CUDART_DEV_VERSION:-12.9.79}"
CUDA_CRT_VERSION="${CUDA_CRT_VERSION:-12.9.86}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
TAU2_REPO="${TAU2_REPO:-https://github.com/sierra-research/tau2-bench.git}"
TAU2_COMMIT="${TAU2_COMMIT:-220b47844fb74d4351037e81055cf1e2948e4734}"
TAU2_DIR="${TAU2_DIR:-$HOME/tau2-bench}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "----------------------------------------------------------------"
echo "Setting up latest-VERL Tau3 vLLM V1 env"
echo "Project root: $PROJECT_ROOT"
echo "Conda env: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo "vLLM: $VLLM_VERSION"
echo "torch backend: $TORCH_BACKEND"
echo "vLLM extra index: $VLLM_EXTRA_INDEX_URL"
echo "install CUDA tools: $INSTALL_CUDA_TOOLS"
echo "install flash-attn: $INSTALL_FLASH_ATTN"
echo "tau2-bench: $TAU2_DIR @ $TAU2_COMMIT"
echo "----------------------------------------------------------------"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is required on the P5 host." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip uv

if uv pip install --help | grep -q -- "--torch-backend"; then
    uv pip install "vllm==$VLLM_VERSION" --torch-backend "$TORCH_BACKEND"
else
    echo "WARN: uv does not expose --torch-backend; using vLLM wheel index: $VLLM_EXTRA_INDEX_URL"
    python -m pip install "vllm==$VLLM_VERSION" --extra-index-url "$VLLM_EXTRA_INDEX_URL"
fi

if [ "$INSTALL_CUDA_TOOLS" = "1" ]; then
    conda install -n "$ENV_NAME" -y \
        -c "$CUDA_TOOLS_CHANNEL" \
        "cuda-nvcc-tools=$CUDA_NVCC_TOOLS_VERSION" \
        "cuda-nvvm-tools=$CUDA_NVVM_TOOLS_VERSION" \
        "cuda-cudart-dev=$CUDA_CUDART_DEV_VERSION" \
        "cuda-crt=$CUDA_CRT_VERSION" \
        --no-update-deps
fi

cd "$PROJECT_ROOT"

export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX/targets/x86_64-linux}"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
mkdir -p "$HOME/tmp"
export TMPDIR="${TMPDIR:-$HOME/tmp}"

if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
    python -m pip install --no-deps "$FLASH_ATTN_WHEEL_URL"
fi

# Install VERL/Tau3 runtime dependencies without letting the local editable
# package resolver drift the torch/vLLM ABI pair that vLLM just installed.
python -m pip install \
    accelerate \
    addict \
    boto3 \
    codetiming \
    datasets \
    deepdiff \
    dill \
    fastuuid \
    "gymnasium>=1.2.2" \
    hydra-core \
    latex2sympy2_extended \
    "litellm>=1.80.15,<1.82.7" \
    mathruler \
    multiprocess \
    "numpy<2.0.0" \
    pandas \
    peft \
    "pyarrow>=19.0.0" \
    pybind11 \
    pylatexenc \
    qwen-vl-utils \
    "ray[default]>=2.41.0" \
    tenacity \
    tensorboard \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    toml \
    torchdata \
    wandb \
    xxhash

python -m pip install -e . --no-deps

if [ -d "$TAU2_DIR/.git" ] && command -v git >/dev/null 2>&1; then
    git -C "$TAU2_DIR" fetch --quiet origin "$TAU2_COMMIT" || true
    git -C "$TAU2_DIR" checkout --detach "$TAU2_COMMIT"
elif [ -f "$TAU2_DIR/pyproject.toml" ] || [ -f "$TAU2_DIR/setup.py" ]; then
    echo "Using existing tau2-bench directory without git metadata: $TAU2_DIR"
elif [ ! -e "$TAU2_DIR" ] && command -v git >/dev/null 2>&1; then
    git clone "$TAU2_REPO" "$TAU2_DIR"
    git -C "$TAU2_DIR" checkout --detach "$TAU2_COMMIT"
else
    echo "Error: tau2-bench is required but $TAU2_DIR is not an installable snapshot." >&2
    echo "Sync a tau2-bench snapshot to $TAU2_DIR from S3, or set TAU2_DIR to an existing copy." >&2
    exit 1
fi
python -m pip install -e "$TAU2_DIR" --no-deps
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$TAU2_DIR/data}"

python - <<'PY'
import os
from pathlib import Path

cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
header = cuda_home / "include" / "cuda_runtime.h"
print(f"CUDA_HOME={cuda_home}")
print(f"CUDA_RUNTIME_HEADER={header if header.exists() else '<missing>'}")
if not header.exists():
    print("WARN: cuda_runtime.h is missing; FlashInfer/GDN JIT may fail until CUDA_HOME points at full CUDA toolkit.")
PY

echo "Setup complete. Before launching training, run:"
echo "  conda activate $ENV_NAME"
echo "  export VLLM_USE_V1=1"
echo "  export CUDA_HOME=$CUDA_HOME"
echo "  export PATH=\"$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$CUDA_HOME/bin:\$PATH\""
echo "  export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}\""
echo "  export LIBRARY_PATH=\"/usr/lib/x86_64-linux-gnu:\${LIBRARY_PATH:-}\""
echo "  export TMPDIR=${TMPDIR:-$HOME/tmp}"
echo "  export TAU2_DATA_DIR=${TAU2_DATA_DIR:-$TAU2_DIR/data}"
echo "  python scripts/p5_preflight_vllm_v1.py --model-path <model> --dataset-dir <dataset>"
