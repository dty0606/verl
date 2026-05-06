#!/usr/bin/env bash
# Build an isolated latest-VERL + vLLM V1 environment for Tau3 smoke tests.
#
# Prefer the official project image once available. This helper is for P5 bring-up
# when we need a reproducible local conda env and want to avoid torch/vLLM ABI
# drift from ad-hoc pip installs.

set -euo pipefail

ENV_NAME="${ENV_NAME:-sdpo-vllm20-v1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.20.0}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "----------------------------------------------------------------"
echo "Setting up latest-VERL Tau3 vLLM V1 env"
echo "Project root: $PROJECT_ROOT"
echo "Conda env: $ENV_NAME"
echo "Python: $PYTHON_VERSION"
echo "vLLM: $VLLM_VERSION"
echo "torch backend: $TORCH_BACKEND"
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
    echo "WARN: uv does not expose --torch-backend; falling back to plain pip install."
    python -m pip install "vllm==$VLLM_VERSION"
fi

cd "$PROJECT_ROOT"
python -m pip install -e .

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
echo "  python scripts/p5_preflight_vllm_v1.py --model-path <model> --dataset-dir <dataset>"
