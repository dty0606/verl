#!/usr/bin/env bash
# Export a frozen runtime manifest from the currently active environment.
#
# This helper intentionally avoids Docker and does not require Git. If Git or
# conda are available, it records their metadata; otherwise it writes a clear
# placeholder and continues.

set -eo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/p5_export_frozen_env.sh [output_dir]

Environment:
  SOURCE_REVISION   Optional source revision to record when Git is unavailable.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-$PROJECT_ROOT/outputs/frozen_env/$(date -u +%Y%m%dT%H%M%SZ)}"

mkdir -p "$OUT_DIR"

MANIFEST="$OUT_DIR/manifest.txt"
PYTHON_INFO="$OUT_DIR/python_info.json"
PIP_FREEZE="$OUT_DIR/pip_freeze.txt"
CONDA_LIST="$OUT_DIR/conda_list.txt"
RUNTIME_VERSIONS="$OUT_DIR/runtime_versions.json"
VLLM_C_STATUS="$OUT_DIR/vllm_c_status.txt"
CUDA_HEADERS="$OUT_DIR/cuda_runtime_headers.txt"
NVIDIA_SMI="$OUT_DIR/nvidia_smi.txt"
SOURCE_REV="$OUT_DIR/source_revision.txt"

write_section() {
    printf '\n[%s]\n' "$1" >> "$MANIFEST"
}

run_or_note() {
    local output_file="$1"
    local status
    shift
    if "$@" > "$output_file" 2>&1; then
        return 0
    fi
    status=$?
    {
        echo "COMMAND_FAILED: $*"
        echo "EXIT_CODE: $status"
    } >> "$output_file"
    return 0
}

: > "$MANIFEST"
write_section "export"
{
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "output_dir=$OUT_DIR"
    echo "project_root=$PROJECT_ROOT"
    echo "hostname=$(hostname 2>/dev/null || echo '<unknown>')"
    echo "user=${USER:-<unset>}"
    echo "conda_prefix=${CONDA_PREFIX:-<unset>}"
    echo "conda_default_env=${CONDA_DEFAULT_ENV:-<unset>}"
    echo "python=$(command -v python 2>/dev/null || echo '<missing>')"
    echo "pip=$(command -v pip 2>/dev/null || echo '<missing>')"
} >> "$MANIFEST"

if command -v python >/dev/null 2>&1; then
    python - <<'PY' > "$PYTHON_INFO"
import json
import os
import platform
import sys

payload = {
    "executable": sys.executable,
    "version": sys.version,
    "version_info": list(sys.version_info),
    "platform": platform.platform(),
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "conda_prefix": os.environ.get("CONDA_PREFIX"),
    "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
else
    echo "python not found" > "$PYTHON_INFO"
fi
write_section "python"
cat "$PYTHON_INFO" >> "$MANIFEST"

if command -v python >/dev/null 2>&1; then
    run_or_note "$PIP_FREEZE" python -m pip freeze
elif command -v pip >/dev/null 2>&1; then
    run_or_note "$PIP_FREEZE" pip freeze
else
    echo "pip not found" > "$PIP_FREEZE"
fi
write_section "pip_freeze"
echo "$PIP_FREEZE" >> "$MANIFEST"

if command -v conda >/dev/null 2>&1; then
    run_or_note "$CONDA_LIST" conda list
else
    echo "conda not found" > "$CONDA_LIST"
fi
write_section "conda_list"
echo "$CONDA_LIST" >> "$MANIFEST"

if command -v python >/dev/null 2>&1; then
    python - <<'PY' > "$RUNTIME_VERSIONS"
import importlib
import importlib.metadata as metadata
import json

packages = [
    ("torch", "torch"),
    ("vllm", "vllm"),
    ("transformers", "transformers"),
    ("tokenizers", "tokenizers"),
    ("flashinfer", "flashinfer-python"),
    ("ray", "ray"),
]

payload = {}
for import_name, dist_name in packages:
    record = {}
    try:
        module = importlib.import_module(import_name)
        record["importable"] = True
        record["module_version"] = getattr(module, "__version__", None)
        record["module_file"] = getattr(module, "__file__", None)
    except Exception as exc:
        record["importable"] = False
        record["import_error"] = repr(exc)
    try:
        record["dist_version"] = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        record["dist_version"] = None
    payload[import_name] = record

try:
    import torch

    payload["torch"]["cuda_version"] = getattr(torch.version, "cuda", None)
    payload["torch"]["cuda_available"] = bool(torch.cuda.is_available())
    payload["torch"]["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    pass

print(json.dumps(payload, indent=2, sort_keys=True))
PY
    python - <<'PY' > "$VLLM_C_STATUS" 2>&1
import importlib

try:
    module = importlib.import_module("vllm._C")
except Exception as exc:
    print("vllm._C=FAIL")
    print(f"error={exc!r}")
else:
    print("vllm._C=PASS")
    print(f"module_file={getattr(module, '__file__', None)}")
PY
else
    echo "python not found" > "$RUNTIME_VERSIONS"
    echo "python not found" > "$VLLM_C_STATUS"
fi
write_section "runtime_versions"
cat "$RUNTIME_VERSIONS" >> "$MANIFEST"
write_section "vllm_c_status"
cat "$VLLM_C_STATUS" >> "$MANIFEST"

: > "$CUDA_HEADERS"
{
    echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
    echo "CUDA_PATH=${CUDA_PATH:-<unset>}"
    echo
    echo "candidate_paths:"
} >> "$CUDA_HEADERS"
for base in \
    "${CUDA_HOME:-}" \
    "${CUDA_PATH:-}" \
    /usr/local/cuda \
    /usr/local/cuda-13 \
    /usr/local/cuda-12 \
    /usr/local/cuda-12.9 \
    /usr/local/cuda-12.8 \
    /usr/local/cuda-12.6; do
    if [ -n "$base" ]; then
        header="$base/include/cuda_runtime.h"
        if [ -f "$header" ]; then
            echo "FOUND $header" >> "$CUDA_HEADERS"
        else
            echo "missing $header" >> "$CUDA_HEADERS"
        fi
    fi
done
write_section "cuda_runtime_headers"
cat "$CUDA_HEADERS" >> "$MANIFEST"

if command -v nvidia-smi >/dev/null 2>&1; then
    run_or_note "$NVIDIA_SMI" nvidia-smi
else
    echo "nvidia-smi not found" > "$NVIDIA_SMI"
fi
write_section "nvidia_smi"
cat "$NVIDIA_SMI" >> "$MANIFEST"

{
    if [ -n "${SOURCE_REVISION:-}" ]; then
        echo "source_revision=$SOURCE_REVISION"
        echo "source_revision_source=SOURCE_REVISION"
    elif [ -d "$PROJECT_ROOT/.git" ] && command -v git >/dev/null 2>&1; then
        echo "source_revision=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo '<git-rev-parse-failed>')"
        echo "source_revision_source=.git"
        echo "source_branch=$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<git-branch-failed>')"
        echo "source_dirty=$(git -C "$PROJECT_ROOT" status --short 2>/dev/null | wc -l | tr -d ' ')"
    elif [ -d "$PROJECT_ROOT/.git" ]; then
        echo "source_revision=<git-directory-present-but-git-command-missing>"
        echo "source_revision_source=.git"
    else
        echo "source_revision=<unknown>"
        echo "source_revision_source=<none>"
    fi
} > "$SOURCE_REV"
write_section "source_revision"
cat "$SOURCE_REV" >> "$MANIFEST"

echo "Frozen environment manifest written to: $OUT_DIR"
echo "Main manifest: $MANIFEST"
