# P5 Environment Issues & Working Setup Guide

Date: 2026-05-07 (updated after successful vLLM V1 bringup)

## Working Stack (Proven on West P5, Profiles 01/02 PASS)

```
Env name:           sdpo-vllm20-v1
Python:             3.12.13
torch:              2.11.0+cu129
vLLM:               0.20.1 (cu129 wheel from wheels.vllm.ai)
flash-attn:         2.8.3 (community wheel for torch 2.11 + cu12)
flashinfer-python:  0.6.8.post1
transformers:       5.8.0
tokenizers:         0.22.2
ray:                2.55.1
CUDA driver:        580.126.09 (supports CUDA 13.0)
CUDA toolkit:       12.9 (conda cuda-nvcc-tools + cuda-nvvm-tools + cuda-cudart-dev + cuda-crt)
GPU:                8× NVIDIA H100 80GB HBM3
VLLM_USE_V1:       1
```

## Correct Setup Flow (Reproducible)

### Step 1: Create conda env and install vLLM from cu129 wheel index

```bash
conda create -y -n sdpo-vllm20-v1 python=3.12
source activate sdpo-vllm20-v1
python -m pip install --upgrade pip uv

# CRITICAL: Install vLLM from their cu129 wheel index, NOT PyPI default.
# PyPI default gives a CUDA 13 wheel that won't work on CUDA 12.x systems.
pip install vllm==0.20.1 --extra-index-url https://wheels.vllm.ai/0.20.1/cu129
```

### Step 2: Install CUDA compiler tools (for FlashInfer GDN kernel JIT)

```bash
# Install nvcc + cicc (NVVM compiler needed for FlashInfer GDN kernels)
conda install -n sdpo-vllm20-v1 -y \
  -c nvidia/label/cuda-12.9.1 \
  cuda-nvcc-tools=12.9.86 \
  cuda-nvvm-tools=12.9.86 \
  cuda-cudart-dev=12.9.79 \
  cuda-crt=12.9.86 \
  --no-update-deps
```

### Step 3: Install flash-attn (community wheel for torch 2.11)

```bash
# No prebuilt official wheel exists for torch 2.11+cu129.
# Use the community wheel from lesj0610 (same source as old env):
pip install --no-deps \
  "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

### Step 4: Install VERL/Tau3 runtime dependencies

```bash
pip install \
    accelerate addict boto3 codetiming datasets deepdiff dill fastuuid \
    "gymnasium>=1.2.2" hydra-core latex2sympy2_extended \
    "litellm>=1.80.15,<1.82.7" mathruler multiprocess "numpy<2.0.0" \
    pandas peft "pyarrow>=19.0.0" pybind11 pylatexenc qwen-vl-utils \
    "ray[default]>=2.41.0" tenacity tensorboard \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" toml torchdata wandb xxhash

# Install VERL repo as editable (no deps to avoid torch/vllm drift)
cd ~/verl_tau3_sdpo_vllm20
pip install -e . --no-deps

# Install tau2-bench
pip install -e ~/tau2-bench --no-deps
```

### Step 5: Set environment variables before every run

```bash
source activate sdpo-vllm20-v1

# CUDA paths (critical for FlashInfer GDN JIT)
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

# Stable temp dir (FlashInfer JIT writes large intermediate files)
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

# vLLM V1 mode
export VLLM_USE_V1=1

# Tau3 runtime
export TAU2_DATA_DIR=~/tau2-bench/data
export TAU3_LIVE_USER_MODEL=us.anthropic.claude-sonnet-4-6
export TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION=0
export TAU3_LIVE_RUNTIME=official_gym
```

### Step 6: Verify

```bash
python -c "import vllm._C; print('vllm._C OK')"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
which nvcc
which cicc
python scripts/p5_preflight_vllm_v1.py --model-path "$MODEL_PATH" --dataset-dir "$TASK_PATH"
```

## Issues Encountered & Resolutions

### 1. vLLM 0.20.1 PyPI wheel requires CUDA 13 (`libcudart.so.13`)

**Symptom:** `ImportError: libcudart.so.13: cannot open shared object file`

**Root cause:** The default PyPI wheel for vLLM 0.20.1 is compiled against CUDA 13. SM CE P5 has CUDA 12.x runtime.

**Fix:** Install from vLLM's cu129 wheel index:
```bash
pip install vllm==0.20.1 --extra-index-url https://wheels.vllm.ai/0.20.1/cu129 --no-deps
```

**Source:** [vLLM forum thread](https://discuss.vllm.ai/t/install-using-torch-backend-cu129-but-try-to-import-cu13/2601)

### 2. FlashInfer GDN kernel JIT fails: `cicc: not found`

**Symptom:** `sh: 1: cicc: not found` during GDN prefill kernel warmup

**Root cause:** conda `cuda-nvcc` package doesn't include `cicc` (NVVM intermediate compiler). Need `cuda-nvvm-tools`.

**Fix:**
```bash
conda install -n sdpo-vllm20-v1 -y -c nvidia/label/cuda-12.9.1 \
  cuda-nvcc-tools=12.9.86 cuda-nvvm-tools=12.9.86 --no-update-deps
export PATH="$CONDA_PREFIX/nvvm/bin:$PATH"
```

### 3. FlashInfer GDN kernel JIT fails: `cannot find -lcuda`

**Symptom:** `ld: cannot find -lcuda: No such file or directory`

**Root cause:** The linker can't find `libcuda.so` (CUDA driver stub). It's at `/usr/lib/x86_64-linux-gnu/` but not on the linker search path.

**Fix:**
```bash
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
```

### 4. FlashInfer GDN kernel JIT fails: `cuda_runtime.h: No such file or directory`

**Symptom:** `fatal error: cuda_runtime.h: No such file or directory`

**Root cause:** `CUDA_HOME` points to the wrong location, or the CUDA runtime
header package was not installed. `cuda-cudart` is not enough on some SM CE
conda builds; install `cuda-cudart-dev` so the header lands under the target
CUDA tree.

**Fix:**
```bash
conda install -n sdpo-vllm20-v1 -y -c nvidia/label/cuda-12.9.1 \
  cuda-cudart-dev=12.9.79 --no-update-deps
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
ls "$CUDA_HOME/include/cuda_runtime.h"
```

### 4b. FlashInfer GDN kernel JIT fails: `crt/host_config.h: No such file or directory`

**Symptom:** `cuda_runtime.h:82:10: fatal error: crt/host_config.h: No such file or directory`

**Root cause:** `cuda_runtime.h` is present, but the CUDA CRT internal header
package is missing. This is the next layer of the same FlashInfer/GDN JIT setup
problem, not a Qwen/vLLM algorithm issue.

**Fix:**
```bash
conda install -n sdpo-vllm20-v1 -y -c nvidia/label/cuda-12.9.1 \
  cuda-crt=12.9.86 --no-update-deps
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
ls "$CUDA_HOME/include/crt/host_config.h"
rm -rf ~/.cache/flashinfer/0.6.8.post1/90a/cached_ops/gdn_prefill_sm90
```

### 5. flash-attn build from source fails (no nvcc / incomplete toolkit)

**Symptom:** `pip install flash-attn` fails with ninja build errors

**Root cause:** SM CE doesn't have a full CUDA toolkit pre-installed. Even after installing conda nvcc, the flash-attn build may fail due to missing headers or 404 on prebuilt wheel download.

**Fix:** Use the community prebuilt wheel:
```bash
pip install --no-deps \
  "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

### 6. vLLM 0.20.x hybrid KV page-size error (resolved in smoke by latest setup)

**Symptom:** `NotImplementedError: The page size of the layer is not divisible by the maximum page size`

**Root cause:** Qwen3.5 has mixed attention/linear-attention layers with different KV cache page sizes. vLLM V1's prefix caching tries to unify them.

**Initial fix:** Disable prefix caching:
```bash
export VLLM_ENABLE_PREFIX_CACHING=false
```

**Status:** Profile 01 (auto KV, no prefix caching) PASSED. Profile 02 (auto KV, prefix caching) also PASSED in the latest west-P5 capacity matrix.

### 7. FlashAttention2 not installed (FSDP actor model loading)

**Symptom:** `ImportError: FlashAttention2 has been toggled on, but it cannot be used`

**Root cause:** The model checkpoint config has `_attn_implementation: flash_attention_2` but flash-attn wasn't installed yet.

**Fix:** Install flash-attn (see #5 above).

### 8. Batch size validation error

**Symptom:** `real_train_batch_size (4) must be divisible by minimal possible batch size (8)`

**Root cause:** Capacity matrix defaults used `TRAIN_BATCH_SIZE=2`, `ROLLOUT_BATCH_SIZE=2`, giving `real_train_batch_size = 2*2 = 4` which isn't divisible by 8 GPUs.

**Fix:** Use `TRAIN_BATCH_SIZE=8 ROLLOUT_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8`. The capacity-matrix script now defaults to these 8-GPU-safe values.

### 9. tokenizers/huggingface_hub resolver conflict

**Symptom:** pip resolver fails with conflicting huggingface-hub version requirements

**Fix:** Install transformers and tokenizers with `--no-deps`, then install huggingface_hub separately:
```bash
pip install transformers==5.6.2 --no-deps
pip install tokenizers==0.22.0 --no-deps
pip install "huggingface_hub>=1.5.0" --no-deps
```

### 10. No Docker on SM CE

**Symptom:** `docker: command not found`, no `systemctl`, no `yum`/`apt`

**Root cause:** SageMaker Code Editor runs inside a container. No Docker-in-Docker.

**Status:** Use conda env path for now. Docker/ECR for reproducibility later on EC2/CodeBuild.

## GDN Kernel JIT: Smoke Warning vs Overnight Gate

The FlashInfer GDN kernel JIT failure for Qwen3.5's linear attention layers can appear as a **warning** and a tiny smoke may continue, but do not treat the env as fully proven for overnight runs while it is still failing. Qwen3.5 depends on these linear-attention/GDN paths during vLLM inference; failed warmup can cause slower first inference, higher memory use, or OOM during autotuning.

To fully fix GDN JIT before full runs:
1. All CUDA paths must be set (CUDA_HOME, PATH with nvvm/bin, LIBRARY_PATH with libcuda.so)
2. Clear the failed cache: `rm -rf ~/.cache/flashinfer/0.6.8.post1/90a/cached_ops/gdn_prefill_sm90`
3. Rerun — kernels will JIT-compile (~5 min first time, then cached)

## Platform Constraints (SM CE)

- Container-based: no `systemctl`, no `yum`/`apt`, no Docker daemon
- 99 GB EBS root (cannot resize after domain creation)
- NVMe is ephemeral (28 TB `/mnt/sagemaker-nvme/`, lost on stop)
- No root access for system-level CUDA installs
- Use `source activate <env>` not `conda activate <env>`
- conda deactivation hooks may fail with `set -u` (unbound CONDA_BACKUP_CXX)

### 11. FP8 KV dynamic-scale initialization fails on Qwen3.5 hybrid cache

**Symptom:** `AttributeError: 'list' object has no attribute 'zero_'` in `gpu_model_runner.py:init_fp8_kv_scales`.

**Root cause:** vLLM 0.20.1 dynamic FP8 KV scale initialization assumes each cache entry is a tensor. Qwen3.5 hybrid attention/linear-attention cache entries can include list-shaped recurrent/GDN state, so `calculate_kv_scales=true` crashes during vLLM wake-up.

**Fix:** First test FP8 KV without dynamic scale calculation:
```bash
export VLLM_KV_CACHE_DTYPE=fp8
export VLLM_CALCULATE_KV_SCALES=false
```

The capacity matrix now defaults FP8 profiles to `calculate_kv_scales=false`. Dynamic scale variants are opt-in only with `RUN_EXPERIMENTAL_FP8_SCALES=1`.

## Capacity Matrix Results (West P5, 2026-05-07)

| Profile | KV dtype | Prefix cache | Response/Model len | Status |
|---------|----------|-------------|-------------------|--------|
| 01_auto_no_prefix_24k_48k | auto | off | 24K/48K | **PASS** ✅ |
| 02_auto_prefix_24k_48k | auto | on | 24K/48K | **PASS** ✅ |
| 03_fp8_no_prefix_calcscales_24k_48k | fp8 | off | 24K/48K | FAIL (`init_fp8_kv_scales` list cache bug) |
| 03_fp8_no_prefix_noscales_24k_48k | fp8 | off | 24K/48K | needs test |
| 04_fp8_prefix_noscales_24k_48k | fp8 | on | 24K/48K | needs test |
| 05_fp8_prefix_noscales_32k_64k | fp8 | on | 32K/64K | needs test |
