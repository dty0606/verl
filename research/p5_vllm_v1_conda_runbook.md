# P5 vLLM V1 Conda Runbook

Date: 2026-05-07

## Purpose

Build a consistent latest-VERL + vLLM V1 environment on SageMaker Code Editor P5 when Docker is unavailable. This is the runnable path for GRPO, original SDPO, and Memory-SDPO bring-up until the reproducibility image is built on a Docker-capable host.

Target stack for this branch:

- Python `3.12`
- vLLM `0.20.1` with `VLLM_USE_V1=1`
- Torch installed as the ABI-matched vLLM dependency, not manually upgraded later
- CUDA wheel backend `cu129`
- CUDA compiler/NVVM tools `12.9.86`, NVCC/NVVM dev headers, `cuda-cudart-dev`, and CUDA CRT/dev headers for FlashInfer GDN JIT
- flash-attn `2.8.3` community wheel for torch `2.11` / Python `3.12`
- Tau3 live runtime via `tau2-bench` commit `220b47844fb74d4351037e81055cf1e2948e4734`

References checked on 2026-05-06: vLLM marks `v0.20.1` as the latest GitHub release, and vLLM GPU install docs recommend `uv pip install ... --torch-backend=auto` / matching vLLM image versions.

## Platform Facts

- Current SageMaker Code Editor P5 is container-based: no `systemctl`, no `yum`/`apt`, no Docker daemon.
- Do not try Docker-in-Docker on SM CE. Build/push ECR images later from EC2, CodeBuild, GitHub Actions, or a Docker-enabled SageMaker domain.
- P5 may not have Git. Use GitHub for Codex/Kiro sync, then S3 sync snapshots to P5.
- The 99 GB EBS root can fill during checkpoints. Put bulky logs/checkpoints on `/mnt/sagemaker-nvme/` and sync important artifacts to S3 because NVMe is ephemeral.

## Kiro To P5 Snapshot

From a Git-capable Kiro/local machine:

```bash
cd ~/verl_tau3_sdpo
SOURCE_REVISION="$(git rev-parse --short HEAD)"

aws s3 sync ~/verl_tau3_sdpo/ s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/repo/ \
  --exclude ".git/*" \
  --exclude "datasets/*" \
  --exclude "checkpoints/*" \
  --exclude "outputs/*" \
  --exclude "output/*" \
  --exclude "wandb/*" \
  --exclude "logs/*" \
  --region us-west-2
```

If P5 has no Git and no existing `~/tau2-bench`, also sync a tau2-bench snapshot to P5:

```bash
git clone https://github.com/sierra-research/tau2-bench.git /tmp/tau2-bench
git -C /tmp/tau2-bench checkout --detach 220b47844fb74d4351037e81055cf1e2948e4734
aws s3 sync /tmp/tau2-bench/ s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/tau2-bench/ \
  --exclude ".git/*" \
  --region us-west-2
```

On P5:

```bash
mkdir -p ~/verl_tau3_sdpo ~/tau2-bench
aws s3 sync s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/repo/ ~/verl_tau3_sdpo/ --region us-west-2
aws s3 sync s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/tau2-bench/ ~/tau2-bench/ --region us-west-2
```

## East P5 Fresh Bootstrap

Use this when bringing up a new SageMaker Code Editor P5 in `us-east-1`.
Keep code separate from large artifacts so S3 `--delete` / resync operations do
not touch checkpoints or rollout outputs.

```bash
export CODE_DIR=~/verl_tau3_sdpo_vllm20
export ARTIFACT_DIR=~/verl_tau3_sdpo
export S3_PREFIX=s3://tianyd-rlvr-research/tau3-sdpo/latest-verl
export S3_REGION=us-west-2
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1

mkdir -p "$CODE_DIR" "$ARTIFACT_DIR/checkpoints" ~/tau2-bench ~/tmp

# Repo snapshot. P5 does not need Git metadata.
aws s3 sync "$S3_PREFIX/repo/" "$CODE_DIR/" \
  --exclude "datasets/*" \
  --exclude "checkpoints/*" \
  --exclude "outputs/*" \
  --exclude "output/*" \
  --exclude "wandb/*" \
  --exclude "logs/*" \
  --region "$S3_REGION"

# Tau3 runtime dependency snapshot.
aws s3 sync "$S3_PREFIX/tau2-bench/" ~/tau2-bench/ --region "$S3_REGION"

# Dataset and SFT checkpoint. If either prefix is missing, upload it from the
# proven west P5 first, then rerun these syncs.
mkdir -p "$CODE_DIR/datasets/tau3_live_airline_canonical_json"
aws s3 sync "$S3_PREFIX/datasets/tau3_live_airline_canonical_json/" \
  "$CODE_DIR/datasets/tau3_live_airline_canonical_json/" \
  --region "$S3_REGION"

mkdir -p "$ARTIFACT_DIR/checkpoints/SDPO/tau3_verl_sft"
aws s3 sync "$S3_PREFIX/checkpoints/SDPO/tau3_verl_sft/" \
  "$ARTIFACT_DIR/checkpoints/SDPO/tau3_verl_sft/" \
  --region "$S3_REGION"
```

Then build the env from the east P5 code snapshot:

```bash
cd "$CODE_DIR"
mkdir -p logs

ENV_NAME=sdpo-vllm20-v1 \
VLLM_VERSION=0.20.1 \
TORCH_BACKEND=cu129 \
INSTALL_CUDA_TOOLS=1 \
INSTALL_FLASH_ATTN=1 \
TAU2_DIR=~/tau2-bench \
bash scripts/p5_setup_vllm_v1_env.sh \
  2>&1 | tee logs/east_p5_vllm_v1_env_setup.log
```

Export the runtime variables before every preflight, smoke, or full run:

```bash
source activate sdpo-vllm20-v1
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
export TMPDIR="$HOME/tmp"
export VLLM_USE_V1=1
export TAU2_DATA_DIR=~/tau2-bench/data
export TAU3_LIVE_USER_MODEL=us.anthropic.claude-sonnet-4-6
export TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION=0
export TAU3_LIVE_RUNTIME=official_gym
export TAU3_LIVE_FEEDBACK_FORMAT=json
export MODEL_PATH="$ARTIFACT_DIR/checkpoints/SDPO/tau3_verl_sft/TAU3-VERL-SFT-FULL-Qwen-Qwen3.5-4B-qwen35_4b_vlm_full_traj_sft_real_9k/global_step_800/huggingface"
export MODEL_ALIAS=real_sft_step800
export TASK_PATH=datasets/tau3_live_airline_canonical_json
```

East-P5 readiness gates:

```bash
cd "$CODE_DIR"
mkdir -p logs

test -f "$TASK_PATH/train.parquet"
test -f "$TASK_PATH/test.parquet"
test -f "$MODEL_PATH/config.json"

python scripts/p5_preflight_vllm_v1.py \
  --model-path "$MODEL_PATH" \
  --dataset-dir "$PWD/$TASK_PATH" \
  2>&1 | tee logs/east_p5_vllm_v1_preflight.log

python -c "import vllm._C; print('vllm._C OK')"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
which nvcc
which cicc
```

After preflight passes, run the known-good vLLM V1 profile for three GRPO steps
to prove engine startup, rollout, weight update, and vLLM wake-up on the new
host:

```bash
TOTAL_TRAINING_STEPS=3 \
TOTAL_EPOCHS=3 \
TEST_FREQ=3 \
SAVE_FREQ=3 \
VLLM_KV_CACHE_DTYPE=auto \
VLLM_ENABLE_PREFIX_CACHING=true \
TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION=0 \
MAX_RESPONSE_LENGTH=24576 \
MAX_MODEL_LEN=49152 \
TRAIN_BATCH_SIZE=8 \
VAL_BATCH_SIZE=8 \
ROLLOUT_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 \
MODE=grpo \
CONTINUE_ON_FAIL=0 \
CAPACITY_PROFILES="02_auto_prefix_24k_48k" \
PROJECT_NAME=SDPO-vllm-v1-east-readiness \
RUN_NAME_PREFIX=east_p5_grpo_vllm_v1_readiness_3step \
ROLLOUT_OUTPUT_ROOT="$PWD/outputs/east_p5_grpo_vllm_v1_readiness_3step" \
bash scripts/p5_run_vllm_v1_capacity_matrix.sh \
  2>&1 | tee logs/east_p5_grpo_vllm_v1_readiness_3step.log
```

If east P5 will run vanilla SDPO, follow with a one-step original-SDPO smoke
using the same profile:

```bash
TOTAL_TRAINING_STEPS=1 \
TOTAL_EPOCHS=1 \
TEST_FREQ=1 \
SAVE_FREQ=1 \
VLLM_KV_CACHE_DTYPE=auto \
VLLM_ENABLE_PREFIX_CACHING=true \
MAX_RESPONSE_LENGTH=24576 \
MAX_MODEL_LEN=49152 \
TRAIN_BATCH_SIZE=8 \
VAL_BATCH_SIZE=8 \
ROLLOUT_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 \
MODE=sdpo \
SDPO_ARM=original \
CONTINUE_ON_FAIL=0 \
CAPACITY_PROFILES="02_auto_prefix_24k_48k" \
PROJECT_NAME=SDPO-vllm-v1-east-sdpo-smoke \
RUN_NAME_PREFIX=east_p5_original_sdpo_vllm_v1_1step \
ROLLOUT_OUTPUT_ROOT="$PWD/outputs/east_p5_original_sdpo_vllm_v1_1step" \
bash scripts/p5_run_vllm_v1_capacity_matrix.sh \
  2>&1 | tee logs/east_p5_original_sdpo_vllm_v1_1step.log
```

## Build Env

```bash
cd ~/verl_tau3_sdpo

ENV_NAME=sdpo-vllm20-v1 \
VLLM_VERSION=0.20.1 \
TORCH_BACKEND=cu129 \
INSTALL_CUDA_TOOLS=1 \
INSTALL_FLASH_ATTN=1 \
TAU2_DIR=~/tau2-bench \
bash scripts/p5_setup_vllm_v1_env.sh

source activate sdpo-vllm20-v1
export VLLM_USE_V1=1
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"
export TAU2_DATA_DIR=~/tau2-bench/data
export TAU3_LIVE_RUNTIME=official_gym
export TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION=0
export TAU3_LIVE_FEEDBACK_FORMAT=json
```

If setup prints `WARN: uv does not expose --torch-backend`, it should fall back to the `https://wheels.vllm.ai/0.20.1/cu129` wheel index. Treat the env as suspect until preflight proves `vllm._C`, Torch CUDA, and the vLLM version. If setup cannot find `cuda_runtime.h`, `crt/host_config.h`, `fatbinary_section.h`, `nvcc`, or `cicc`, fix the CUDA/NVVM env before full runs; FlashInfer/GDN JIT depends on those pieces for Qwen3.5 linear-attention kernels. `cuda-cudart` alone may not install headers on SM CE; install `cuda-cudart-dev=12.9.79` if `cuda_runtime.h` is missing, `cuda-crt=12.9.86` plus `cuda-crt-dev_linux-64=12.9.86` if `crt/host_config.h` is missing, and `cuda-nvcc-dev_linux-64=12.9.86` plus `cuda-nvvm-dev_linux-64=12.9.86` if `fatbinary_section.h` or other NVCC internal headers are missing.

East P5 confirmation: the final missing-header ladder was
`cuda_runtime.h` -> `crt/host_config.h` -> `fatbinary_section.h`. Installing
the CUDA dev packages above and clearing the FlashInfer cache fixed the GDN JIT
startup path on the east instance.

## Verify Inputs

```bash
export AWS_REGION=us-east-1
aws sts get-caller-identity

export TAU3_LIVE_USER_MODEL=us.anthropic.claude-sonnet-4-6
export MODEL_PATH=~/verl_tau3_sdpo/checkpoints/SDPO/tau3_verl_sft/TAU3-VERL-SFT-FULL-Qwen-Qwen3.5-4B-qwen35_4b_vlm_full_traj_sft_real_9k/global_step_800/huggingface
export MODEL_ALIAS=real_sft_step800
export TASK_PATH=datasets/tau3_live_airline_canonical_json

test -f "$TASK_PATH/train.parquet"
test -f "$TASK_PATH/test.parquet"
test -f "$MODEL_PATH/config.json"
```

Use an S3 checkpoint/dataset sync before this block if any path is missing. Do not let direct launchers fall back to `Qwen/Qwen3.5-4B`; the capacity matrix requires `MODEL_PATH` and is safer for gating.

## Preflight

```bash
mkdir -p logs
python scripts/p5_preflight_vllm_v1.py \
  --model-path "$MODEL_PATH" \
  --dataset-dir "$PWD/$TASK_PATH" \
  2>&1 | tee logs/vllm_v1_preflight_$(date +%Y%m%d_%H%M%S).log

python -c "import vllm._C; print('vllm._C OK')"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
which nvcc
which cicc
test -f "$CUDA_HOME/include/cuda_runtime.h"
test -f "$CUDA_HOME/include/crt/host_config.h"
test -f "$CUDA_HOME/include/fatbinary_section.h"
```

Required pass signals:

- `PREFLIGHT=PASS`
- `VLLM_USE_V1=1`
- `VLLM_C_EXTENSION=PASS`
- `CUDA_RUNTIME_HEADER` is not `<missing>`
- CUDA compiler internal headers resolve: `crt/host_config.h` and `fatbinary_section.h`
- `flash_attn` imports
- `nvcc` and `cicc` resolve from the conda CUDA 12.9 toolchain
- `TAU3_PARSER=PASS` and numeric XML params remain ints
- Torch/vLLM/FlashInfer/Transformers versions print coherently
- `train.parquet`, `test.parquet`, and model config exist

Preflight is necessary but not sufficient. It does not instantiate the vLLM engine or catch Qwen3.5 hybrid-KV page-size failures. The GRPO smoke is the real engine proof.

## Capacity Smoke

Exploratory run that records all profiles:

```bash
CONTINUE_ON_FAIL=1 MODE=grpo bash scripts/p5_run_vllm_v1_capacity_matrix.sh
cat logs/vllm_v1_capacity_matrix/*/summary.txt
```

Gating run that stops on first failure:

```bash
CONTINUE_ON_FAIL=0 MODE=grpo bash scripts/p5_run_vllm_v1_capacity_matrix.sh
```

Ten-step readiness run for the known-good auto-KV + prefix-caching profile:

```bash
TOTAL_TRAINING_STEPS=10 \
TOTAL_EPOCHS=10 \
TEST_FREQ=10 \
SAVE_FREQ=10 \
VLLM_KV_CACHE_DTYPE=auto \
VLLM_ENABLE_PREFIX_CACHING=true \
TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION=0 \
MAX_RESPONSE_LENGTH=24576 \
MAX_MODEL_LEN=49152 \
TRAIN_BATCH_SIZE=8 \
VAL_BATCH_SIZE=8 \
ROLLOUT_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 \
MODE=grpo \
CONTINUE_ON_FAIL=0 \
CAPACITY_PROFILES="02_auto_prefix_24k_48k" \
bash scripts/p5_run_vllm_v1_capacity_matrix.sh
```

`CAPACITY_PROFILES` is intentionally required for readiness gating. Without it,
the script will run the full matrix and may continue into experimental FP8 or
larger-context profiles.

The profile order is:

1. `01_auto_no_prefix_24k_48k`
2. `02_auto_prefix_24k_48k`
3. `03_fp8_no_prefix_noscales_24k_48k`
4. `04_fp8_prefix_noscales_24k_48k`
5. `05_fp8_prefix_noscales_32k_64k`

The capacity matrix defaults to `TRAIN_BATCH_SIZE=8`, `ROLLOUT_BATCH_SIZE=8`, and `PPO_MINI_BATCH_SIZE=8` so the real train batch is divisible by 8 P5 GPUs. Do not lower those values on 8-GPU P5 unless you also change the GPU count/config coherently.

The default FP8 profiles intentionally use `VLLM_CALCULATE_KV_SCALES=false`. On Qwen3.5 hybrid attention/linear-attention models, vLLM 0.20.1 can crash during dynamic FP8 KV scale initialization because some hybrid cache entries are lists rather than tensors. Only run the dynamic-scale variants with `RUN_EXPERIMENTAL_FP8_SCALES=1` after the no-scale FP8 profiles pass.

The environment is ready for overnight experiments only after at least one GRPO smoke completes 2-3 training steps without ABI errors, FlashInfer/GDN CUDA-header errors, hybrid-KV page-size errors, parser/tool-call regressions, Bedrock access failures, or disk-pressure checkpoint errors.

Run original SDPO only after GRPO proves engine startup:

```bash
MODE=sdpo SDPO_ARM=original CONTINUE_ON_FAIL=0 \
  bash scripts/p5_run_vllm_v1_capacity_matrix.sh
```

## Hybrid-KV Fallbacks

Use these only for tiny smokes, not overnight training:

```bash
VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER=true CONTINUE_ON_FAIL=0 MODE=grpo \
  bash scripts/p5_run_vllm_v1_capacity_matrix.sh
```

```bash
VLLM_ENABLE_PREFIX_CACHING=true \
VLLM_BLOCK_SIZE=16 \
VLLM_MAMBA_BLOCK_SIZE=16 \
VLLM_MAMBA_CACHE_MODE=align \
CONTINUE_ON_FAIL=0 MODE=grpo \
  bash scripts/p5_run_vllm_v1_capacity_matrix.sh
```

If the page-size error persists after these bounded checks, stop and report. Do not silently fall back to vLLM V0 in this branch.

## Freeze Artifact

After the first passing profile:

```bash
RUN_NAME=vllm_v1_$(date +%Y%m%d_%H%M%S)
SOURCE_REVISION=<github_commit_short_sha> \
  bash scripts/p5_export_frozen_env.sh "outputs/frozen_env/$RUN_NAME"

python scripts/probe_runtime_surface.py --indent 2 \
  > "outputs/frozen_env/$RUN_NAME/runtime_surface.json"

aws s3 sync "outputs/frozen_env/$RUN_NAME/" \
  "s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/frozen_env/$RUN_NAME/" \
  --region us-west-2
```

Record in `research/session_sync.md`: source revision, env name, vLLM/Torch/CUDA versions, passing profile, exact command, S3 log prefix, and stop condition if any.

## Stop Conditions

- `vllm._C` import failure or undefined symbol: torch/vLLM ABI mismatch.
- `CUDA_RUNTIME_HEADER=<missing>`, missing `cicc`, missing `crt/host_config.h` / `fatbinary_section.h`, or repeated FlashInfer GDN JIT failure: Qwen3.5 linear-attention inference is not proven for full runs.
- `NotImplementedError: The page size of the layer is not divisible by the maximum page size`: Qwen3.5 hybrid-KV incompatibility.
- `TAU3_PARSER=FAIL`: numeric XML parser regression.
- Bedrock credential/model-access failure after preflight/engine proof.
- Disk errors on EBS root; move outputs to NVMe/S3 before continuing.
