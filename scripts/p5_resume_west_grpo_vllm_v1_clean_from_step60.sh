#!/usr/bin/env bash
# Resume the West-P5 clean GRPO vLLM V1 run from its saved step-60 checkpoint.
#
# This is intentionally infrastructure-only: same fixed parser, same cleaned
# Tau3 observation mode, same 24K/49K auto-KV prefix-caching profile.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
cd "$PROJECT_ROOT"

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Error: activate the sdpo-vllm20-v1 conda env before running this helper." >&2
    exit 1
fi

NVME_ROOT="${NVME_ROOT:-/mnt/sagemaker-nvme/tau3_sdpo}"
RUN_STEM="${RUN_STEM:-grpo_vllm_v1_clean_300_v2_resume60}"
OLD_PROJECT_NAME="${OLD_PROJECT_NAME:-SDPO-vllm-v1-clean-grpo-v2}"
OLD_EXPERIMENT_NAME="${OLD_EXPERIMENT_NAME:-LOCAL-TAU3-GRPO-json-real_sft_step800-grpo_vllm_v1_clean_300_v2_02_auto_prefix_24k_48k}"
mkdir -p "$NVME_ROOT"/{tmp,ray_tmp,logs,outputs,output,checkpoints,wandb,cache}

export AWS_REGION="${AWS_REGION:-us-west-2}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-$AWS_REGION}"
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX/targets/x86_64-linux}"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

# Keep Ray, Python temp files, W&B, HF/Triton/Torch caches, checkpoints, logs,
# and rollout dumps off the tiny container overlay root.
export TMPDIR="$NVME_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export RAY_TMPDIR="$NVME_ROOT/ray_tmp"
export SDPO_OUTPUT_ROOT="$NVME_ROOT/output/SDPO"
export SDPO_CHECKPOINT_ROOT="$NVME_ROOT/checkpoints/SDPO"
export WANDB_DIR="$NVME_ROOT/wandb"
export WANDB_CACHE_DIR="$NVME_ROOT/cache/wandb"
export HF_HOME="$NVME_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$NVME_ROOT/cache/torch"
export TRITON_CACHE_DIR="$NVME_ROOT/cache/triton"
export XDG_CACHE_HOME="$NVME_ROOT/cache/xdg"

if [ -z "${RESUME_FROM_PATH:-}" ]; then
    RESUME_FROM_PATH="$(find \
        "$PROJECT_ROOT/checkpoints/SDPO" \
        "$NVME_ROOT/checkpoints/SDPO" \
        -path "*/${OLD_PROJECT_NAME}/${OLD_EXPERIMENT_NAME}/global_step_60" \
        -type d 2>/dev/null | sort | head -1 || true)"
fi
if [ -z "${RESUME_FROM_PATH:-}" ]; then
    echo "Error: could not find global_step_60 for $OLD_EXPERIMENT_NAME." >&2
    echo "Set RESUME_FROM_PATH=/absolute/path/to/global_step_60 and rerun." >&2
    exit 1
fi
if [ ! -f "$RESUME_FROM_PATH/data.pt" ] || [ ! -d "$RESUME_FROM_PATH/actor" ]; then
    echo "Error: resume path does not look like a VERL global_step checkpoint:" >&2
    echo "  $RESUME_FROM_PATH" >&2
    find "$RESUME_FROM_PATH" -maxdepth 2 -type f | head -20 >&2 || true
    exit 1
fi

overlay_avail_kb=$(df -Pk / 2>/dev/null | awk 'NR==2 {print $4}')
min_overlay_avail_kb=$(( ${MIN_OVERLAY_FREE_GB:-2} * 1024 * 1024 ))
if [ -n "${overlay_avail_kb:-}" ] && [ "$overlay_avail_kb" -lt "$min_overlay_avail_kb" ]; then
    echo "Error: container root '/' has less than ${MIN_OVERLAY_FREE_GB:-2}GB free." >&2
    echo "Clean overlay-backed temp paths before resuming, for example /tmp/ray and /tmp/tmpxft_*." >&2
    df -h / >&2 || true
    exit 1
fi

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$HOME/tau2-bench/data}"
export TAU3_LIVE_USER_MODEL="${TAU3_LIVE_USER_MODEL:-us.anthropic.claude-sonnet-4-6}"
export TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION="${TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION:-0}"
export TAU3_LIVE_RUNTIME="${TAU3_LIVE_RUNTIME:-official_gym}"
export TAU3_LIVE_FEEDBACK_FORMAT="${TAU3_LIVE_FEEDBACK_FORMAT:-json}"
export TAU3_BEDROCK_MAX_RETRIES="${TAU3_BEDROCK_MAX_RETRIES:-3}"
export TAU3_BEDROCK_RETRY_DELAYS="${TAU3_BEDROCK_RETRY_DELAYS:-15,30,60}"
export TAU3_BEDROCK_RETRY_JITTER="${TAU3_BEDROCK_RETRY_JITTER:-0.2}"
export TAU3_MASK_ENV_ERROR_ROLLOUTS="${TAU3_MASK_ENV_ERROR_ROLLOUTS:-1}"
export TAU3_ENV_ERROR_SKIP_UPDATE_THRESHOLD="${TAU3_ENV_ERROR_SKIP_UPDATE_THRESHOLD:-0.25}"
export TAU3_RETRY_STEP_ON_TRANSIENT="${TAU3_RETRY_STEP_ON_TRANSIENT:-0}"
if [ -z "${TAU3_LIVE_USER_ARGS_JSON:-}" ]; then
    export TAU3_LIVE_USER_ARGS_JSON='{"num_retries":8,"timeout":120}'
fi

export MODEL_PATH="${MODEL_PATH:-$HOME/verl_tau3_sdpo/checkpoints/SDPO/tau3_verl_sft/TAU3-VERL-SFT-FULL-Qwen-Qwen3.5-4B-qwen35_4b_vlm_full_traj_sft_real_9k/global_step_800/huggingface}"
export MODEL_ALIAS="${MODEL_ALIAS:-real_sft_step800}"
export TASK_PATH="${TASK_PATH:-datasets/tau3_live_airline_canonical_json}"

export VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
export VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-true}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-24576}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export VAL_N="${VAL_N:-4}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-$TOTAL_TRAINING_STEPS}"
export TEST_FREQ="${TEST_FREQ:-30}"
export SAVE_FREQ="${SAVE_FREQ:-30}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-4}"
export MODE=grpo
export CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
export CAPACITY_PROFILES="${CAPACITY_PROFILES:-02_auto_prefix_24k_48k}"
export PROJECT_NAME="${PROJECT_NAME:-SDPO-vllm-v1-clean-grpo-v2}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-$RUN_STEM}"
export LOG_ROOT="${LOG_ROOT:-$NVME_ROOT/logs/$RUN_STEM}"
export ROLLOUT_OUTPUT_ROOT="${ROLLOUT_OUTPUT_ROOT:-$NVME_ROOT/outputs/$RUN_STEM}"
export RESUME_MODE=resume_path
export RESUME_FROM_PATH
export CHECKPOINT_LOAD_CONTENTS="${CHECKPOINT_LOAD_CONTENTS:-[\"model\",\"optimizer\",\"extra\"]}"
export CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"optimizer\",\"extra\"]}"

mkdir -p "$LOG_ROOT" "$ROLLOUT_OUTPUT_ROOT" "$SDPO_CHECKPOINT_ROOT"

echo "----------------------------------------------------------------"
echo "West P5 clean-GRPO resume"
echo "Project root: $PROJECT_ROOT"
echo "NVME_ROOT: $NVME_ROOT"
echo "Run stem: $RUN_STEM"
echo "Project: $PROJECT_NAME"
echo "Resume path: $RESUME_FROM_PATH"
echo "New checkpoint root: $SDPO_CHECKPOINT_ROOT"
echo "Rollouts: $ROLLOUT_OUTPUT_ROOT"
echo "Logs: $LOG_ROOT"
echo "Ray tmp: $RAY_TMPDIR"
echo "TMPDIR: $TMPDIR"
echo "W&B dir: $WANDB_DIR"
echo "----------------------------------------------------------------"
df -h / /home/sagemaker-user /mnt/sagemaker-nvme || true

bash "$PROJECT_ROOT/scripts/p5_run_vllm_v1_capacity_matrix.sh" \
    2>&1 | tee "$LOG_ROOT.console.log"
