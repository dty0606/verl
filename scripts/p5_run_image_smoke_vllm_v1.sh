#!/usr/bin/env bash
# Run a containerized Tau3 vLLM V1 preflight or tiny training smoke on P5.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
IMAGE_URI="${IMAGE_URI:-}"
SMOKE_MODE="${SMOKE_MODE:-preflight}" # image_preflight, preflight, grpo, sdpo
TASK_PATH="${TASK_PATH:-datasets/tau3_live_airline_canonical_json}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/${TASK_PATH#./}}"
MODEL_PATH="${MODEL_PATH:-}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
TAU3_LIVE_USER_MODEL="${TAU3_LIVE_USER_MODEL:-us.anthropic.claude-sonnet-4-6}"
CONTAINER_NAME="${CONTAINER_NAME:-tau3-vllm-v1-smoke-$(date +%Y%m%d%H%M%S)}"
MOUNT_PROJECT="${MOUNT_PROJECT:-}"

if [ -z "$MOUNT_PROJECT" ]; then
    if [ "$SMOKE_MODE" = "image_preflight" ]; then
        MOUNT_PROJECT=0
    else
        MOUNT_PROJECT=1
    fi
fi

if [ -z "$IMAGE_URI" ]; then
    echo "Error: IMAGE_URI must be set to the local or ECR image tag." >&2
    exit 1
fi
if [ "$SMOKE_MODE" != "image_preflight" ] && [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH must be set to the SFT HF model path visible inside the mounted project root." >&2
    exit 1
fi

case "$SMOKE_MODE" in
    image_preflight|preflight|grpo|sdpo) ;;
    *) echo "Error: SMOKE_MODE must be image_preflight, preflight, grpo, or sdpo" >&2; exit 1 ;;
esac

if [ "$MOUNT_PROJECT" = "1" ]; then
    CONTAINER_PROJECT_ROOT="$PROJECT_ROOT"
else
    CONTAINER_PROJECT_ROOT="/workspace/verl_tau3_sdpo"
fi

INNER_SCRIPT=$(cat <<'EOS'
set -euo pipefail
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
echo "Container cwd: $(pwd)"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "SMOKE_MODE=$SMOKE_MODE"

if [ "$SMOKE_MODE" = "image_preflight" ]; then
  python scripts/p5_preflight_vllm_v1.py --skip-model-config
  exit 0
fi

python scripts/p5_preflight_vllm_v1.py \
  --model-path "$MODEL_PATH" \
  --dataset-dir "$DATASET_DIR"

if [ "$SMOKE_MODE" = "preflight" ]; then
  exit 0
fi

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export TEST_FREQ="${TEST_FREQ:-3}"
export SAVE_FREQ="${SAVE_FREQ:-3}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
export VAL_N="${VAL_N:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
export MODEL_ALIAS="${MODEL_ALIAS:-real_sft_step800}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"
export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-true}"
export TAU3_VLLM_PROFILE="${TAU3_VLLM_PROFILE:-qwen35_v1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TAU3_LIVE_RUNTIME="${TAU3_LIVE_RUNTIME:-official_gym}"
mkdir -p logs outputs/vllm_v1_image_smoke/rollout_data

if [ "$SMOKE_MODE" = "grpo" ]; then
  ROLLOUT_DATA_DIR=outputs/vllm_v1_image_smoke/grpo_rollout_data \
    bash run_local_tau3_grpo_live_p5.sh "$TASK_PATH" vllm_v1_image_smoke json \
    2>&1 | tee logs/vllm_v1_image_grpo_smoke.log
else
  export SDPO_ARM="${SDPO_ARM:-original}"
  ROLLOUT_DATA_DIR=outputs/vllm_v1_image_smoke/sdpo_rollout_data \
    bash run_local_tau3_sdpo_live_p5.sh "$TASK_PATH" vllm_v1_image_smoke json \
    2>&1 | tee logs/vllm_v1_image_sdpo_smoke.log
fi
EOS
)

echo "----------------------------------------------------------------"
echo "Running Tau3 vLLM V1 image smoke"
echo "Image: $IMAGE_URI"
echo "Mode: $SMOKE_MODE"
echo "Project root: $PROJECT_ROOT"
echo "Container project root: $CONTAINER_PROJECT_ROOT"
echo "Mount project: $MOUNT_PROJECT"
echo "Dataset dir: $DATASET_DIR"
echo "Model path: $MODEL_PATH"
echo "AWS region: $AWS_REGION"
echo "----------------------------------------------------------------"

DOCKER_ARGS=(
    --rm
    --gpus all
    --name "$CONTAINER_NAME"
    --net=host
    --ipc=host
    --shm-size="${SHM_SIZE:-64g}"
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface"
    -w "$CONTAINER_PROJECT_ROOT"
    -e "PROJECT_ROOT=$CONTAINER_PROJECT_ROOT"
    -e "TASK_PATH=$TASK_PATH"
    -e "DATASET_DIR=$DATASET_DIR"
    -e "MODEL_PATH=$MODEL_PATH"
    -e "MODEL_ALIAS=${MODEL_ALIAS:-real_sft_step800}"
    -e "AWS_REGION=$AWS_REGION"
    -e "AWS_DEFAULT_REGION=$AWS_REGION"
    -e "TAU3_LIVE_USER_MODEL=$TAU3_LIVE_USER_MODEL"
    -e "TAU3_LIVE_RUNTIME=${TAU3_LIVE_RUNTIME:-official_gym}"
    -e "TAU3_VLLM_PROFILE=${TAU3_VLLM_PROFILE:-qwen35_v1}"
    -e "VLLM_USE_V1=${VLLM_USE_V1:-1}"
    -e "VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
    -e "VLLM_LANGUAGE_MODEL_ONLY=${VLLM_LANGUAGE_MODEL_ONLY:-true}"
    -e "SMOKE_MODE=$SMOKE_MODE"
)

add_optional_env() {
    local key="$1"
    if [ "${!key+x}" = "x" ] && [ -n "${!key}" ]; then
        DOCKER_ARGS+=(-e "$key=${!key}")
    fi
}

for key in \
    VLLM_ENABLE_PREFIX_CACHING \
    VLLM_ENABLE_CHUNKED_PREFILL \
    VLLM_MAX_NUM_BATCHED_TOKENS \
    VLLM_ENFORCE_EAGER \
    VLLM_COMPILATION_CONFIG_JSON \
    VLLM_BLOCK_SIZE \
    VLLM_MAMBA_BLOCK_SIZE \
    VLLM_MAMBA_CACHE_MODE \
    VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER \
    VLLM_KV_CACHE_MEMORY_BYTES \
    VLLM_DISABLE_CASCADE_ATTN \
    TRAIN_BATCH_SIZE \
    VAL_BATCH_SIZE \
    ROLLOUT_BATCH_SIZE \
    PPO_MINI_BATCH_SIZE \
    PPO_MICRO_BATCH_SIZE_PER_GPU \
    TOTAL_TRAINING_STEPS \
    TOTAL_EPOCHS \
    TEST_FREQ \
    SAVE_FREQ \
    VAL_N \
    MAX_PROMPT_LENGTH \
    MAX_RESPONSE_LENGTH \
    MAX_MODEL_LEN \
    ROLLOUT_TP_SIZE \
    ROLLOUT_GPU_MEMORY_UTILIZATION \
    ENABLE_THINKING \
    SDPO_ARM; do
    add_optional_env "$key"
done

if [ "$MOUNT_PROJECT" = "1" ]; then
    DOCKER_ARGS+=(-v "${PROJECT_ROOT}:${PROJECT_ROOT}")
fi

if [ -d "${HOME}/.aws" ]; then
    DOCKER_ARGS+=(-v "${HOME}/.aws:/root/.aws:ro")
fi

docker run "${DOCKER_ARGS[@]}" "$IMAGE_URI" -lc "$INNER_SCRIPT"
