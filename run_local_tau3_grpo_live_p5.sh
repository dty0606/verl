#!/usr/bin/env bash
# Vanilla GRPO baseline for tau3 live on latest VERL.
#
# Usage:
#   ./run_local_tau3_grpo_live_p5.sh <task_path> [experiment_name_suffix] [feedback_mode]

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task_path> [experiment_name_suffix] [feedback_mode]"
    exit 1
fi

TASK_INPUT="$1"
SUFFIX="${2:-tau3_grpo_baseline}"
FEEDBACK_MODE="${3:-structured}"

case "$FEEDBACK_MODE" in
    structured|json) FEEDBACK_MODE="json" ;;
    text|plain|plain_text) FEEDBACK_MODE="plain_text" ;;
    *) echo "Error: feedback_mode must be json or plain_text"; exit 1 ;;
esac

export TAU3_LIVE_RUNTIME="${TAU3_LIVE_RUNTIME:-official_gym}"
export TAU3_LIVE_FEEDBACK_FORMAT="$FEEDBACK_MODE"
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"
export SDPO_OUTPUT_ROOT="${SDPO_OUTPUT_ROOT:-$PROJECT_ROOT/output/SDPO}"
export SDPO_CHECKPOINT_ROOT="${SDPO_CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints/SDPO}"

if [ -z "${TAU3_LIVE_USER_MODEL:-}" ]; then
    echo "Error: TAU3_LIVE_USER_MODEL must be set."
    exit 1
fi

if [[ "$TASK_INPUT" = /* ]]; then
    TASK_DIR="$TASK_INPUT"
else
    TASK_DIR="$PROJECT_ROOT/${TASK_INPUT#./}"
fi
if [ ! -f "$TASK_DIR/train.parquet" ] || [ ! -f "$TASK_DIR/test.parquet" ]; then
    echo "Error: missing train.parquet or test.parquet in $TASK_DIR"
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/:' '--')
EXP_NAME="LOCAL-TAU3-GRPO-${FEEDBACK_MODE}-${MODEL_NAME}-${SUFFIX}"
mkdir -p "$SDPO_OUTPUT_ROOT" "$SDPO_CHECKPOINT_ROOT" logs
rm -f /dev/shm/verl_dist_store_* 2>/dev/null || true

ARGS=(
    --config-name=tau3_grpo_live
    "data.train_files=$TASK_DIR/train.parquet"
    "data.val_files=$TASK_DIR/test.parquet"
    "data.train_batch_size=${TRAIN_BATCH_SIZE:-12}"
    "data.val_batch_size=${VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-12}}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH:-16384}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH:-4096}"
    "data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING:-true}"
    "max_model_len=${MAX_MODEL_LEN:-24576}"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.actor.optim.lr=${LR:-1e-6}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS:-0}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-12}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
    "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
    "actor_rollout_ref.rollout.n=${ROLLOUT_BATCH_SIZE:-8}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE:-1}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-0.4}"
    "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-0.95}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.4}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.95}"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-4}"
    "algorithm.adv_estimator=grpo"
    "trainer.project_name=${PROJECT_NAME:-SDPO-${USER}}"
    "trainer.experiment_name=$EXP_NAME"
    "trainer.total_epochs=${TOTAL_EPOCHS:-300}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-300}"
    "trainer.test_freq=${TEST_FREQ:-30}"
    "trainer.save_freq=${SAVE_FREQ:-30}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8}"
    "trainer.nnodes=${NNODES:-1}"
)

if [ -n "${ROLLOUT_DATA_DIR:-}" ]; then
    ARGS+=("trainer.rollout_data_dir=$ROLLOUT_DATA_DIR")
fi
if [ -n "${VLLM_LANGUAGE_MODEL_ONLY:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=$VLLM_LANGUAGE_MODEL_ONLY")
fi

echo "----------------------------------------------------------------"
echo "Starting latest-VERL tau3 GRPO baseline"
echo "Experiment: $EXP_NAME"
echo "Model: $MODEL_PATH"
echo "Task dir: $TASK_DIR"
echo "Thinking: ${ENABLE_THINKING:-true}"
echo "Rollout n: ${ROLLOUT_BATCH_SIZE:-8}"
echo "----------------------------------------------------------------"

python3 -m verl.trainer.main_ppo "${ARGS[@]}" "${@:4}"
