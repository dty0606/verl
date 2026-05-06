#!/usr/bin/env bash
# SDPO baseline for tau3 live on latest VERL.
#
# SDPO_ARM controls the teacher-context variant:
#   original      -> paper-style hybrid: successful peer if available,
#                    otherwise failed-sample environment feedback
#   vanilla_peer  -> diagnostic successful-peer-only teacher, no env feedback
#   feedback_only -> diagnostic failed-sample feedback-only teacher, no peer solution
#
# Usage:
#   ./run_local_tau3_sdpo_live_p5.sh <task_path> [experiment_name_suffix] [feedback_mode]

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task_path> [experiment_name_suffix] [feedback_mode]"
    exit 1
fi

TASK_INPUT="$1"
SUFFIX="${2:-tau3_sdpo_baseline}"
FEEDBACK_MODE="${3:-json}"

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

compact_model_name() {
    local model_path="$1"
    local alias="${MODEL_ALIAS:-${MODEL_NAME:-}}"
    if [ -n "$alias" ]; then
        echo "$alias"
        return
    fi

    local step
    step=$(echo "$model_path" | grep -oE 'global_step_[0-9]+' | tail -1 || true)
    if [ -n "$step" ]; then
        echo "ckpt-${step#global_step_}"
        return
    fi

    echo "$model_path" | tr '/:' '--'
}

MODEL_NAME=$(compact_model_name "$MODEL_PATH" | tr -cs '[:alnum:]_.-' '-' | sed -E 's/^-+|-+$//g; s/-{2,}/-/g')
SDPO_ARM="${SDPO_ARM:-original}"
case "$SDPO_ARM" in
    original|paper|paper_hybrid|hybrid|vanilla|vanilla_sdpo) SDPO_ARM="original" ;;
    vanilla_peer|peer|successful_peer) SDPO_ARM="vanilla_peer" ;;
    feedback|feedback_only|ff_sdpo) SDPO_ARM="feedback_only" ;;
    *) echo "Error: SDPO_ARM must be original, vanilla_peer, or feedback_only"; exit 1 ;;
esac
EXP_NAME="LOCAL-TAU3-SDPO-${SDPO_ARM}-${FEEDBACK_MODE}-${MODEL_NAME}-${SUFFIX}"
mkdir -p "$SDPO_OUTPUT_ROOT" "$SDPO_CHECKPOINT_ROOT" logs
rm -f /dev/shm/verl_dist_store_* 2>/dev/null || true

TAU3_VLLM_PROFILE="${TAU3_VLLM_PROFILE:-qwen35_v1}"
case "$TAU3_VLLM_PROFILE" in
    qwen35_v1|vllm_v1|v1)
        export VLLM_USE_V1="${VLLM_USE_V1:-1}"
        export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
        VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
        VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-true}"
        VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
        VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"
        VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-true}"
        ;;
    legacy|off|none)
        echo "Error: TAU3_VLLM_PROFILE=legacy is not supported by this vLLM V1 server path."
        echo "Use the previous known-good env/image for legacy vLLM instead."
        exit 1
        ;;
    *)
        echo "Error: TAU3_VLLM_PROFILE must be qwen35_v1 or legacy"
        exit 1
        ;;
esac

ARGS=(
    --config-name=tau3_sdpo_live
    "data.train_files=$TASK_DIR/train.parquet"
    "data.val_files=$TASK_DIR/test.parquet"
    "data.train_batch_size=${TRAIN_BATCH_SIZE:-8}"
    "data.val_batch_size=${VAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-8}}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH:-16384}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH:-16384}"
    "+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING:-true}"
    "max_model_len=${MAX_MODEL_LEN:-32768}"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.actor.optim.lr=${LR:-1e-6}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS:-0}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
    "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo"
    "actor_rollout_ref.actor.policy_loss.sdpo_alpha=${SDPO_ALPHA:-1.0}"
    "actor_rollout_ref.actor.policy_loss.sdpo_is_clip=${SDPO_IS_CLIP:-2.0}"
    "actor_rollout_ref.actor.policy_loss.sdpo_loss_coef=${SDPO_LOSS_COEF:-1.0}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_BATCH_SIZE:-8}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE:-1}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-0.4}"
    "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-0.95}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.4}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.95}"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-4}"
    "algorithm.adv_estimator=grpo"
    "tau3.sdpo.max_reprompt_len=${SDPO_MAX_REPROMPT_LEN:-16384}"
    "tau3.sdpo.reprompt_truncation=${SDPO_REPROMPT_TRUNCATION:-right}"
    "trainer.project_name=${PROJECT_NAME:-SDPO-${USER}}"
    "trainer.experiment_name=$EXP_NAME"
    "trainer.total_epochs=${TOTAL_EPOCHS:-300}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-300}"
    "trainer.test_freq=${TEST_FREQ:-30}"
    "trainer.save_freq=${SAVE_FREQ:-30}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8}"
    "trainer.nnodes=${NNODES:-1}"
)

if [ "$SDPO_ARM" = "original" ]; then
    # Match the SDPO paper/code default teacher-context routing: use a successful
    # peer demonstration when one exists, otherwise fall back to rich environment
    # feedback for failed samples. This avoids the zero-target failure mode of
    # the peer-only diagnostic arm on all-fail rollout groups.
    ARGS+=("tau3.sdpo.include_environment_feedback=true")
    ARGS+=("tau3.sdpo.use_successful_peer_solution=true")
    ARGS+=("tau3.sdpo.only_failed_with_feedback=true")
    ARGS+=("tau3.sdpo.dont_reprompt_on_self_success=true")
    ARGS+=("tau3.sdpo.environment_feedback_only_without_solution=true")
    ARGS+=("tau3.sdpo.serialize_nonstring_feedback=true")
elif [ "$SDPO_ARM" = "vanilla_peer" ]; then
    # Diagnostic peer-only arm: successful peer solution demonstrations are the
    # teacher context; environment feedback is disabled. This is not the main
    # paper-style baseline for Tau3.
    ARGS+=("tau3.sdpo.include_environment_feedback=false")
    ARGS+=("tau3.sdpo.use_successful_peer_solution=true")
    ARGS+=("tau3.sdpo.only_failed_with_feedback=false")
    ARGS+=("tau3.sdpo.dont_reprompt_on_self_success=false")
    ARGS+=("tau3.sdpo.environment_feedback_only_without_solution=false")
    ARGS+=("tau3.sdpo.serialize_nonstring_feedback=false")
else
    # Diagnostic feedback-only Tau3 SDPO variant for ablation/debugging.
    ARGS+=("tau3.sdpo.include_environment_feedback=true")
    ARGS+=("tau3.sdpo.use_successful_peer_solution=false")
    ARGS+=("tau3.sdpo.only_failed_with_feedback=true")
    ARGS+=("tau3.sdpo.dont_reprompt_on_self_success=true")
    ARGS+=("tau3.sdpo.environment_feedback_only_without_solution=true")
    ARGS+=("tau3.sdpo.serialize_nonstring_feedback=true")
fi

if [ -n "${ROLLOUT_DATA_DIR:-}" ]; then
    ARGS+=("trainer.rollout_data_dir=$ROLLOUT_DATA_DIR")
fi
if [ -n "${VLLM_ENABLE_PREFIX_CACHING:-}" ]; then
    ARGS+=("actor_rollout_ref.rollout.enable_prefix_caching=$VLLM_ENABLE_PREFIX_CACHING")
fi
if [ -n "${VLLM_ENABLE_CHUNKED_PREFILL:-}" ]; then
    ARGS+=("actor_rollout_ref.rollout.enable_chunked_prefill=$VLLM_ENABLE_CHUNKED_PREFILL")
fi
if [ -n "${VLLM_MAX_NUM_BATCHED_TOKENS:-}" ]; then
    ARGS+=("actor_rollout_ref.rollout.max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS")
fi
if [ -n "${VLLM_ENFORCE_EAGER:-}" ]; then
    ARGS+=("actor_rollout_ref.rollout.enforce_eager=$VLLM_ENFORCE_EAGER")
fi
if [ -n "${VLLM_LANGUAGE_MODEL_ONLY:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=$VLLM_LANGUAGE_MODEL_ONLY")
fi
if [ -n "${VLLM_COMPILATION_CONFIG_JSON:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config=$VLLM_COMPILATION_CONFIG_JSON")
fi
if [ -n "${VLLM_BLOCK_SIZE:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=$VLLM_BLOCK_SIZE")
fi
if [ -n "${VLLM_MAMBA_BLOCK_SIZE:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_block_size=$VLLM_MAMBA_BLOCK_SIZE")
fi
if [ -n "${VLLM_MAMBA_CACHE_MODE:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=$VLLM_MAMBA_CACHE_MODE")
fi
if [ -n "${VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_hybrid_kv_cache_manager=$VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER")
fi
if [ -n "${VLLM_KV_CACHE_MEMORY_BYTES:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=$VLLM_KV_CACHE_MEMORY_BYTES")
fi
if [ -n "${VLLM_DISABLE_CASCADE_ATTN:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=$VLLM_DISABLE_CASCADE_ATTN")
fi

echo "----------------------------------------------------------------"
echo "Starting latest-VERL tau3 SDPO baseline"
echo "Experiment: $EXP_NAME"
echo "SDPO arm: $SDPO_ARM"
echo "Model: $MODEL_PATH"
echo "Model alias: $MODEL_NAME"
echo "Task dir: $TASK_DIR"
echo "Thinking: ${ENABLE_THINKING:-true}"
echo "Rollout n: ${ROLLOUT_BATCH_SIZE:-8}"
echo "Max response length: ${MAX_RESPONSE_LENGTH:-16384}"
echo "SDPO max reprompt len: ${SDPO_MAX_REPROMPT_LEN:-16384}"
echo "SDPO reprompt truncation: ${SDPO_REPROMPT_TRUNCATION:-right}"
echo "vLLM profile: $TAU3_VLLM_PROFILE"
echo "VLLM_USE_V1: ${VLLM_USE_V1:-<unset>}"
echo "vLLM prefix/chunked/max-batched/eager: ${VLLM_ENABLE_PREFIX_CACHING:-<config>}/${VLLM_ENABLE_CHUNKED_PREFILL:-<config>}/${VLLM_MAX_NUM_BATCHED_TOKENS:-<config>}/${VLLM_ENFORCE_EAGER:-<config>}"
echo "----------------------------------------------------------------"

python3 -m verl.trainer.main_ppo "${ARGS[@]}" "${@:4}"
