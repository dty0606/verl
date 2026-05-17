#!/usr/bin/env bash
# SDPO baseline for tau3 live on latest VERL.
#
# SDPO_ARM controls the teacher-context variant:
#   peer_only -> faithful/minimal Tau3 SDPO: successful peer only, no
#                heuristic environment-feedback injection and no memory cards.
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

export TAU3_LIVE_RUNTIME="${TAU3_LIVE_RUNTIME:-official_gym}"
if [ "$TAU3_LIVE_RUNTIME" != "official_gym" ]; then
    echo "Error: faithful Tau3 SDPO baseline requires TAU3_LIVE_RUNTIME=official_gym."
    echo "Proxy/legacy runtimes are not allowed for claim-bearing SDPO runs."
    exit 1
fi
export TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION="${TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION:-0}"
export TAU3_BEDROCK_MAX_RETRIES="${TAU3_BEDROCK_MAX_RETRIES:-3}"
export TAU3_BEDROCK_RETRY_DELAYS="${TAU3_BEDROCK_RETRY_DELAYS:-15,30,60}"
export TAU3_BEDROCK_RETRY_JITTER="${TAU3_BEDROCK_RETRY_JITTER:-0.2}"
export TAU3_MASK_ENV_ERROR_ROLLOUTS="${TAU3_MASK_ENV_ERROR_ROLLOUTS:-1}"
export TAU3_ENV_ERROR_SKIP_UPDATE_THRESHOLD="${TAU3_ENV_ERROR_SKIP_UPDATE_THRESHOLD:-0.25}"
export TAU3_RETRY_STEP_ON_TRANSIENT="${TAU3_RETRY_STEP_ON_TRANSIENT:-0}"
export TAU3_MARK_ENV_EXCEPTIONS="${TAU3_MARK_ENV_EXCEPTIONS:-false}"
TAU3_MARK_ENV_EXCEPTIONS_NORMALIZED="$(printf '%s' "$TAU3_MARK_ENV_EXCEPTIONS" | tr '[:upper:]' '[:lower:]')"
if [ "$TAU3_MARK_ENV_EXCEPTIONS_NORMALIZED" != "false" ] && [ "$TAU3_MARK_ENV_EXCEPTIONS_NORMALIZED" != "0" ]; then
    echo "Error: faithful Tau3 SDPO baseline requires TAU3_MARK_ENV_EXCEPTIONS=false."
    echo "Only classified Tau3/Bedrock env errors may be masked; unknown env exceptions should crash."
    exit 1
fi
if [ -z "${TAU3_LIVE_USER_ARGS_JSON:-}" ]; then
    export TAU3_LIVE_USER_ARGS_JSON='{"num_retries":8,"timeout":120}'
fi
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"
export STORAGE_ROOT="${STORAGE_ROOT:-$HOME/tw}"
mkdir -p "$STORAGE_ROOT"
SDPO_MIN_FREE_GB="${SDPO_MIN_FREE_GB:-500}"
if [ "$SDPO_MIN_FREE_GB" -gt 0 ]; then
    FREE_GB=$(df -BG "$STORAGE_ROOT" | awk 'NR==2 {gsub("G","",$4); print $4}')
    if [ "${FREE_GB:-0}" -lt "$SDPO_MIN_FREE_GB" ]; then
        echo "ABORT: less than ${SDPO_MIN_FREE_GB}GB free under $STORAGE_ROOT"
        exit 1
    fi
fi
export SDPO_OUTPUT_ROOT="${SDPO_OUTPUT_ROOT:-$STORAGE_ROOT/output/SDPO}"
export SDPO_CHECKPOINT_ROOT="${SDPO_CHECKPOINT_ROOT:-$STORAGE_ROOT/checkpoints/SDPO}"
export WANDB_DIR="${WANDB_DIR:-$STORAGE_ROOT/wandb}"
export TMPDIR="${TMPDIR:-$STORAGE_ROOT/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-$STORAGE_ROOT/ray_tmp}"
mkdir -p "$SDPO_OUTPUT_ROOT" "$SDPO_CHECKPOINT_ROOT" "$WANDB_DIR" "$TMPDIR" "$RAY_TMPDIR" "$STORAGE_ROOT/logs"

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
SDPO_ARM="${SDPO_ARM:-peer_only}"
case "$SDPO_ARM" in
    peer_only|faithful|minimal|vanilla_peer|peer|successful_peer) SDPO_ARM="peer_only" ;;
    original|paper|paper_hybrid|hybrid|feedback_hybrid|feedback|feedback_only|ff_sdpo|vanilla|vanilla_sdpo)
        echo "Error: SDPO_ARM=$SDPO_ARM is not supported by the guarded faithful baseline."
        echo "Use SDPO_ARM=peer_only here; run feedback/memory ablations on a separate branch."
        exit 1
        ;;
    *) echo "Error: SDPO_ARM must be peer_only"; exit 1 ;;
esac
RAW_FEEDBACK_MODE="${3:-${TAU3_LIVE_FEEDBACK_FORMAT:-}}"
if [ -z "$RAW_FEEDBACK_MODE" ]; then
    RAW_FEEDBACK_MODE="none"
fi
case "$RAW_FEEDBACK_MODE" in
    structured|json) FEEDBACK_MODE="json" ;;
    text|plain|plain_text) FEEDBACK_MODE="plain_text" ;;
    none|off|false|0) FEEDBACK_MODE="none" ;;
    *) echo "Error: feedback_mode must be json, plain_text, or none"; exit 1 ;;
esac
export TAU3_LIVE_FEEDBACK_FORMAT="$FEEDBACK_MODE"
if [ "$FEEDBACK_MODE" != "none" ]; then
    echo "Error: guarded faithful Tau3 SDPO does not allow heuristic teacher feedback."
    echo "Set feedback_mode=none; diagnostic-feedback ablations belong on a separate branch."
    exit 1
fi
SDPO_MEMORY_ENABLED_NORMALIZED="$(printf '%s' "${SDPO_MEMORY_ENABLED:-false}" | tr '[:upper:]' '[:lower:]')"
if [ "$SDPO_MEMORY_ENABLED_NORMALIZED" != "false" ] && [ "$SDPO_MEMORY_ENABLED_NORMALIZED" != "0" ]; then
    echo "Error: guarded faithful Tau3 SDPO does not allow SDPO memory/note cards."
    echo "Use the Note-SDPO branch for memory experiments."
    exit 1
fi
SDPO_TARGET_GUARD_ENABLED_NORMALIZED="$(printf '%s' "${SDPO_TARGET_GUARD_ENABLED:-false}" | tr '[:upper:]' '[:lower:]')"
if [ "$SDPO_TARGET_GUARD_ENABLED_NORMALIZED" != "false" ] && [ "$SDPO_TARGET_GUARD_ENABLED_NORMALIZED" != "0" ]; then
    echo "Error: faithful Tau3 SDPO baseline keeps tau3.sdpo.target_guard disabled."
    echo "Run target-guard ablations separately; this launcher keeps only runtime/tensor safety checks."
    exit 1
fi
SDPO_TEACHER_BACKEND="${SDPO_TEACHER_BACKEND:-ema_ref}"
case "$SDPO_TEACHER_BACKEND" in
    ema|ema_ref|paper_ema) SDPO_TEACHER_BACKEND="ema_ref" ;;
    actor|actor_snapshot|current_actor) SDPO_TEACHER_BACKEND="actor_snapshot" ;;
    *) echo "Error: SDPO_TEACHER_BACKEND must be ema_ref or actor_snapshot"; exit 1 ;;
esac
EXP_NAME="LOCAL-TAU3-SDPO-${SDPO_ARM}-${SDPO_TEACHER_BACKEND}-${FEEDBACK_MODE}-${MODEL_NAME}-${SUFFIX}"
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
    "data.train_batch_size=${TRAIN_BATCH_SIZE:-4}"
    "data.val_batch_size=${VAL_BATCH_SIZE:-4}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH:-8192}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH:-6144}"
    "+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING:-true}"
    "max_model_len=${MAX_MODEL_LEN:-14336}"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    "actor_rollout_ref.actor.optim.lr=${LR:-1e-6}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS:-0}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
    "actor_rollout_ref.actor.use_dynamic_bsz=${ACTOR_USE_DYNAMIC_BSZ:-${LOG_PROB_USE_DYNAMIC_BSZ:-false}}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-${MAX_MODEL_LEN:-14336}}"
    "actor_rollout_ref.actor.freeze_vision_tower=${FREEZE_VISION_TOWER:-true}"
    "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo"
    "actor_rollout_ref.actor.policy_loss.sdpo_full_logit_distillation=${SDPO_FULL_LOGIT_DISTILLATION:-true}"
    "actor_rollout_ref.actor.policy_loss.sdpo_alpha=${SDPO_ALPHA:-0.5}"
    "actor_rollout_ref.actor.policy_loss.sdpo_distillation_topk=${SDPO_DISTILLATION_TOPK:-100}"
    "actor_rollout_ref.actor.policy_loss.sdpo_distillation_add_tail=${SDPO_DISTILLATION_ADD_TAIL:-true}"
    "actor_rollout_ref.actor.policy_loss.sdpo_topk_source=${SDPO_TOPK_SOURCE:-student_pre_update}"
    "actor_rollout_ref.actor.policy_loss.sdpo_is_clip=${SDPO_IS_CLIP:-2.0}"
    "actor_rollout_ref.actor.policy_loss.sdpo_loss_coef=${SDPO_LOSS_COEF:-1.0}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_BATCH_SIZE:-8}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE:-1}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
    "actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-0.4}"
    "actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-0.95}"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ:-${LOG_PROB_USE_DYNAMIC_BSZ:-false}}"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${MAX_MODEL_LEN:-14336}}}"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${REF_LOG_PROB_USE_DYNAMIC_BSZ:-${LOG_PROB_USE_DYNAMIC_BSZ:-false}}"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${MAX_MODEL_LEN:-14336}}}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.4}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.95}"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-4}"
    "algorithm.adv_estimator=grpo"
    "tau3.sdpo.max_reprompt_len=${SDPO_MAX_REPROMPT_LEN:-6144}"
    "tau3.sdpo.reprompt_truncation=${SDPO_REPROMPT_TRUNCATION:-right}"
    "tau3.sdpo.teacher_backend=${SDPO_TEACHER_BACKEND:-ema_ref}"
    "tau3.sdpo.teacher_update_rate=${SDPO_TEACHER_UPDATE_RATE:-0.05}"
    "tau3.sdpo.gt_metadata_enabled=${SDPO_GT_METADATA_ENABLED:-false}"
    "tau3.sdpo.failed_peer_enabled=${SDPO_FAILED_PEER_ENABLED:-false}"
    "tau3.sdpo.failed_peer_max_chars=${SDPO_FAILED_PEER_MAX_CHARS:-4096}"
    "tau3.sdpo.target_guard.enabled=${SDPO_TARGET_GUARD_ENABLED:-false}"
    "tau3.sdpo.target_guard.corrupted_row_weight=${SDPO_TARGET_GUARD_CORRUPTED_ROW_WEIGHT:-0.0}"
    "tau3.sdpo.target_guard.mask_nonterminal=${SDPO_TARGET_GUARD_MASK_NONTERMINAL:-true}"
    "tau3.sdpo.target_guard.mask_budget_exhausted=${SDPO_TARGET_GUARD_MASK_BUDGET_EXHAUSTED:-true}"
    "tau3.sdpo.target_guard.mask_response_saturated=${SDPO_TARGET_GUARD_MASK_RESPONSE_SATURATED:-true}"
    "tau3.sdpo.target_guard.mask_parse_error=${SDPO_TARGET_GUARD_MASK_PARSE_ERROR:-true}"
    "tau3.sdpo.target_guard.mask_open_think=${SDPO_TARGET_GUARD_MASK_OPEN_THINK:-true}"
    "tau3.sdpo.target_guard.mask_repetition=${SDPO_TARGET_GUARD_MASK_REPETITION:-true}"
    "tau3.sdpo.target_guard.mask_tool_loop=${SDPO_TARGET_GUARD_MASK_TOOL_LOOP:-true}"
    "tau3.sdpo.target_guard.max_response_tokens=${SDPO_TARGET_GUARD_MAX_RESPONSE_TOKENS:-${MAX_RESPONSE_LENGTH:-6144}}"
    "tau3.sdpo.target_guard.repetition_ngram_size=${SDPO_TARGET_GUARD_REPETITION_NGRAM_SIZE:-8}"
    "tau3.sdpo.target_guard.repetition_max_count=${SDPO_TARGET_GUARD_REPETITION_MAX_COUNT:-4}"
    "tau3.sdpo.target_guard.max_tool_count=${SDPO_TARGET_GUARD_MAX_TOOL_COUNT:-32}"
    "trainer.project_name=${PROJECT_NAME:-SDPO-${USER}}"
    "trainer.experiment_name=$EXP_NAME"
    "trainer.total_epochs=${TOTAL_EPOCHS:-30}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-240}"
    "trainer.test_freq=${TEST_FREQ:-30}"
    "trainer.save_freq=${SAVE_FREQ:-30}"
    "trainer.resume_mode=${RESUME_MODE:-auto}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8}"
    "trainer.nnodes=${NNODES:-1}"
    "trainer.val_before_train=false"
)

# Faithful/minimal Tau3 SDPO baseline: successful same-UID peer
# demonstrations only. No heuristic Tau3 diagnostic feedback and no memory.
ARGS+=("tau3.sdpo.use_successful_peer_solution=true")
ARGS+=("tau3.sdpo.only_failed_rollouts=true")
ARGS+=("tau3.sdpo.dont_reprompt_on_self_success=true")

if [ -n "${ROLLOUT_DATA_DIR:-}" ]; then
    ARGS+=("trainer.rollout_data_dir=$ROLLOUT_DATA_DIR")
fi
if [ -n "${RESUME_FROM_PATH:-}" ]; then
    ARGS+=("trainer.resume_from_path=$RESUME_FROM_PATH")
fi
if [ -n "${MAX_ACTOR_CKPT_TO_KEEP:-}" ]; then
    ARGS+=("trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP")
fi
if [ -n "${MAX_CRITIC_CKPT_TO_KEEP:-}" ]; then
    ARGS+=("trainer.max_critic_ckpt_to_keep=$MAX_CRITIC_CKPT_TO_KEEP")
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
if [ -n "${VLLM_KV_CACHE_DTYPE:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=$VLLM_KV_CACHE_DTYPE")
fi
if [ -n "${VLLM_CALCULATE_KV_SCALES:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.calculate_kv_scales=$VLLM_CALCULATE_KV_SCALES")
fi
if [ -n "${VLLM_DISABLE_CASCADE_ATTN:-}" ]; then
    ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=$VLLM_DISABLE_CASCADE_ATTN")
fi

for extra_arg in "${@:4}"; do
    case "$extra_arg" in
        tau3.sdpo.reprompt_template=*|tau3.sdpo.solution_template=*|tau3.sdpo.include_environment_feedback=*|tau3.sdpo.memory.*|+tau3.sdpo.memory.*|tau3.sdpo.target_guard.enabled=*|+tau3.sdpo.target_guard.enabled=*)
            echo "Error: faithful Tau3 SDPO launcher forbids override: $extra_arg" >&2
            echo "Use a separate ablation branch for feedback, memory, custom teacher templates, or target guards." >&2
            exit 1
            ;;
    esac
done

echo "----------------------------------------------------------------"
echo "Starting latest-VERL tau3 SDPO baseline"
echo "Experiment: $EXP_NAME"
echo "SDPO arm: $SDPO_ARM"
echo "Model: $MODEL_PATH"
echo "Model alias: $MODEL_NAME"
echo "Task dir: $TASK_DIR"
echo "Thinking: ${ENABLE_THINKING:-true}"
echo "Rollout n: ${ROLLOUT_BATCH_SIZE:-8}"
echo "Max response length: ${MAX_RESPONSE_LENGTH:-6144}"
echo "SDPO max reprompt len: ${SDPO_MAX_REPROMPT_LEN:-6144}"
echo "SDPO reprompt truncation: ${SDPO_REPROMPT_TRUNCATION:-right}"
echo "SDPO teacher: backend=${SDPO_TEACHER_BACKEND:-ema_ref} ema_rate=${SDPO_TEACHER_UPDATE_RATE:-0.05}"
echo "SDPO v3.5 teacher context: gt_metadata=${SDPO_GT_METADATA_ENABLED:-false} failed_peer=${SDPO_FAILED_PEER_ENABLED:-false} failed_peer_max_chars=${SDPO_FAILED_PEER_MAX_CHARS:-4096}"
echo "SDPO loss: full_logit=${SDPO_FULL_LOGIT_DISTILLATION:-true} alpha=${SDPO_ALPHA:-0.5} topk=${SDPO_DISTILLATION_TOPK:-100} add_tail=${SDPO_DISTILLATION_ADD_TAIL:-true} topk_source=${SDPO_TOPK_SOURCE:-student_pre_update}"
echo "SDPO logprob dynamic bsz: actor=${ACTOR_USE_DYNAMIC_BSZ:-${LOG_PROB_USE_DYNAMIC_BSZ:-false}} rollout=${ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ:-${LOG_PROB_USE_DYNAMIC_BSZ:-false}} ref=${REF_LOG_PROB_USE_DYNAMIC_BSZ:-${LOG_PROB_USE_DYNAMIC_BSZ:-false}} token_caps actor=${PPO_MAX_TOKEN_LEN_PER_GPU:-${MAX_MODEL_LEN:-14336}} rollout=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${MAX_MODEL_LEN:-14336}}} ref=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${MAX_MODEL_LEN:-14336}}}"
echo "SDPO fail-fast finite probes: fail_fast=${SDPO_FAIL_FAST_NONFINITE:-0} ema=${SDPO_EMA_FINITE_CHECK:-${SDPO_FAIL_FAST_NONFINITE:-0}}"
echo "SDPO logprob diagnostics: ${SDPO_LOGPROB_DIAGNOSTICS:-0}"
echo "SDPO CUDA memory diagnostics: ${SDPO_CUDA_MEMORY_DIAGNOSTICS:-${SDPO_LOGPROB_DIAGNOSTICS:-0}}"
echo "SDPO target guard: enabled=${SDPO_TARGET_GUARD_ENABLED:-false} corrupted_row_weight=${SDPO_TARGET_GUARD_CORRUPTED_ROW_WEIGHT:-0.0} parse_error=${SDPO_TARGET_GUARD_MASK_PARSE_ERROR:-true} open_think=${SDPO_TARGET_GUARD_MASK_OPEN_THINK:-true} repetition=${SDPO_TARGET_GUARD_MASK_REPETITION:-true}"
echo "SDPO teacher context: successful_peer=true heuristic_feedback=false memory=false gt_metadata=${SDPO_GT_METADATA_ENABLED:-false} failed_peer=${SDPO_FAILED_PEER_ENABLED:-false}"
echo "Storage roots: storage=$STORAGE_ROOT checkpoints=$SDPO_CHECKPOINT_ROOT output=$SDPO_OUTPUT_ROOT wandb=$WANDB_DIR tmp=$TMPDIR ray_tmp=$RAY_TMPDIR free_gb=${FREE_GB:-unknown}"
echo "Trainer resume: mode=${RESUME_MODE:-auto} from=${RESUME_FROM_PATH:-<auto/latest>}"
echo "vLLM profile: $TAU3_VLLM_PROFILE"
echo "VLLM_USE_V1: ${VLLM_USE_V1:-<unset>}"
echo "vLLM prefix/chunked/max-batched/eager: ${VLLM_ENABLE_PREFIX_CACHING:-<config>}/${VLLM_ENABLE_CHUNKED_PREFILL:-<config>}/${VLLM_MAX_NUM_BATCHED_TOKENS:-<config>}/${VLLM_ENFORCE_EAGER:-<config>}"
echo "vLLM KV dtype/scales: ${VLLM_KV_CACHE_DTYPE:-auto}/${VLLM_CALCULATE_KV_SCALES:-<config>}"
echo "Tau3 all-messages observation: $TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION"
echo "Diagnostic env snapshot:"
env | grep -E '^(SDPO|TAU3|VLLM|WANDB|RAY|CUDA|NCCL|PYTHONHASHSEED|CUBLAS_WORKSPACE_CONFIG|TORCH_DISABLE_ADDR2LINE|HYDRA_FULL_ERROR)=' | sort || true
echo "Command:"
printf '%q ' python3 -m verl.trainer.main_ppo "${ARGS[@]}" "${@:4}"
echo
echo "----------------------------------------------------------------"

python3 -m verl.trainer.main_ppo "${ARGS[@]}" "${@:4}"
