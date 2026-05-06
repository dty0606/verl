#!/usr/bin/env bash
# Run staged vLLM V1 capacity smokes on P5 without Docker.
#
# This script is intentionally ordered from safest to most aggressive:
#   1. BF16/auto KV, prefix caching disabled.
#   2. BF16/auto KV, prefix caching enabled.
#   3. FP8 KV, prefix caching disabled.
#   4. FP8 KV, prefix caching enabled.
#   5. FP8 KV, prefix caching enabled, 64K model length.
#
# Use this after activating the candidate latest-VERL + vLLM V1 conda env.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
TASK_PATH="${TASK_PATH:-datasets/tau3_live_airline_canonical_json}"
TASK_DIR="$PROJECT_ROOT/${TASK_PATH#./}"
MODEL_PATH="${MODEL_PATH:-}"
MODE="${MODE:-grpo}" # grpo or sdpo
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/vllm_v1_capacity_matrix/$(date +%Y%m%d_%H%M%S)}"

if [ -z "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH must point to the SFT HF checkpoint." >&2
    exit 1
fi
if [ -z "${TAU3_LIVE_USER_MODEL:-}" ]; then
    echo "Error: TAU3_LIVE_USER_MODEL must be set for the Tau3 user simulator." >&2
    exit 1
fi
if [ "$MODE" != "grpo" ] && [ "$MODE" != "sdpo" ]; then
    echo "Error: MODE must be grpo or sdpo." >&2
    exit 1
fi
if [ ! -f "$TASK_DIR/train.parquet" ] || [ ! -f "$TASK_DIR/test.parquet" ]; then
    echo "Error: missing train.parquet or test.parquet in $TASK_DIR" >&2
    exit 1
fi

mkdir -p "$LOG_ROOT"

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export MODEL_PATH
export TASK_PATH
export TAU3_VLLM_PROFILE="${TAU3_VLLM_PROFILE:-qwen35_v1}"
export TAU3_LIVE_RUNTIME="${TAU3_LIVE_RUNTIME:-official_gym}"
export TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION="${TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION:-0}"
export TAU3_LIVE_FEEDBACK_FORMAT="${TAU3_LIVE_FEEDBACK_FORMAT:-json}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export TEST_FREQ="${TEST_FREQ:-3}"
export SAVE_FREQ="${SAVE_FREQ:-3}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export VAL_N="${VAL_N:-1}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
export MODEL_ALIAS="${MODEL_ALIAS:-real_sft_step800}"
export PROJECT_NAME="${PROJECT_NAME:-SDPO-vllm-v1-capacity}"

echo "----------------------------------------------------------------"
echo "Tau3 vLLM V1 capacity matrix"
echo "Project root: $PROJECT_ROOT"
echo "Task dir: $TASK_DIR"
echo "Model path: $MODEL_PATH"
echo "Mode: $MODE"
echo "Log root: $LOG_ROOT"
echo "VLLM_USE_V1: $VLLM_USE_V1"
echo "Tau3 all-messages observation: $TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION"
echo "----------------------------------------------------------------"

python "$PROJECT_ROOT/scripts/p5_preflight_vllm_v1.py" \
    --model-path "$MODEL_PATH" \
    --dataset-dir "$TASK_DIR" \
    2>&1 | tee "$LOG_ROOT/00_preflight.log"

run_profile() {
    local name="$1"
    local max_response="$2"
    local max_model="$3"
    local prefix="$4"
    local kv_dtype="$5"
    local calc_scales="$6"

    echo "----------------------------------------------------------------"
    echo "Profile: $name"
    echo "MAX_RESPONSE_LENGTH=$max_response"
    echo "MAX_MODEL_LEN=$max_model"
    echo "VLLM_ENABLE_PREFIX_CACHING=$prefix"
    echo "VLLM_KV_CACHE_DTYPE=$kv_dtype"
    echo "VLLM_CALCULATE_KV_SCALES=$calc_scales"
    echo "----------------------------------------------------------------"

    (
        set -euo pipefail
        cd "$PROJECT_ROOT"
        export MAX_RESPONSE_LENGTH="$max_response"
        export MAX_MODEL_LEN="$max_model"
        export VLLM_ENABLE_PREFIX_CACHING="$prefix"
        export VLLM_KV_CACHE_DTYPE="$kv_dtype"
        export VLLM_CALCULATE_KV_SCALES="$calc_scales"
        export ROLLOUT_DATA_DIR="$PROJECT_ROOT/outputs/vllm_v1_capacity_matrix/$name/rollout_data"
        mkdir -p "$ROLLOUT_DATA_DIR"
        if [ "$MODE" = "grpo" ]; then
            bash "$PROJECT_ROOT/run_local_tau3_grpo_live_p5.sh" "$TASK_PATH" "vllm_v1_${name}" json
        else
            SDPO_ARM="${SDPO_ARM:-original}" \
                bash "$PROJECT_ROOT/run_local_tau3_sdpo_live_p5.sh" "$TASK_PATH" "vllm_v1_${name}" json
        fi
    ) 2>&1 | tee "$LOG_ROOT/${name}.log"
}

run_or_record() {
    local name="$1"
    shift
    if run_profile "$name" "$@"; then
        echo "PASS $name" | tee -a "$LOG_ROOT/summary.txt"
    else
        echo "FAIL $name" | tee -a "$LOG_ROOT/summary.txt"
        if [ "$CONTINUE_ON_FAIL" != "1" ]; then
            exit 1
        fi
    fi
}

run_or_record "01_auto_no_prefix_24k_48k" 24576 49152 false auto false
run_or_record "02_auto_prefix_24k_48k" 24576 49152 true auto false
run_or_record "03_fp8_no_prefix_24k_48k" 24576 49152 false fp8 true
run_or_record "04_fp8_prefix_24k_48k" 24576 49152 true fp8 true
run_or_record "05_fp8_prefix_32k_64k" 32768 65536 true fp8 true

echo "----------------------------------------------------------------"
echo "Capacity matrix complete. Summary:"
cat "$LOG_ROOT/summary.txt"
echo "Logs: $LOG_ROOT"
echo "----------------------------------------------------------------"
