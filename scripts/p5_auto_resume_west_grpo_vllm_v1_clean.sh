#!/usr/bin/env bash
# Auto-resume the West-P5 clean GRPO vLLM V1 run after transient vLLM/Ray crashes.
#
# This is a supervisor around p5_resume_west_grpo_vllm_v1_clean_from_step60.sh.
# It does not change the GRPO/VERL config. It repeatedly finds the newest valid
# global_step_* checkpoint, launches the same resume path, and retries only for
# known transient runtime failures such as vLLM EngineDeadError/sample_tokens
# timeouts.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
cd "$PROJECT_ROOT"

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Error: activate the sdpo-vllm20-v1 conda env before running this helper." >&2
    exit 1
fi

# West P5 has persistent multi-TB storage under /home/sagemaker-user. Keep the
# default there; callers can still override NVME_ROOT explicitly.
export NVME_ROOT="${NVME_ROOT:-$HOME/tw}"
export PROJECT_NAME="${PROJECT_NAME:-SDPO-vllm-v1-clean-grpo-v2}"
export RUN_STEM="${RUN_STEM:-grpo_vllm_v1_clean_300_v2}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-$RUN_STEM}"
export OLD_PROJECT_NAME="${OLD_PROJECT_NAME:-$PROJECT_NAME}"
export OLD_EXPERIMENT_NAME="${OLD_EXPERIMENT_NAME:-LOCAL-TAU3-GRPO-json-real_sft_step800-grpo_vllm_v1_clean_300_v2_02_auto_prefix_24k_48k}"
AUTO_RESUME_EXPERIMENT_GLOB="${AUTO_RESUME_EXPERIMENT_GLOB:-$OLD_EXPERIMENT_NAME}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-$TOTAL_TRAINING_STEPS}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"

AUTO_RESUME_MAX_ATTEMPTS="${AUTO_RESUME_MAX_ATTEMPTS:-4}"
AUTO_RESUME_SLEEP_SECONDS="${AUTO_RESUME_SLEEP_SECONDS:-30}"
AUTO_RESUME_CLEANUP="${AUTO_RESUME_CLEANUP:-1}"
AUTO_RESUME_RETRY_ANY_FAILURE="${AUTO_RESUME_RETRY_ANY_FAILURE:-0}"
AUTO_RESUME_NAME="${AUTO_RESUME_NAME:-grpo_vllm_v1_clean_300_v2_auto_resume}"
AUTO_RESUME_LOG_ROOT="${AUTO_RESUME_LOG_ROOT:-$NVME_ROOT/logs/$AUTO_RESUME_NAME}"
mkdir -p "$AUTO_RESUME_LOG_ROOT" "$NVME_ROOT"/{tmp,ray_tmp,logs,outputs,output,checkpoints,wandb,cache}

checkpoint_roots=(
    "$PROJECT_ROOT/checkpoints/SDPO"
    "$NVME_ROOT/checkpoints/SDPO"
)
if [ -n "${AUTO_RESUME_CHECKPOINT_ROOTS:-}" ]; then
    IFS=':' read -r -a checkpoint_roots <<< "$AUTO_RESUME_CHECKPOINT_ROOTS"
fi

checkpoint_step() {
    local path="$1"
    basename "$path" | sed -nE 's/^global_step_([0-9]+)$/\1/p'
}

is_valid_checkpoint() {
    local path="$1"
    [ -d "$path" ] && [ -f "$path/data.pt" ] && [ -d "$path/actor" ]
}

find_latest_checkpoint() {
    local root path step best_step=-1 best_path=""
    for root in "${checkpoint_roots[@]}"; do
        [ -d "$root" ] || continue
        while IFS= read -r -d '' path; do
            is_valid_checkpoint "$path" || continue
            step="$(checkpoint_step "$path")"
            [ -n "$step" ] || continue
            if [ "$step" -gt "$best_step" ]; then
                best_step="$step"
                best_path="$path"
            fi
        done < <(find "$root" -path "*/${OLD_PROJECT_NAME}/${AUTO_RESUME_EXPERIMENT_GLOB}/global_step_*" -type d -print0 2>/dev/null)
    done
    if [ -n "$best_path" ]; then
        printf '%s\n' "$best_path"
    fi
}

latest_step_or_zero() {
    local path="${1:-}"
    local step=""
    if [ -n "$path" ]; then
        step="$(checkpoint_step "$path")"
    fi
    printf '%s\n' "${step:-0}"
}

cleanup_after_failure() {
    if [ "$AUTO_RESUME_CLEANUP" != "1" ]; then
        return 0
    fi
    echo "Cleaning stale Ray/vLLM/VERL runtime after failed attempt..."
    ray stop --force >/dev/null 2>&1 || true
    pkill -9 -f "verl.trainer.main_ppo|vllm|ray::|raylet" >/dev/null 2>&1 || true
    rm -rf /tmp/ray /tmp/tmpxft_* /tmp/torchinductor_* 2>/dev/null || true
    rm -f /dev/shm/verl_dist_store_* 2>/dev/null || true
}

is_retryable_failure() {
    local log_file="$1"
    if [ "$AUTO_RESUME_RETRY_ANY_FAILURE" = "1" ]; then
        return 0
    fi
    if grep -Eqi \
        "EngineDeadError|RPC call to sample_tokens timed out|EngineCore encountered a fatal error|RayTaskError\\(EngineDeadError\\)" \
        "$log_file"; then
        return 0
    fi
    return 1
}

echo "----------------------------------------------------------------"
echo "West P5 clean-GRPO auto-resume supervisor"
echo "Project root: $PROJECT_ROOT"
echo "NVME_ROOT: $NVME_ROOT"
echo "Project: $PROJECT_NAME"
echo "Experiment search: $OLD_PROJECT_NAME / $AUTO_RESUME_EXPERIMENT_GLOB / global_step_*"
echo "Run stem: $RUN_STEM"
echo "Target steps: $TOTAL_TRAINING_STEPS"
echo "Max attempts: $AUTO_RESUME_MAX_ATTEMPTS"
echo "vLLM RPC timeout: $VLLM_RPC_TIMEOUT"
echo "Supervisor logs: $AUTO_RESUME_LOG_ROOT"
echo "Checkpoint roots: ${checkpoint_roots[*]}"
echo "----------------------------------------------------------------"
df -h / /home/sagemaker-user /mnt/sagemaker-nvme 2>/dev/null || true

attempt=1
while [ "$attempt" -le "$AUTO_RESUME_MAX_ATTEMPTS" ]; do
    latest_ckpt="$(find_latest_checkpoint || true)"
    latest_step="$(latest_step_or_zero "$latest_ckpt")"

    if [ -z "$latest_ckpt" ]; then
        echo "Error: no valid global_step_* checkpoint found under: ${checkpoint_roots[*]}" >&2
        exit 1
    fi

    if [ "$latest_step" -ge "$TOTAL_TRAINING_STEPS" ]; then
        echo "Target already reached: latest checkpoint is global_step_${latest_step}."
        exit 0
    fi

    export RESUME_FROM_PATH="$latest_ckpt"
    export LOG_ROOT="$AUTO_RESUME_LOG_ROOT/attempt_${attempt}_from_step_${latest_step}"
    export ROLLOUT_OUTPUT_ROOT="$NVME_ROOT/outputs/${AUTO_RESUME_NAME}/attempt_${attempt}_from_step_${latest_step}"
    mkdir -p "$LOG_ROOT" "$ROLLOUT_OUTPUT_ROOT"

    attempt_log="$AUTO_RESUME_LOG_ROOT/attempt_${attempt}_from_step_${latest_step}.console.log"
    echo "----------------------------------------------------------------" | tee -a "$AUTO_RESUME_LOG_ROOT/supervisor.log"
    echo "Attempt $attempt/$AUTO_RESUME_MAX_ATTEMPTS from $RESUME_FROM_PATH" | tee -a "$AUTO_RESUME_LOG_ROOT/supervisor.log"
    echo "Attempt log: $attempt_log" | tee -a "$AUTO_RESUME_LOG_ROOT/supervisor.log"
    echo "----------------------------------------------------------------" | tee -a "$AUTO_RESUME_LOG_ROOT/supervisor.log"

    set +e
    bash "$PROJECT_ROOT/scripts/p5_resume_west_grpo_vllm_v1_clean_from_step60.sh" 2>&1 | tee "$attempt_log"
    exit_code=${PIPESTATUS[0]}
    set -e

    latest_after="$(find_latest_checkpoint || true)"
    latest_after_step="$(latest_step_or_zero "$latest_after")"
    echo "Attempt $attempt exit_code=$exit_code latest_step_after=$latest_after_step" | tee -a "$AUTO_RESUME_LOG_ROOT/supervisor.log"

    if [ "$latest_after_step" -ge "$TOTAL_TRAINING_STEPS" ]; then
        echo "Auto-resume completed: reached global_step_${latest_after_step}." | tee -a "$AUTO_RESUME_LOG_ROOT/supervisor.log"
        exit 0
    fi

    if [ "$exit_code" -eq 0 ]; then
        echo "Run exited cleanly before target; rechecking after ${AUTO_RESUME_SLEEP_SECONDS}s." | tee -a "$AUTO_RESUME_LOG_ROOT/supervisor.log"
        sleep "$AUTO_RESUME_SLEEP_SECONDS"
        attempt=$((attempt + 1))
        continue
    fi

    if ! is_retryable_failure "$attempt_log"; then
        echo "Non-retryable failure. Stop and inspect: $attempt_log" >&2
        exit "$exit_code"
    fi

    echo "Retryable vLLM/Ray transient detected. Will resume from newest checkpoint after cleanup." | tee -a "$AUTO_RESUME_LOG_ROOT/supervisor.log"
    cleanup_after_failure
    sleep "$AUTO_RESUME_SLEEP_SECONDS"
    attempt=$((attempt + 1))
done

latest_ckpt="$(find_latest_checkpoint || true)"
latest_step="$(latest_step_or_zero "$latest_ckpt")"
echo "Auto-resume exhausted $AUTO_RESUME_MAX_ATTEMPTS attempts; latest_step=$latest_step." >&2
exit 1
