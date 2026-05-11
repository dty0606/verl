#!/usr/bin/env bash
# Evaluate selected original-SDPO overnight checkpoints on canonical Tau3
# airline test20. Runs the same paired grid used by the GRPO 270 baseline so
# SDPO vs GRPO comparisons stay apples-to-apples.
#
# Usage:
#   # Smoke: one checkpoint, one seed, 2 tasks — sanity-check the pipeline.
#   bash scripts/tau3/eval_sdpo_original_checkpoints_test20.sh --smoke
#
#   # Default: four checkpoints (300, 210, 150, 120), 3 seeds, 20 tasks.
#   bash scripts/tau3/eval_sdpo_original_checkpoints_test20.sh
#
#   # Explicit step list (comma-separated).
#   bash scripts/tau3/eval_sdpo_original_checkpoints_test20.sh --steps 300,270,210
#
# Environment knobs (all optional):
#   CKPT_ROOT   override the training checkpoint root.
#   EVAL_SEEDS  override the seed grid (default: "42 123 456", smoke: "42").
#   EVAL_N      override val_n (default: 4, smoke: 1).
#   RUN_SUFFIX  appended to each run-name (default: "").
#   DRY_RUN=1   print commands without executing.

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
CKPT_ROOT_DEFAULT="/home/sagemaker-user/tw/checkpoints/SDPO/SDPO-vllm-v1-original-sdpo-overnight/LOCAL-TAU3-SDPO-original-ema_ref-json-real_sft_step800-west_p5_original_sdpo_r4k_b4n8_overnight_300step_v1"
CKPT_ROOT="${CKPT_ROOT:-$CKPT_ROOT_DEFAULT}"
DEFAULT_STEPS="300,210,150,120"
SMOKE_STEPS="300"
SMOKE_TASKS="2,6"
SMOKE_SEEDS="42"
SMOKE_N="1"
# Canonical Tau3 airline test20 — same as GRPO 270 baseline.
FULL_TASK_IDS="2,6,8,13,16,18,19,22,24,25,26,29,30,31,32,35,37,44,45,48"
RUN_PREFIX="sdpo_original_r4k_b4n8"

MODE="default"
STEPS="$DEFAULT_STEPS"
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke)
            MODE="smoke"
            STEPS="$SMOKE_STEPS"
            shift
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --steps=*)
            STEPS="${1#--steps=}"
            shift
            ;;
        -h|--help)
            grep '^#' "$0" | sed -n '2,20p'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ "$MODE" = "smoke" ]; then
    EVAL_SEEDS="${EVAL_SEEDS:-$SMOKE_SEEDS}"
    EVAL_N="${EVAL_N:-$SMOKE_N}"
    TEST_TASK_IDS="${TEST_TASK_IDS:-$SMOKE_TASKS}"
    RUN_SUFFIX="${RUN_SUFFIX:-_smoke}"
else
    EVAL_SEEDS="${EVAL_SEEDS:-42 123 456}"
    EVAL_N="${EVAL_N:-4}"
    TEST_TASK_IDS="${TEST_TASK_IDS:-$FULL_TASK_IDS}"
    RUN_SUFFIX="${RUN_SUFFIX:-}"
fi

# Evaluation grid pinned to the GRPO 270 baseline for paired comparability.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"
export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-true}"
export THINKING_MODE="${THINKING_MODE:-on}"

echo "================================================================"
echo "SDPO original overnight — canonical test20 eval"
echo "Mode:       $MODE"
echo "CKPT_ROOT:  $CKPT_ROOT"
echo "Steps:      $STEPS"
echo "Seeds:      $EVAL_SEEDS"
echo "Val N:      $EVAL_N"
echo "Tasks:      $TEST_TASK_IDS"
echo "Max model:  $MAX_MODEL_LEN  (VLLM_LANGUAGE_MODEL_ONLY=$VLLM_LANGUAGE_MODEL_ONLY, THINKING_MODE=$THINKING_MODE)"
echo "================================================================"

if [ ! -d "$CKPT_ROOT" ]; then
    echo "ERROR: CKPT_ROOT not found: $CKPT_ROOT" >&2
    exit 3
fi

# -----------------------------------------------------------------------------
# Per-step loop: resolve actor dir, merge FSDP→HF if needed, then run eval.
# -----------------------------------------------------------------------------
IFS=',' read -r -a STEP_ARRAY <<< "$STEPS"
FAILED_STEPS=()
for STEP in "${STEP_ARRAY[@]}"; do
    STEP="${STEP//[[:space:]]/}"
    [ -z "$STEP" ] && continue

    ACTOR_DIR="$CKPT_ROOT/global_step_${STEP}/actor"
    HF_DIR="$ACTOR_DIR/huggingface"
    RUN_NAME="${RUN_PREFIX}_step${STEP}_test20${RUN_SUFFIX}"

    echo ""
    echo "----------------------------------------------------------------"
    echo "[step $STEP] actor_dir=$ACTOR_DIR"
    echo "[step $STEP] run_name=$RUN_NAME"

    if [ ! -d "$ACTOR_DIR" ]; then
        echo "[step $STEP] SKIP: actor dir missing" >&2
        FAILED_STEPS+=("$STEP:missing_actor")
        continue
    fi

    # HF merge: only if weights are not already present. The GRPO 270 eval
    # recipe proved that the FSDP legacy_model_merger path is safe and
    # idempotent; this mirrors that exactly.
    if [ ! -f "$HF_DIR/config.json" ] || ! ls "$HF_DIR"/*.safetensors >/dev/null 2>&1; then
        echo "[step $STEP] merging FSDP shards → $HF_DIR"
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "[DRY_RUN] python3 scripts/legacy_model_merger.py merge --backend fsdp --local_dir $ACTOR_DIR --target_dir $HF_DIR"
        else
            if ! python3 scripts/legacy_model_merger.py merge \
                --backend fsdp \
                --local_dir "$ACTOR_DIR" \
                --target_dir "$HF_DIR"; then
                echo "[step $STEP] MERGE FAILED" >&2
                FAILED_STEPS+=("$STEP:merge_failed")
                continue
            fi
        fi
    else
        echo "[step $STEP] HF weights already present, skipping merge"
    fi

    # Hand off to the shared paired-eval launcher. It already clears pyc
    # caches, sets PYTHONPATH-first, and emits eval_config.json + results.json.
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY_RUN] EVAL_SEEDS='$EVAL_SEEDS' EVAL_N=$EVAL_N TEST_TASK_IDS='$TEST_TASK_IDS' MAX_MODEL_LEN=$MAX_MODEL_LEN VLLM_LANGUAGE_MODEL_ONLY=$VLLM_LANGUAGE_MODEL_ONLY THINKING_MODE=$THINKING_MODE bash scripts/tau3/eval_tau3_paired.sh $HF_DIR $RUN_NAME"
        continue
    fi

    EVAL_SEEDS="$EVAL_SEEDS" \
    EVAL_N="$EVAL_N" \
    TEST_TASK_IDS="$TEST_TASK_IDS" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    VLLM_LANGUAGE_MODEL_ONLY="$VLLM_LANGUAGE_MODEL_ONLY" \
    THINKING_MODE="$THINKING_MODE" \
    bash scripts/tau3/eval_tau3_paired.sh "$HF_DIR" "$RUN_NAME" || {
        echo "[step $STEP] EVAL FAILED" >&2
        FAILED_STEPS+=("$STEP:eval_failed")
        continue
    }

    RESULTS_PATH="outputs/eval_paired/$RUN_NAME/results.json"
    if [ -f "$RESULTS_PATH" ]; then
        echo "[step $STEP] OK → $RESULTS_PATH"
        python3 -c "import json, sys; r = json.load(open(sys.argv[1])); print(f\"  pass^1={r['pass_at_1']:.4f}  pass^2={r['pass_pow_2']:.4f}  pass^3={r['pass_pow_3']:.4f}  pass^4={r['pass_pow_4']:.4f}  tasks={r['tasks']}  rollouts={r['total_rollouts']}\")" "$RESULTS_PATH" || true
    else
        echo "[step $STEP] WARN: results.json missing" >&2
        FAILED_STEPS+=("$STEP:no_results")
    fi

    # Optional: run the artifact auditor if it exists. This proves parser
    # diagnostics line up with the recorded eval_config and catches bogus
    # context-length / malformed-tool-format runs before we aggregate.
    if [ -f "scripts/tau3/audit_tau3_eval_artifact.py" ]; then
        python3 scripts/tau3/audit_tau3_eval_artifact.py \
            --run-dir "outputs/eval_paired/$RUN_NAME" \
            > "outputs/eval_paired/$RUN_NAME/audit.log" 2>&1 || true
        echo "[step $STEP] audit log → outputs/eval_paired/$RUN_NAME/audit.log"
    fi
done

echo ""
echo "================================================================"
echo "SDPO test20 eval complete."
if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
    echo "Failures: ${FAILED_STEPS[*]}" >&2
    exit 4
fi
echo "All requested steps evaluated successfully."
echo "================================================================"
