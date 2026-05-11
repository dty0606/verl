#!/usr/bin/env bash
# Evaluate selected original-SDPO overnight checkpoints on canonical Tau3
# airline test20. Runs the same paired grid used by the GRPO 270 baseline so
# SDPO vs GRPO comparisons stay apples-to-apples.
#
# Usage:
#   # Smoke: one checkpoint, one seed, 2 tasks — sanity-check the pipeline.
#   bash scripts/tau3/eval_sdpo_original_checkpoints_test20.sh --smoke
#
#   # Default: five checkpoints (300, 270, 210, 150, 120), 3 seeds, 20 tasks.
#   bash scripts/tau3/eval_sdpo_original_checkpoints_test20.sh
#
#   # Explicit step list (comma-separated).
#   bash scripts/tau3/eval_sdpo_original_checkpoints_test20.sh --steps 300,270,210
#
# Recommended launch (overnight):
#   nohup bash scripts/tau3/eval_sdpo_original_checkpoints_test20.sh \
#     > ~/tw/logs/eval_sdpo_multickpt_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
#   # Progress:
#   tail -f ~/tw/logs/eval_sdpo_multickpt_*.log
#   ls outputs/eval_paired/ | grep sdpo_original_r4k_b4n8
#
# Environment knobs (all optional):
#   CKPT_ROOT   override the training checkpoint root.
#   EVAL_SEEDS  override the seed grid (default: "42 123 456", smoke: "42").
#   EVAL_N      override val_n (default: 4, smoke: 1).
#   MAX_MODEL_LEN  eval context cap (default: 49152, matches GRPO 270 grid).
#                  Inherited training caps are refused — see guard below.
#   TAU3_LIVE_USER_MODEL  user model for eval (default: us.anthropic.claude-sonnet-4-6,
#                  matches GRPO 270 for paired comparability).
#   RUN_SUFFIX  appended to each run-name (default: "").
#   DRY_RUN=1   print commands without executing.
#   FORCE_MAX_MODEL_LEN=1  override the MAX_MODEL_LEN guard (use with intent).

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
CKPT_ROOT_DEFAULT="/home/sagemaker-user/tw/checkpoints/SDPO/SDPO-vllm-v1-original-sdpo-overnight/LOCAL-TAU3-SDPO-original-ema_ref-json-real_sft_step800-west_p5_original_sdpo_r4k_b4n8_overnight_300step_v1"
CKPT_ROOT="${CKPT_ROOT:-$CKPT_ROOT_DEFAULT}"
# Default includes 270 (late-stage comparison). Order is earliest→latest so
# progressive output is easier to read.
DEFAULT_STEPS="120,150,210,270,300"
SMOKE_STEPS="300"
SMOKE_TASKS="2,6"
SMOKE_SEEDS="42"
SMOKE_N="1"
# Canonical Tau3 airline test20 — same as GRPO 270 baseline.
FULL_TASK_IDS="2,6,8,13,16,18,19,22,24,25,26,29,30,31,32,35,37,44,45,48"
RUN_PREFIX="sdpo_original_r4k_b4n8"
# Canonical GRPO 270 eval grid.
EXPECTED_MAX_MODEL_LEN="49152"
DEFAULT_USER_MODEL="us.anthropic.claude-sonnet-4-6"

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
            grep '^#' "$0" | sed -n '2,30p'
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

# -----------------------------------------------------------------------------
# MAX_MODEL_LEN guard — inherited training cap (12288 for r4k_b4n8) would
# make eval look worse just from context truncation. Refuse to run unless
# the operator explicitly opts in with FORCE_MAX_MODEL_LEN=1.
# -----------------------------------------------------------------------------
if [ -n "${MAX_MODEL_LEN:-}" ] && [ "$MAX_MODEL_LEN" != "$EXPECTED_MAX_MODEL_LEN" ] && [ "${FORCE_MAX_MODEL_LEN:-0}" != "1" ]; then
    echo "ERROR: MAX_MODEL_LEN=$MAX_MODEL_LEN is inherited from the shell and does not match" >&2
    echo "       the GRPO 270 eval grid ($EXPECTED_MAX_MODEL_LEN)." >&2
    echo "       Fix: 'unset MAX_MODEL_LEN' before launch, or set FORCE_MAX_MODEL_LEN=1 if intentional." >&2
    exit 5
fi
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$EXPECTED_MAX_MODEL_LEN}"
export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-true}"
export THINKING_MODE="${THINKING_MODE:-on}"
export TAU3_LIVE_USER_MODEL="${TAU3_LIVE_USER_MODEL:-$DEFAULT_USER_MODEL}"

# -----------------------------------------------------------------------------
# Eval script sanity check — the paired launcher's default lives in the
# historical SDPO-qwen35 tree. Fail loudly if it is missing so we do not
# silently fall back to a different parser.
# -----------------------------------------------------------------------------
EVAL_SCRIPT="${EVAL_SCRIPT:-$HOME/SDPO-qwen35/scripts/eval_tau3_base_models.py}"
export EVAL_SCRIPT
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: EVAL_SCRIPT not found: $EVAL_SCRIPT" >&2
    echo "       Set EVAL_SCRIPT=<path> to the current eval_tau3_base_models.py." >&2
    exit 6
fi

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
START_TS="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "================================================================"
echo "SDPO original overnight — canonical test20 eval"
echo "Mode:         $MODE"
echo "Git SHA:      $GIT_SHA"
echo "CKPT_ROOT:    $CKPT_ROOT"
echo "EVAL_SCRIPT:  $EVAL_SCRIPT"
echo "Steps:        $STEPS"
echo "Seeds:        $EVAL_SEEDS"
echo "Val N:        $EVAL_N"
echo "Tasks:        $TEST_TASK_IDS"
echo "Max model:    $MAX_MODEL_LEN"
echo "User model:   $TAU3_LIVE_USER_MODEL"
echo "Thinking:     $THINKING_MODE"
echo "VLLM lang only: $VLLM_LANGUAGE_MODEL_ONLY"
echo "Started (UTC): $START_TS"
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
EVALUATED_RUNS=()
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
        echo "[DRY_RUN] EVAL_SEEDS='$EVAL_SEEDS' EVAL_N=$EVAL_N TEST_TASK_IDS='$TEST_TASK_IDS' MAX_MODEL_LEN=$MAX_MODEL_LEN VLLM_LANGUAGE_MODEL_ONLY=$VLLM_LANGUAGE_MODEL_ONLY THINKING_MODE=$THINKING_MODE TAU3_LIVE_USER_MODEL=$TAU3_LIVE_USER_MODEL EVAL_SCRIPT=$EVAL_SCRIPT bash scripts/tau3/eval_tau3_paired.sh $HF_DIR $RUN_NAME"
        continue
    fi

    EVAL_SEEDS="$EVAL_SEEDS" \
    EVAL_N="$EVAL_N" \
    TEST_TASK_IDS="$TEST_TASK_IDS" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    VLLM_LANGUAGE_MODEL_ONLY="$VLLM_LANGUAGE_MODEL_ONLY" \
    THINKING_MODE="$THINKING_MODE" \
    TAU3_LIVE_USER_MODEL="$TAU3_LIVE_USER_MODEL" \
    EVAL_SCRIPT="$EVAL_SCRIPT" \
    bash scripts/tau3/eval_tau3_paired.sh "$HF_DIR" "$RUN_NAME" || {
        echo "[step $STEP] EVAL FAILED" >&2
        FAILED_STEPS+=("$STEP:eval_failed")
        continue
    }

    RESULTS_PATH="outputs/eval_paired/$RUN_NAME/results.json"
    if [ -f "$RESULTS_PATH" ]; then
        echo "[step $STEP] OK → $RESULTS_PATH"
        python3 -c "import json, sys; r = json.load(open(sys.argv[1])); print(f\"  pass^1={r['pass_at_1']:.4f}  pass^2={r['pass_pow_2']:.4f}  pass^3={r['pass_pow_3']:.4f}  pass^4={r['pass_pow_4']:.4f}  tasks={r['tasks']}  rollouts={r['total_rollouts']}\")" "$RESULTS_PATH" || true
        EVALUATED_RUNS+=("$STEP:$RUN_NAME")
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

# -----------------------------------------------------------------------------
# Aggregate summary — single-file view of all evaluated checkpoints plus the
# resolved config + git SHA so the result is audit-proof later.
# -----------------------------------------------------------------------------
if [ ${#EVALUATED_RUNS[@]} -gt 0 ] && [ "${DRY_RUN:-0}" != "1" ]; then
    SUMMARY_DIR="outputs/eval_paired"
    SUMMARY_PATH="$SUMMARY_DIR/sdpo_overnight_test20_summary_$(date -u '+%Y%m%dT%H%M%SZ').json"
    mkdir -p "$SUMMARY_DIR"
    python3 - "$SUMMARY_PATH" "$GIT_SHA" "$START_TS" "$CKPT_ROOT" "$EVAL_SCRIPT" "$MAX_MODEL_LEN" "$TAU3_LIVE_USER_MODEL" "$THINKING_MODE" "$EVAL_SEEDS" "$EVAL_N" "$TEST_TASK_IDS" "${EVALUATED_RUNS[@]}" <<'PY'
import json
import sys
from pathlib import Path

(
    summary_path,
    git_sha,
    start_ts,
    ckpt_root,
    eval_script,
    max_model_len,
    user_model,
    thinking_mode,
    eval_seeds,
    eval_n,
    task_ids,
    *run_specs,
) = sys.argv[1:]

per_checkpoint = []
for spec in run_specs:
    step, run_name = spec.split(":", 1)
    results_path = Path("outputs/eval_paired") / run_name / "results.json"
    if not results_path.exists():
        per_checkpoint.append({"step": int(step), "run_name": run_name, "results_path": str(results_path), "error": "missing_results_json"})
        continue
    r = json.loads(results_path.read_text())
    per_checkpoint.append({
        "step": int(step),
        "run_name": run_name,
        "results_path": str(results_path),
        "pass_at_1": r.get("pass_at_1"),
        "pass_pow_2": r.get("pass_pow_2"),
        "pass_pow_3": r.get("pass_pow_3"),
        "pass_pow_4": r.get("pass_pow_4"),
        "per_seed_pass_at_1": r.get("per_seed_pass_at_1"),
        "per_seed_pass_pow": r.get("per_seed_pass_pow"),
        "per_task": r.get("per_task"),
        "tasks": r.get("tasks"),
        "total_rollouts": r.get("total_rollouts"),
    })

per_checkpoint.sort(key=lambda x: x.get("step", -1))

summary = {
    "git_sha": git_sha,
    "started_utc": start_ts,
    "resolved_config": {
        "ckpt_root": ckpt_root,
        "eval_script": eval_script,
        "max_model_len": int(max_model_len),
        "user_model": user_model,
        "thinking_mode": thinking_mode,
        "eval_seeds": eval_seeds.split(),
        "eval_n": int(eval_n),
        "task_ids": task_ids,
    },
    "per_checkpoint": per_checkpoint,
}
Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print()
print("================================================================")
print(f"Aggregate summary → {summary_path}")
print("================================================================")
print(f"{'step':>6s}  {'pass^1':>8s} {'pass^2':>8s} {'pass^3':>8s} {'pass^4':>8s}")
for row in per_checkpoint:
    if "error" in row:
        print(f"{row['step']:>6d}  MISSING  ({row['error']})")
    else:
        print(f"{row['step']:>6d}  {row['pass_at_1']:>8.4f} {row['pass_pow_2']:>8.4f} {row['pass_pow_3']:>8.4f} {row['pass_pow_4']:>8.4f}")
PY
fi

echo ""
echo "================================================================"
echo "SDPO test20 eval complete."
if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
    echo "Failures: ${FAILED_STEPS[*]}" >&2
    exit 4
fi
echo "All requested steps evaluated successfully."
echo "================================================================"
