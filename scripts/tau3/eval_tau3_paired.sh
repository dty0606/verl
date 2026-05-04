#!/usr/bin/env bash
# Paired tau3 airline eval — same task × trial grid for fair model comparison.
#
# Usage:
#   bash scripts/tau3/eval_tau3_paired.sh <model_path> <run_name>
#
# Examples:
#   # Base model
#   bash scripts/tau3/eval_tau3_paired.sh Qwen/Qwen3.5-4B base_qwen35_4b
#
#   # SFT checkpoint
#   bash scripts/tau3/eval_tau3_paired.sh checkpoints/.../huggingface sft_5k_balanced
#
# All models are evaluated with identical:
#   - 20 canonical test tasks
#   - Same seeds (EVAL_SEEDS, default: 42 123 456)
#   - Same val_n (EVAL_N, default: 4)
#   - Same user model, temperature, max_steps, parser, runtime
#
# Output: outputs/eval_paired/<run_name>/seed*/trajectories/*.json
#         outputs/eval_paired/<run_name>/results.json

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model_path> <run_name>"
    exit 1
fi

MODEL_PATH="$1"
RUN_NAME="$2"

# --- Eval grid (same for ALL models) ---
EVAL_SEEDS="${EVAL_SEEDS:-42 123 456}"
EVAL_N="${EVAL_N:-4}"
TEST_TASK_IDS="${TEST_TASK_IDS:-2,6,8,13,16,18,19,22,24,25,26,29,30,31,32,35,37,44,45,48}"
MAX_STEPS="${MAX_STEPS:-50}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.95}"
THINKING_MODE="${THINKING_MODE:-on}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
USER_MODEL="${TAU3_LIVE_USER_MODEL:-us.anthropic.claude-sonnet-4-6}"
NUM_GPUS="${NUM_GPUS:-8}"
TASK_SPLIT="${TASK_SPLIT:-test}"

# --- Paths ---
EVAL_SCRIPT="${EVAL_SCRIPT:-$HOME/SDPO-qwen35/scripts/eval_tau3_base_models.py}"
OUTPUT_ROOT="outputs/eval_paired/${RUN_NAME}"
LOG_DIR="logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

# --- vLLM flags ---
export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-true}"

# --- Task split across GPUs ---
# Split TEST_TASK_IDS evenly and contiguously across available GPUs. This keeps
# the recorded eval grid and the actual launched task grid identical.
IFS=',' read -r -a TASK_ID_ARRAY <<< "$TEST_TASK_IDS"
TASK_COUNT=${#TASK_ID_ARRAY[@]}
if [ "$TASK_COUNT" -eq 0 ]; then
    echo "No TEST_TASK_IDS provided."
    exit 1
fi
ACTIVE_GROUPS=$NUM_GPUS
if [ "$TASK_COUNT" -lt "$ACTIVE_GROUPS" ]; then
    ACTIVE_GROUPS=$TASK_COUNT
fi
BASE_GROUP_SIZE=$((TASK_COUNT / ACTIVE_GROUPS))
EXTRA_TASKS=$((TASK_COUNT % ACTIVE_GROUPS))
TASK_GROUPS=()
TASK_OFFSET=0
for gpu in $(seq 0 $((ACTIVE_GROUPS - 1))); do
    GROUP_SIZE=$BASE_GROUP_SIZE
    if [ "$gpu" -lt "$EXTRA_TASKS" ]; then
        GROUP_SIZE=$((GROUP_SIZE + 1))
    fi
    GROUP_TASKS=()
    for idx in $(seq 0 $((GROUP_SIZE - 1))); do
        TASK_ID="${TASK_ID_ARRAY[$((TASK_OFFSET + idx))]}"
        TASK_ID="${TASK_ID//[[:space:]]/}"
        if [ -n "$TASK_ID" ]; then
            GROUP_TASKS+=("$TASK_ID")
        fi
    done
    TASK_GROUPS+=("$(IFS=','; echo "${GROUP_TASKS[*]}")")
    TASK_OFFSET=$((TASK_OFFSET + GROUP_SIZE))
done
EXPECTED_TRAJ_COUNT=$((TASK_COUNT * EVAL_N))

# --- Record eval config ---
cat > "${OUTPUT_ROOT}/eval_config.json" <<EOF
{
  "model_path": "$MODEL_PATH",
  "run_name": "$RUN_NAME",
  "eval_seeds": "$(echo $EVAL_SEEDS)",
  "eval_n": $EVAL_N,
  "test_task_ids": "$TEST_TASK_IDS",
  "task_split": "$TASK_SPLIT",
  "max_steps": $MAX_STEPS,
  "temperature": $TEMPERATURE,
  "top_p": $TOP_P,
  "thinking_mode": "$THINKING_MODE",
  "max_model_len": $MAX_MODEL_LEN,
  "user_model": "$USER_MODEL",
  "vllm_language_model_only": "$VLLM_LANGUAGE_MODEL_ONLY",
  "eval_script": "$EVAL_SCRIPT",
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF

echo "================================================================"
echo "Paired Tau3 Eval: $RUN_NAME"
echo "Model: $MODEL_PATH"
echo "Seeds: $EVAL_SEEDS"
echo "Tasks: $TEST_TASK_IDS"
echo "Val N: $EVAL_N per task per seed"
echo "User:  $USER_MODEL"
echo "================================================================"

for seed in $EVAL_SEEDS; do
    echo ""
    echo "--- Seed $seed ---"
    SEED_DIR="${OUTPUT_ROOT}/seed${seed}"
    TRAJ_DIR="${SEED_DIR}/trajectories"
    rm -rf "$SEED_DIR"
    mkdir -p "$TRAJ_DIR"

    for gpu in $(seq 0 $((${#TASK_GROUPS[@]} - 1))); do
        CUDA_VISIBLE_DEVICES=$gpu python3 "$EVAL_SCRIPT" \
            --domain airline \
            --models "$MODEL_PATH" \
            --task-ids "${TASK_GROUPS[$gpu]}" \
            --val-n "$EVAL_N" \
            --task-split "$TASK_SPLIT" \
            --max-steps "$MAX_STEPS" \
            --temperature "$TEMPERATURE" \
            --top-p "$TOP_P" \
            --thinking-mode "$THINKING_MODE" \
            --user-model "$USER_MODEL" \
            --output-dir "${SEED_DIR}/gpu${gpu}" \
            --dump-trajectories "$TRAJ_DIR" \
            --seed "$seed" \
            --vllm-language-model-only "$VLLM_LANGUAGE_MODEL_ONLY" \
            --max-model-len "$MAX_MODEL_LEN" \
            --tensor-parallel-size 1 \
            > "${LOG_DIR}/eval_${RUN_NAME}_seed${seed}_gpu${gpu}.log" 2>&1 &
        echo "  GPU $gpu: tasks=${TASK_GROUPS[$gpu]}"
    done

    wait
    TRAJ_COUNT=$(ls "$TRAJ_DIR"/*.json 2>/dev/null | wc -l)
    echo "  Seed $seed complete: $TRAJ_COUNT trajectories"
    if [ "$TRAJ_COUNT" -ne "$EXPECTED_TRAJ_COUNT" ]; then
        echo "  ERROR: expected $EXPECTED_TRAJ_COUNT trajectories for seed $seed, found $TRAJ_COUNT"
        exit 2
    fi
done

echo ""
echo "================================================================"
echo "All seeds complete. Running aggregation..."
echo "================================================================"

# --- Aggregate results ---
python3 - "$OUTPUT_ROOT" <<'PYEOF'
import json, glob, sys
from collections import defaultdict
from math import comb
from pathlib import Path

output_root = Path(sys.argv[1])
task_results = defaultdict(list)
seed_task_results = defaultdict(lambda: defaultdict(list))

for f in sorted(output_root.rglob("trajectories/*.json")):
    t = json.load(open(f))
    task_id = str(t.get("task_id", ""))
    reward = float(t.get("final_reward", 0))
    seed = str(f.parent.parent.name).replace("seed", "")
    task_results[task_id].append(reward)
    seed_task_results[seed][task_id].append(reward)

print(f"\n{'task':>4s} | ", end="")
seeds = sorted(seed_task_results.keys())
for s in seeds:
    print(f"{'s'+s:>6s}", end=" ")
print(f"| {'p@1':>5s} {'p^2':>5s} {'p^3':>5s} {'p^4':>5s}")
print("-" * (12 + 7 * len(seeds) + 25))

agg = {k: [] for k in ["p1", "p2", "p3", "p4"]}
per_seed_agg = {s: [] for s in seeds}
per_seed_passk = {s: {k: [] for k in ["p1", "p2", "p3", "p4"]} for s in seeds}
task_table = {}

def pass_hat(success_count, trial_count, k):
    if trial_count < k:
        return 0.0
    if k == 1:
        return success_count / trial_count if trial_count else 0.0
    return comb(success_count, k) / comb(trial_count, k) if success_count >= k else 0.0

for task_id in sorted(task_results.keys(), key=lambda x: int(x)):
    rs = task_results[task_id]
    c = sum(1 for r in rs if r >= 1.0)
    n = len(rs)
    p1 = pass_hat(c, n, 1)
    p2 = pass_hat(c, n, 2)
    p3 = pass_hat(c, n, 3)
    p4 = pass_hat(c, n, 4)
    agg["p1"].append(p1)
    agg["p2"].append(p2)
    agg["p3"].append(p3)
    agg["p4"].append(p4)

    print(f"{task_id:>4s} | ", end="")
    seed_p1s = {}
    for s in seeds:
        srs = seed_task_results[s].get(task_id, [])
        sc = sum(1 for r in srs if r >= 1.0)
        sn = len(srs)
        sp1 = pass_hat(sc, sn, 1)
        per_seed_passk[s]["p1"].append(sp1)
        per_seed_passk[s]["p2"].append(pass_hat(sc, sn, 2))
        per_seed_passk[s]["p3"].append(pass_hat(sc, sn, 3))
        per_seed_passk[s]["p4"].append(pass_hat(sc, sn, 4))
        per_seed_agg[s].append(sp1)
        seed_p1s[s] = sp1
        print(f"{sp1:>6.2f}", end=" ")
    print(f"| {p1:>5.2f} {p2:>5.2f} {p3:>5.2f} {p4:>5.2f}")
    task_table[task_id] = {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "per_seed": seed_p1s}

nt = len(agg["p1"])
print("-" * (12 + 7 * len(seeds) + 25))
print(f"mean | ", end="")
for s in seeds:
    sp1 = sum(per_seed_agg[s]) / nt if nt else 0
    print(f"{sp1:>6.3f}", end=" ")
print(f"| {sum(agg['p1'])/nt:>5.3f} {sum(agg['p2'])/nt:>5.3f} {sum(agg['p3'])/nt:>5.3f} {sum(agg['p4'])/nt:>5.3f}")
print(f"\n{nt} tasks, {sum(len(v) for v in task_results.values())} total rollouts")

results = {
    "tasks": nt,
    "total_rollouts": sum(len(v) for v in task_results.values()),
    "seeds": seeds,
    "pass_at_1": sum(agg["p1"]) / nt if nt else 0,
    "pass_pow_2": sum(agg["p2"]) / nt if nt else 0,
    "pass_pow_3": sum(agg["p3"]) / nt if nt else 0,
    "pass_pow_4": sum(agg["p4"]) / nt if nt else 0,
    "per_seed_pass_at_1": {s: sum(per_seed_agg[s]) / nt for s in seeds} if nt else {},
    "per_seed_pass_pow": {
        s: {f"pass^{i}": sum(per_seed_passk[s][f"p{i}"]) / nt for i in range(1, 5)}
        for s in seeds
    } if nt else {},
    "per_task": task_table,
}
results_path = output_root / "results.json"
results_path.write_text(json.dumps(results, indent=2) + "\n")
print(f"\nResults saved to {results_path}")
PYEOF

echo "Done."
