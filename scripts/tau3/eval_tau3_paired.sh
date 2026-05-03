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

# --- Paths ---
EVAL_SCRIPT="${EVAL_SCRIPT:-$HOME/SDPO-qwen35/scripts/eval_tau3_base_models.py}"
OUTPUT_ROOT="outputs/eval_paired/${RUN_NAME}"
LOG_DIR="logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

# --- vLLM flags ---
export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-true}"

# --- Task split across GPUs ---
# 20 tasks split across 8 GPUs (2-3 tasks each)
TASKS_0="2,6,8"
TASKS_1="13,16"
TASKS_2="18,19"
TASKS_3="22,24"
TASKS_4="25,26"
TASKS_5="29,30,31"
TASKS_6="32,35,37"
TASKS_7="44,45,48"
TASK_GROUPS=("$TASKS_0" "$TASKS_1" "$TASKS_2" "$TASKS_3" "$TASKS_4" "$TASKS_5" "$TASKS_6" "$TASKS_7")

# --- Record eval config ---
cat > "${OUTPUT_ROOT}/eval_config.json" <<EOF
{
  "model_path": "$MODEL_PATH",
  "run_name": "$RUN_NAME",
  "eval_seeds": "$(echo $EVAL_SEEDS)",
  "eval_n": $EVAL_N,
  "test_task_ids": "$TEST_TASK_IDS",
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
    mkdir -p "$TRAJ_DIR"

    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        if [ $gpu -ge ${#TASK_GROUPS[@]} ]; then break; fi
        CUDA_VISIBLE_DEVICES=$gpu python3 "$EVAL_SCRIPT" \
            --domain airline \
            --models "$MODEL_PATH" \
            --task-ids "${TASK_GROUPS[$gpu]}" \
            --val-n "$EVAL_N" \
            --task-split test \
            --max-steps "$MAX_STEPS" \
            --temperature "$TEMPERATURE" \
            --top-p "$TOP_P" \
            --thinking-mode "$THINKING_MODE" \
            --user-model "$USER_MODEL" \
            --output-dir "${SEED_DIR}/gpu${gpu}" \
            --dump-trajectories "$TRAJ_DIR" \
            --seed "$seed" \
            --vllm-language-model-only true \
            --max-model-len "$MAX_MODEL_LEN" \
            --tensor-parallel-size 1 \
            > "${LOG_DIR}/eval_${RUN_NAME}_seed${seed}_gpu${gpu}.log" 2>&1 &
        echo "  GPU $gpu: tasks=${TASK_GROUPS[$gpu]}"
    done

    wait
    TRAJ_COUNT=$(ls "$TRAJ_DIR"/*.json 2>/dev/null | wc -l)
    echo "  Seed $seed complete: $TRAJ_COUNT trajectories"
done

echo ""
echo "================================================================"
echo "All seeds complete. Running aggregation..."
echo "================================================================"

# --- Aggregate results ---
python3 - "$OUTPUT_ROOT" <<'PYEOF'
import json, glob, sys
from collections import defaultdict
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
task_table = {}

for task_id in sorted(task_results.keys(), key=lambda x: int(x)):
    rs = task_results[task_id]
    p1 = sum(1 for r in rs if r >= 1.0) / len(rs) if rs else 0
    agg["p1"].append(p1)
    agg["p2"].append(p1 ** 2)
    agg["p3"].append(p1 ** 3)
    agg["p4"].append(p1 ** 4)

    print(f"{task_id:>4s} | ", end="")
    seed_p1s = {}
    for s in seeds:
        srs = seed_task_results[s].get(task_id, [])
        sp1 = sum(1 for r in srs if r >= 1.0) / len(srs) if srs else 0
        per_seed_agg[s].append(sp1)
        seed_p1s[s] = sp1
        print(f"{sp1:>6.2f}", end=" ")
    print(f"| {p1:>5.2f} {p1**2:>5.2f} {p1**3:>5.2f} {p1**4:>5.2f}")
    task_table[task_id] = {"p1": p1, "p2": p1**2, "p3": p1**3, "p4": p1**4, "per_seed": seed_p1s}

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
    "per_task": task_table,
}
results_path = output_root / "results.json"
results_path.write_text(json.dumps(results, indent=2) + "\n")
print(f"\nResults saved to {results_path}")
PYEOF

echo "Done."
