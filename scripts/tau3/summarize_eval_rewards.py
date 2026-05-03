#!/usr/bin/env python3
"""Summarize per-trajectory 0/1 rewards from eval rollouts.

Usage:
    python3 scripts/tau3/summarize_eval_rewards.py outputs/eval_sft_5k_balanced_test20
    python3 scripts/tau3/summarize_eval_rewards.py outputs/eval_base_qwen35_test20
"""
import json
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: summarize_eval_rewards.py <eval_output_dir>")
        sys.exit(1)

    root = Path(sys.argv[1])
    seed_task_rollouts = defaultdict(lambda: defaultdict(list))

    for f in sorted(root.rglob("trajectories/*.json")):
        t = json.load(open(f))
        task_id = str(t.get("task_id", ""))
        reward = float(t.get("final_reward", 0))
        success = 1 if reward >= 1.0 else 0
        m = re.search(r"seed(\d+)", str(f))
        seed = m.group(1) if m else "?"
        sample_id = t.get("sample_id", "?")
        seed_task_rollouts[seed][task_id].append({
            "sample_id": sample_id,
            "reward": reward,
            "success": success,
            "file": str(f.name),
        })

    seeds = sorted(seed_task_rollouts.keys())
    all_tasks = sorted(
        set(tid for s in seeds for tid in seed_task_rollouts[s]),
        key=lambda x: int(x)
    )

    # Per-rollout table
    print(f"{'seed':>5s} {'task':>4s} {'sample':>6s} {'reward':>6s} {'success':>7s}")
    print("-" * 35)
    for seed in seeds:
        for task_id in all_tasks:
            for r in seed_task_rollouts[seed].get(task_id, []):
                print(f"{seed:>5s} {task_id:>4s} {str(r['sample_id']):>6s} {r['reward']:>6.1f} {r['success']:>7d}")

    # Summary
    print(f"\n{'seed':>5s} {'task':>4s} {'n':>3s} {'succ':>4s} {'rate':>5s}")
    print("-" * 25)
    for seed in seeds:
        for task_id in all_tasks:
            rollouts = seed_task_rollouts[seed].get(task_id, [])
            n = len(rollouts)
            c = sum(r["success"] for r in rollouts)
            rate = c / n if n > 0 else 0
            print(f"{seed:>5s} {task_id:>4s} {n:>3d} {c:>4d} {rate:>5.2f}")


if __name__ == "__main__":
    main()
