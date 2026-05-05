#!/usr/bin/env python3
"""Audit Tau3 paired-eval outputs for parser/runtime artifacts.

This is intentionally post-hoc and read-only. It helps separate genuine policy
failures from evaluation mechanics such as context exhaustion or XML scalar
type parsing issues.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


NUMERIC_TOOL_FIELDS = {"amount", "total_baggages", "nonfree_baggages"}


def iter_jsonl_rows(output_root: Path):
    for path in sorted(output_root.rglob("rollouts.jsonl")):
        seed = next((part[4:] for part in path.parts if part.startswith("seed")), "")
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_seed"] = seed
            row["_path"] = str(path)
            row["_line"] = line_no
            yield row


def iter_trajectories(output_root: Path):
    for path in sorted(output_root.rglob("trajectories/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["_path"] = str(path)
        yield row


def error_bucket(error_summary: str | None) -> str:
    text = error_summary or ""
    if not text:
        return "none"
    if "maximum context length" in text or "requested 0 output tokens" in text:
        return "context_length"
    if "RateLimit" in text or "Too many tokens" in text:
        return "rate_limit"
    return "other"


def walk_values(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
            yield from walk_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_values(value)


def trajectory_has_numeric_string_tool_arg(traj: dict) -> bool:
    for turn in traj.get("turns", []):
        tool_call = turn.get("tool_call")
        if not isinstance(tool_call, dict):
            continue
        for key, value in walk_values(tool_call.get("arguments") or {}):
            if key in NUMERIC_TOOL_FIELDS and isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip()):
                return True
    return False


def trajectory_has_unsupported_operand(traj: dict) -> bool:
    text = "\n".join(str(turn.get("observation") or "") for turn in traj.get("turns", []))
    return "unsupported operand type" in text


def print_task_table(rows: list[dict], trajs: list[dict]) -> None:
    rows_by_task = defaultdict(list)
    for row in rows:
        rows_by_task[str(row.get("task_id"))].append(row)
    trajs_by_task = defaultdict(list)
    for traj in trajs:
        trajs_by_task[str(traj.get("task_id"))].append(traj)

    print("task n succ ctx_err unsupported_operand numeric_string_arg running max_resp max_ctx")
    for task_id in sorted(rows_by_task, key=lambda x: int(x) if x.isdigit() else x):
        task_rows = rows_by_task[task_id]
        task_trajs = trajs_by_task.get(task_id, [])
        resp_vals = [int(row.get("trajectory_response_tokens_total") or 0) for row in task_rows]
        ctx_vals = [int(row.get("trajectory_continuation_tokens_total") or 0) for row in task_rows]
        print(
            f"{task_id:>4s} "
            f"{len(task_rows):3d} "
            f"{sum(bool(row.get('success')) for row in task_rows):4d} "
            f"{sum(error_bucket(row.get('error_summary')) == 'context_length' for row in task_rows):7d} "
            f"{sum(trajectory_has_unsupported_operand(traj) for traj in task_trajs):19d} "
            f"{sum(trajectory_has_numeric_string_tool_arg(traj) for traj in task_trajs):18d} "
            f"{sum(row.get('terminal_reason') == 'running' for row in task_rows):7d} "
            f"{max(resp_vals) if resp_vals else 0:8d} "
            f"{max(ctx_vals) if ctx_vals else 0:7d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path, help="Unpacked outputs/eval_paired/<run_name> directory")
    args = parser.parse_args()

    rows = list(iter_jsonl_rows(args.output_root))
    trajs = list(iter_trajectories(args.output_root))
    if not rows:
        raise SystemExit(f"No rollouts.jsonl rows found under {args.output_root}")

    print(f"rollout_rows: {len(rows)}")
    print(f"trajectory_jsons: {len(trajs)}")
    print(f"successes: {sum(bool(row.get('success')) for row in rows)}")
    print(f"error_buckets: {dict(Counter(error_bucket(row.get('error_summary')) for row in rows))}")
    print(f"json_parse_error_rows: {sum(int(row.get('json_parse_errors') or 0) > 0 for row in rows)}")
    print(f"unknown_tool_error_rows: {sum(int(row.get('unknown_tool_errors') or 0) > 0 for row in rows)}")
    print(f"tool_execution_error_rows: {sum(int(row.get('tool_execution_errors') or 0) > 0 for row in rows)}")
    print(f"response_clip_rows: {sum(int(row.get('response_clip_count') or 0) > 0 for row in rows)}")
    print(f"unsupported_operand_trajectory_rows: {sum(trajectory_has_unsupported_operand(traj) for traj in trajs)}")
    print(f"numeric_string_tool_arg_trajectory_rows: {sum(trajectory_has_numeric_string_tool_arg(traj) for traj in trajs)}")

    turn_vals = [int(row.get("num_turns") or 0) for row in rows]
    tool_vals = [int(row.get("num_tool_calls") or 0) for row in rows]
    print(f"turns_mean: {statistics.mean(turn_vals):.2f}")
    print(f"tool_calls_mean: {statistics.mean(tool_vals):.2f}")
    print()
    print_task_table(rows, trajs)


if __name__ == "__main__":
    main()
