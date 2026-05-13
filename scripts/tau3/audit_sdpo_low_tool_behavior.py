#!/usr/bin/env python3
"""Audit SDPO low-tool-count behavior on train rollouts.

Codex flagged that val `tool_count` collapses from ~4 -> 0.05 across steps
30-180 on the clean r6k SDPO run while `pass^1` stays in the 0.15-0.25 band.
The question is whether zero/near-zero-tool successes are legitimate (refuse/
transfer tasks) or a shallow-final-text exploit.

This script consumes training rollouts from
``$ROLLOUT_DATA_DIR/<step>.jsonl`` and produces:

1. Per-rollout extracted fields: task_id, reward/pass, tool_count,
   turn_count, response_length, think_chars, tool_call_chars,
   final_visible_chars, has_open_think, has_tool_call_markup.
2. Per-step summary: means, and counts for the diagnostic buckets
   (reward=1 AND tool_count=0, reward=1 AND tool_count<=1, reward=0
   AND tool_count=0, high response_length AND low tool_count).
3. Optional category guess per task_id using the task description.

Usage:
    python audit_sdpo_low_tool_behavior.py \\
        --rollout-root ~/tw/outputs/west_p5_original_sdpo_clean_r6k_b4n8_s46_300step_v1/rollout_data \\
        --steps 30,60,90,120,150,180,210 \\
        --output-csv audit_per_rollout.csv \\
        --summary-json audit_summary.json

Heuristics kept conservative. Where a field is missing we emit None and keep
the rollout; we never drop rows silently.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
OPEN_THINK_RE = re.compile(r"<think>", flags=re.IGNORECASE)
CLOSE_THINK_RE = re.compile(r"</think>", flags=re.IGNORECASE)
TOOL_CALL_MARKUP_RE = re.compile(
    r"<tool_call>|</tool_call>|<function=|\"name\"\s*:\s*\"",
    flags=re.IGNORECASE,
)
TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>.*?</tool_call>", flags=re.DOTALL | re.IGNORECASE
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARN parse error {path}:{line_no}: {exc}", file=sys.stderr)


def extract_task_id(row: dict[str, Any]) -> str | None:
    gts = row.get("gts")
    if isinstance(gts, str):
        try:
            gts = json.loads(gts)
        except json.JSONDecodeError:
            gts = None
    if isinstance(gts, dict):
        return str(gts.get("id") or gts.get("task_id") or "")
    return None


def extract_task_category(row: dict[str, Any]) -> dict[str, Any]:
    """Best-effort task-category hints used to decide whether 0-tool success
    is legitimate. Conservative: returns 'unknown' unless explicit signals."""
    gts = row.get("gts")
    if isinstance(gts, str):
        try:
            gts = json.loads(gts)
        except json.JSONDecodeError:
            gts = None
    if not isinstance(gts, dict):
        return {"category": "unknown"}

    desc = gts.get("description") or {}
    if isinstance(desc, str):
        try:
            desc = json.loads(desc)
        except json.JSONDecodeError:
            desc = {}
    scenario = gts.get("user_scenario") or {}
    instr = scenario.get("instructions") or {}
    task_instr = (instr.get("task_instructions") or "").lower() if isinstance(instr, dict) else ""
    reason = (instr.get("reason_for_call") or "").lower() if isinstance(instr, dict) else ""
    purpose = (desc.get("purpose") or "").lower() if isinstance(desc, dict) else ""
    nl_assertions = []
    crit = gts.get("evaluation_criteria") or {}
    if isinstance(crit, dict):
        nl_assertions = [str(a).lower() for a in (crit.get("nl_assertions") or [])]
    actions = []
    if isinstance(crit, dict):
        actions = crit.get("actions") or []
    num_actions = len(actions) if isinstance(actions, list) else 0
    required_action_names = [
        (a.get("name") or "").lower()
        for a in actions
        if isinstance(a, dict)
    ] if isinstance(actions, list) else []

    text_blob = " ".join([purpose, reason, task_instr] + nl_assertions)
    refuse_signal = any(k in text_blob for k in (
        "not allowed",
        "refuse",
        "should not offer",
        "policy does not allow",
        "agent should not",
        "deny",
    ))
    transfer_signal = any(k in text_blob for k in (
        "transfer to",
        "human agent",
        "supervisor",
    ))

    # A task can legitimately succeed with zero/one tool if:
    # - expected required actions are zero
    # - the scenario is a refusal/policy-boundary
    # - OR the scenario requires a transfer
    if num_actions == 0 and (refuse_signal or transfer_signal):
        category = "refuse_or_transfer_zero_action"
    elif num_actions == 0:
        category = "zero_required_actions"
    elif refuse_signal and not required_action_names:
        category = "refuse_signal_no_action"
    elif transfer_signal and not required_action_names:
        category = "transfer_signal_no_action"
    else:
        category = "requires_tools"

    return {
        "category": category,
        "num_required_actions": num_actions,
        "required_action_names": required_action_names,
        "refuse_signal": refuse_signal,
        "transfer_signal": transfer_signal,
    }


def extract_task_text_fields(row: dict[str, Any]) -> dict[str, str]:
    """Extract short task text snippets for manual audit artifacts."""
    gts = row.get("gts")
    if isinstance(gts, str):
        try:
            gts = json.loads(gts)
        except json.JSONDecodeError:
            gts = None
    if not isinstance(gts, dict):
        return {
            "task_purpose": "",
            "reason_for_call": "",
            "task_instructions": "",
            "known_info": "",
        }

    desc = gts.get("description") or {}
    scenario = gts.get("user_scenario") or {}
    instr = scenario.get("instructions") or {}
    return {
        "task_purpose": str(desc.get("purpose") or "") if isinstance(desc, dict) else "",
        "reason_for_call": str(instr.get("reason_for_call") or "") if isinstance(instr, dict) else "",
        "task_instructions": str(instr.get("task_instructions") or "") if isinstance(instr, dict) else "",
        "known_info": str(instr.get("known_info") or "") if isinstance(instr, dict) else "",
    }


def _excerpt(value: Any, limit: int = 2000) -> str:
    text = str(value or "")
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def decompose_output(text: str) -> dict[str, Any]:
    """Split the assistant output into think / tool / final-visible slices."""
    if not text:
        return {
            "think_chars": 0,
            "tool_call_chars": 0,
            "final_visible_chars": 0,
            "has_open_think": False,
            "has_tool_call_markup": False,
            "tool_call_count_text": 0,
        }
    think_total = sum(len(m.group(1)) for m in THINK_RE.finditer(text))
    tool_matches = TOOL_CALL_BLOCK_RE.findall(text)
    tool_chars = sum(len(m) for m in tool_matches)
    residual = THINK_RE.sub(" ", text)
    residual = TOOL_CALL_BLOCK_RE.sub(" ", residual)
    final_visible = " ".join(residual.split())
    return {
        "think_chars": think_total,
        "tool_call_chars": tool_chars,
        "final_visible_chars": len(final_visible),
        "has_open_think": len(OPEN_THINK_RE.findall(text)) > len(CLOSE_THINK_RE.findall(text)),
        "has_tool_call_markup": bool(TOOL_CALL_MARKUP_RE.search(text)),
        "tool_call_count_text": len(tool_matches),
    }


def row_record(row: dict[str, Any], step: int) -> dict[str, Any]:
    task_id = extract_task_id(row) or ""
    cat = extract_task_category(row)
    out = decompose_output(str(row.get("output", "")))
    inp = str(row.get("input", ""))
    tool_count = row.get("tau3_live/tool_count")
    turn_count = row.get("tau3_live/turn_count")
    reward = row.get("score", row.get("reward"))
    try:
        reward_f = float(reward)
    except (TypeError, ValueError):
        reward_f = 0.0
    record = {
        "step": step,
        "source_index": row.get("_source_index"),
        "task_id": task_id,
        "category": cat["category"],
        "num_required_actions": cat["num_required_actions"],
        "required_action_names": "|".join(cat["required_action_names"]),
        "refuse_signal": cat["refuse_signal"],
        "transfer_signal": cat["transfer_signal"],
        "reward": reward_f,
        "pass": int(reward_f >= 1.0),
        "tool_count_metric": tool_count,
        "turn_count_metric": turn_count,
        "input_chars": len(inp),
        "output_chars": len(str(row.get("output", ""))),
        **out,
        "feedback_chars": len(str(row.get("feedback", "") or "")),
    }
    if isinstance(tool_count, (int, float)):
        record["tool_count_metric_text_delta"] = float(tool_count) - float(out["tool_call_count_text"])
        record["tool_count_metric_text_mismatch"] = int(abs(record["tool_count_metric_text_delta"]) > 0.5)
    else:
        record["tool_count_metric_text_delta"] = None
        record["tool_count_metric_text_mismatch"] = None
    return record


def suspicious_record(record: dict[str, Any], source_row: dict[str, Any], bucket: str) -> dict[str, Any]:
    """Return a compact but reviewable suspicious-row artifact.

    The CSV stays metric-only, but suspicious rows need enough text for humans
    to decide whether a zero-tool pass is a legitimate refusal/no-op task or an
    evaluator loophole.
    """
    task_text = extract_task_text_fields(source_row)
    return {
        **record,
        "bucket": bucket,
        "output_excerpt": _excerpt(source_row.get("output"), 4000),
        "feedback_excerpt": _excerpt(source_row.get("feedback"), 4000),
        "task_purpose_excerpt": _excerpt(task_text["task_purpose"], 1200),
        "reason_for_call_excerpt": _excerpt(task_text["reason_for_call"], 1200),
        "task_instructions_excerpt": _excerpt(task_text["task_instructions"], 1600),
        "known_info_excerpt": _excerpt(task_text["known_info"], 1200),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_step[int(r["step"])].append(r)
    out: dict[int, dict[str, Any]] = {}
    for step, rs in sorted(by_step.items()):
        n = len(rs)
        def _mean(key: str) -> float:
            vs = [float(r[key]) for r in rs if isinstance(r[key], (int, float)) and r[key] is not None]
            return fmean(vs) if vs else 0.0
        pass_rows = [r for r in rs if r["pass"] == 1]
        zero_tool_pass = [r for r in pass_rows if (r.get("tool_count_metric") or 0) == 0]
        leq_one_tool_pass = [r for r in pass_rows if (r.get("tool_count_metric") or 0) <= 1]
        zero_tool_fail = [r for r in rs if r["pass"] == 0 and (r.get("tool_count_metric") or 0) == 0]
        long_low_tool = [
            r for r in rs
            if r.get("output_chars", 0) >= 4000 and (r.get("tool_count_metric") or 0) <= 1
        ]
        open_think = [r for r in rs if r.get("has_open_think")]
        no_markup_pass = [r for r in pass_rows if not r.get("has_tool_call_markup")]
        metric_text_mismatch = [r for r in rs if r.get("tool_count_metric_text_mismatch")]
        metric_zero_text_tool = [
            r for r in rs
            if (r.get("tool_count_metric") or 0) == 0 and (r.get("tool_call_count_text") or 0) > 0
        ]
        # Category breakdown of zero-tool passes
        cat_counter: dict[str, int] = defaultdict(int)
        for r in zero_tool_pass:
            cat_counter[r.get("category", "unknown")] += 1
        out[step] = {
            "n_rollouts": n,
            "pass_rate": len(pass_rows) / n,
            "mean_tool_count": _mean("tool_count_metric"),
            "mean_text_tool_call_count": _mean("tool_call_count_text"),
            "mean_turn_count": _mean("turn_count_metric"),
            "mean_output_chars": _mean("output_chars"),
            "mean_think_chars": _mean("think_chars"),
            "mean_tool_call_chars": _mean("tool_call_chars"),
            "mean_final_visible_chars": _mean("final_visible_chars"),
            "open_think_rate": len(open_think) / n,
            "tool_markup_rate": sum(1 for r in rs if r.get("has_tool_call_markup")) / n,
            "pass_n": len(pass_rows),
            "pass_with_zero_tools_n": len(zero_tool_pass),
            "pass_with_zero_tools_rate": len(zero_tool_pass) / max(1, len(pass_rows)),
            "pass_with_leq_one_tool_n": len(leq_one_tool_pass),
            "pass_with_leq_one_tool_rate": len(leq_one_tool_pass) / max(1, len(pass_rows)),
            "pass_with_no_tool_markup_n": len(no_markup_pass),
            "pass_with_no_tool_markup_rate": len(no_markup_pass) / max(1, len(pass_rows)),
            "fail_with_zero_tools_n": len(zero_tool_fail),
            "long_low_tool_n": len(long_low_tool),
            "tool_count_metric_text_mismatch_n": len(metric_text_mismatch),
            "metric_zero_but_text_tool_call_n": len(metric_zero_text_tool),
            "zero_tool_pass_by_category": dict(cat_counter),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--steps", default="30,60,90,120,150,180,210,240,270,300")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--suspicious-jsonl", default="", help="Optional: dump per-rollout rows for suspicious buckets")
    args = parser.parse_args()

    root = Path(args.rollout_root).expanduser()
    steps = [int(s) for s in args.steps.split(",") if s.strip()]

    all_records: list[dict[str, Any]] = []
    suspicious_records: list[dict[str, Any]] = []
    step_file_counts: dict[int, int] = {}
    for step in steps:
        path = root / f"{step}.jsonl"
        if not path.exists():
            print(f"SKIP missing {path}", file=sys.stderr)
            continue
        n = 0
        for i, row in enumerate(iter_jsonl(path)):
            row["_source_index"] = i
            rec = row_record(row, step)
            all_records.append(rec)
            if rec["pass"] == 1 and (rec.get("tool_count_metric") or 0) == 0:
                suspicious_records.append(suspicious_record(rec, row, "pass_zero_tool"))
            elif rec["pass"] == 1 and not rec.get("has_tool_call_markup"):
                suspicious_records.append(suspicious_record(rec, row, "pass_no_markup"))
            elif rec.get("output_chars", 0) >= 4000 and (rec.get("tool_count_metric") or 0) <= 1:
                suspicious_records.append(suspicious_record(rec, row, "long_low_tool"))
            elif (rec.get("tool_count_metric") or 0) == 0 and (rec.get("tool_call_count_text") or 0) > 0:
                suspicious_records.append(suspicious_record(rec, row, "metric_zero_but_text_tool_call"))
            n += 1
        step_file_counts[step] = n

    # Write per-rollout CSV
    if all_records:
        keys = sorted({k for r in all_records for k in r.keys()})
        out_csv = Path(args.output_csv).expanduser()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_records)
        print(f"Wrote {len(all_records)} rollout records to {out_csv}")

    # Summary JSON
    summary = summarize(all_records)
    out_json = Path(args.summary_json).expanduser()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "step_file_counts": step_file_counts,
                "per_step": summary,
                "total_rollouts": len(all_records),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote summary to {out_json}")

    # Print compact table to stdout
    print()
    print(
        f"{'step':>5}  {'n':>4}  {'pass':>5}  {'tool':>5}  {'turn':>5}  "
        f"{'think':>6}  {'visib':>6}  {'p0tool':>7}  {'p<=1tool':>8}  {'mismatch':>8}  {'open_thk':>9}"
    )
    for step, s in summary.items():
        print(
            f"{step:>5}  {s['n_rollouts']:>4}  {s['pass_rate']:>5.3f}  "
            f"{s['mean_tool_count']:>5.2f}  {s['mean_turn_count']:>5.2f}  "
            f"{s['mean_think_chars']:>6.0f}  {s['mean_final_visible_chars']:>6.0f}  "
            f"{s['pass_with_zero_tools_rate']:>7.2%}  "
            f"{s['pass_with_leq_one_tool_rate']:>8.2%}  "
            f"{s['tool_count_metric_text_mismatch_n']:>8}  "
            f"{s['open_think_rate']:>9.2%}"
        )

    # Optional: dump the suspicious rows for manual review
    if args.suspicious_jsonl:
        out_susp = Path(args.suspicious_jsonl).expanduser()
        out_susp.parent.mkdir(parents=True, exist_ok=True)
        with out_susp.open("w", encoding="utf-8") as f:
            for r in suspicious_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote suspicious rows to {out_susp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
