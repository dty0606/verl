#!/usr/bin/env python3
"""Analyze a bundled Tau3 GRPO readiness smoke.

The bundle can be the directory created by
``scripts/tau3/bundle_grpo_readiness_from_p5.sh`` or an unpacked equivalent.
It is intentionally heuristic: the goal is to catch environment wiring,
runaway generation, transcript duplication, and obvious runtime blockers before
starting a long GRPO/SDPO run.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_KEYS = (
    "response_length/clip_ratio",
    "response_length/max",
    "response_length/mean",
    "response_length_non_aborted/clip_ratio",
    "prompt_length/clip_ratio",
    "prompt_length/max",
    "tau3_live/terminal_fraction",
    "tau3_live/nonterminal_fraction",
    "tau3_live/budget_exhausted_fraction",
    "tau3_live/turn_count",
    "tau3_live/tool_count",
    "critic/rewards/mean",
    "critic/score/mean",
    "actor/pg_loss",
    "actor/grad_norm",
)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_metric_values(text: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    for key in METRIC_KEYS:
        pattern = re.compile(rf"{re.escape(key)}['\"]?\s*[:=]\s*([-+0-9.eE]+)")
        for match in pattern.finditer(text):
            try:
                values[key].append(float(match.group(1)))
            except ValueError:
                pass
    return {key: vals for key, vals in values.items() if vals}


def max_run_length(text: str) -> int:
    longest = 0
    current = 0
    prev = None
    for char in text:
        if char == prev:
            current += 1
        else:
            current = 1
            prev = char
        longest = max(longest, current)
    return longest


def repeated_ngram_ratio(text: str, n: int = 5) -> float:
    words = re.findall(r"\S+", text.lower())
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / max(len(grams), 1)


def transcript_marker_count(text: str) -> int:
    return len(re.findall(r"(?im)^\s*(assistant|user|tool)\s*:", text))


def analyze_rollouts(bundle_dir: Path) -> dict[str, Any]:
    rollout_dir = bundle_dir / "rollout_data"
    if not rollout_dir.exists():
        candidates = sorted(bundle_dir.rglob("rollout_data"))
        rollout_dir = candidates[0] if candidates else rollout_dir
    files = sorted(rollout_dir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    samples: list[dict[str, Any]] = []
    for path in files:
        for row in iter_jsonl(path):
            row["_file"] = path.name
            samples.append(row)

    output_lengths = [len(str(row.get("output", ""))) for row in samples]
    input_lengths = [len(str(row.get("input", ""))) for row in samples]
    scores = []
    for row in samples:
        try:
            scores.append(float(row.get("score")))
        except (TypeError, ValueError):
            pass

    suspicious = []
    for row in samples:
        output = str(row.get("output", ""))
        open_think = output.count("<think>") > output.count("</think>")
        marker_count = transcript_marker_count(output)
        repeat_ratio = repeated_ngram_ratio(output)
        char_run = max_run_length(output)
        if open_think or marker_count > 30 or repeat_ratio > 0.25 or char_run > 80:
            suspicious.append(
                {
                    "file": row.get("_file"),
                    "score": row.get("score"),
                    "output_chars": len(output),
                    "input_chars": len(str(row.get("input", ""))),
                    "open_think": open_think,
                    "transcript_markers": marker_count,
                    "repeat_5gram_ratio": round(repeat_ratio, 4),
                    "max_repeated_char_run": char_run,
                    "output_preview": output[:500],
                }
            )

    return {
        "rollout_files": [p.name for p in files],
        "sample_count": len(samples),
        "score_mean": mean(scores) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "input_chars": summarize_numbers(input_lengths),
        "output_chars": summarize_numbers(output_lengths),
        "suspicious_sample_count": len(suspicious),
        "suspicious_samples": suspicious[:10],
    }


def summarize_numbers(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "mean": round(mean(ordered), 2),
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def summarize_metrics(metrics: dict[str, list[float]]) -> dict[str, dict[str, float | int | None]]:
    return {key: summarize_numbers(vals) for key, vals in metrics.items()}


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    log_text = summary.get("log_text", "")
    metric_summary = summary.get("metric_summary", {})
    rollout = summary.get("rollout_summary", {})

    if "PREFLIGHT=PASS" not in log_text:
        blockers.append("Missing PREFLIGHT=PASS in bundled logs.")
    if "PASS 02_auto_prefix_24k_48k" not in log_text:
        blockers.append("Missing PASS 02_auto_prefix_24k_48k in bundled logs.")
    if "Tau3 all-messages observation: 0" not in log_text:
        blockers.append("Did not prove TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION=0.")
    if re.search(r"AttributeError|undefined symbol|page size of the layer|out of memory|Traceback", log_text, re.I):
        blockers.append("Runtime error signature appears in log.")
    if re.search(r"cuda_runtime\.h.*No such file|cicc: not found", log_text, re.I):
        blockers.append("CUDA/GDN compiler failure signature appears in log.")

    prompt_clip = metric_summary.get("prompt_length/clip_ratio", {}).get("max")
    response_clip = metric_summary.get("response_length/clip_ratio", {}).get("max")
    budget = metric_summary.get("tau3_live/budget_exhausted_fraction", {}).get("max")
    terminal = metric_summary.get("tau3_live/terminal_fraction", {}).get("min")

    if prompt_clip is not None and prompt_clip > 0:
        blockers.append(f"prompt_length/clip_ratio reached {prompt_clip}.")
    if response_clip is not None and response_clip > 0.30:
        warnings.append(f"response_length/clip_ratio reached {response_clip}; inspect rollouts before overnight.")
    if budget is not None and budget > 0.30:
        warnings.append(f"tau3_live/budget_exhausted_fraction reached {budget}.")
    if terminal is not None and terminal < 0.20:
        warnings.append(f"tau3_live/terminal_fraction dropped to {terminal}.")
    if rollout.get("suspicious_sample_count", 0):
        warnings.append(f"{rollout['suspicious_sample_count']} rollout samples triggered repetition/transcript heuristics.")

    decision = "BLOCK" if blockers else ("REVIEW" if warnings else "PASS")
    return {"decision": decision, "blockers": blockers, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    bundle_dir = args.bundle_dir
    explicit_logs = [
        bundle_dir / "full_log.txt",
        bundle_dir / "metrics_summary.txt",
        bundle_dir / "capacity_matrix" / "summary.txt",
        bundle_dir / "capacity_matrix" / "00_preflight.log",
        bundle_dir / "capacity_matrix" / "02_auto_prefix_24k_48k.log",
    ]
    discovered_logs = sorted(bundle_dir.rglob("*.log")) + sorted(bundle_dir.rglob("summary.txt"))
    log_paths = []
    for path in explicit_logs + discovered_logs:
        if path.exists() and path not in log_paths:
            log_paths.append(path)
    log_text = "\n".join(read_text(path) for path in log_paths)
    metrics = extract_metric_values(log_text)
    summary = {
        "bundle_dir": str(bundle_dir),
        "log_files": [str(path.relative_to(bundle_dir)) for path in log_paths],
        "metric_summary": summarize_metrics(metrics),
        "rollout_summary": analyze_rollouts(bundle_dir),
        "log_text": log_text,
    }
    summary["classification"] = classify(summary)
    summary.pop("log_text", None)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
