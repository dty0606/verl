#!/usr/bin/env python3
"""Read-only QC harness for Tau3 SDPO P1 stabilization smokes.

This script does not launch training or change training semantics. It prints
the local one-command test set, emits a P5 smoke report template, and can scan
captured P5 logs/metrics for the evidence gates expected by the stabilization
runbook.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOCAL_TEST_COMMANDS = [
    "python -m pytest tests/utils/test_tau3_sdpo_full_logit_loss.py "
    "tests/utils/test_tau3_sdpo_ema_teacher.py "
    "tests/utils/test_tau3_strict_action_reward.py "
    "tests/utils/test_tau3_sdpo_mask_debug.py "
    "tests/utils/test_tau3_faithful_sdpo_config.py "
    "tests/utils/test_tau3_length_metrics.py "
    "tests/utils/test_rollout_skip_on_cpu.py "
    "tests/utils/test_mlflow_key_sanitization.py "
    "tests/utils/test_sdpo_same_uid_group_metrics.py "
    "tests/utils/test_sdpo_p1_stabilization_qc.py",
    "python scripts/tau3/sdpo_p1_stabilization_qc.py --help",
]

P5_SMOKE_COMMAND = """\
SDPO_LOGPROB_DIAGNOSTICS=1 \\
SDPO_CUDA_MEMORY_DIAGNOSTICS=1 \\
SDPO_FAIL_FAST_NONFINITE=1 \\
SDPO_EMA_FINITE_CHECK=1 \\
TOTAL_TRAINING_STEPS=2 \\
TEST_FREQ=1 \\
SAVE_FREQ=1 \\
bash run_local_tau3_sdpo_live_p5.sh "$TASK_PATH" "$RUN_NAME" none 2>&1 | tee "$RUN_LOG"
"""

GATE_ORDER = [
    "strict_reward_metrics_present",
    "mask_debug_sample_present",
    "checkpoint_saved_when_update_counter_changed",
    "early_skip_cause_logged",
    "same_uid_homogeneity_metrics_present",
    "cuda_memory_phase_metrics_present",
    "teacher_scheduling_equivalence",
]

CUDA_PHASES = ("ref_compute_log_prob", "actor_compute_log_prob", "actor_update")


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    detail: str


def flatten_metric_keys(obj: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            full = f"{prefix}/{key_str}" if prefix else key_str
            keys.add(full)
            keys.update(flatten_metric_keys(value, full))
    elif isinstance(obj, list):
        for value in obj:
            keys.update(flatten_metric_keys(value, prefix))
    return keys


def load_metrics(paths: list[Path]) -> tuple[set[str], str]:
    metric_keys: set[str] = set()
    blobs: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        blobs.append(text)
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            metric_keys.update(flatten_metric_keys(rows))
        else:
            metric_keys.update(flatten_metric_keys(json.loads(text)))
    return metric_keys, "\n".join(blobs)


def collect_text(log_paths: list[Path], metric_paths: list[Path]) -> tuple[str, set[str]]:
    log_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in log_paths)
    metric_keys, metric_text = load_metrics(metric_paths) if metric_paths else (set(), "")
    return "\n".join(part for part in (log_text, metric_text) if part), metric_keys


def has_any_metric(metric_keys: set[str], patterns: list[str]) -> bool:
    return any(any(re.search(pattern, key) for pattern in patterns) for key in metric_keys)


def gate_strict_reward_metrics(text: str, metric_keys: set[str]) -> GateResult:
    pass_pow = has_any_metric(metric_keys, [r"(^|/)pass\^1$"]) or bool(re.search(r"\bpass\^1\b|\bpass_at_1\b", text))
    reward_mean = has_any_metric(metric_keys, [r"(^|/)reward/(mean@|mean_at_)1$", r"(^|/)reward/mean$"]) or bool(
        re.search(r"reward/(mean@1|mean_at_1|mean)\b", text)
    )
    if pass_pow and reward_mean:
        return GateResult("strict_reward_metrics_present", "PASS", "found pass^1 and reward mean evidence")
    missing = []
    if not pass_pow:
        missing.append("pass^1/pass_at_1")
    if not reward_mean:
        missing.append("reward mean@1/mean_at_1")
    return GateResult("strict_reward_metrics_present", "FAIL", "missing " + ", ".join(missing))


def gate_mask_debug_sample(text: str, enabled: bool) -> GateResult:
    if not enabled:
        return GateResult("mask_debug_sample_present", "SKIP", "mask debug was not requested for this QC run")
    found = bool(re.search(r"mask[-_ ]debug|sdpo_mask_debug|selected.*mask.*sample|target.*mask.*sample", text, re.I))
    if found:
        return GateResult("mask_debug_sample_present", "PASS", "found mask-debug/sample evidence")
    return GateResult("mask_debug_sample_present", "FAIL", "mask debug enabled but no sample evidence found")


def gate_checkpoint_saved(text: str, update_counter_changed: bool) -> GateResult:
    if not update_counter_changed:
        return GateResult(
            "checkpoint_saved_when_update_counter_changed",
            "SKIP",
            "update counter was not marked changed for this QC run",
        )
    found = bool(
        re.search(
            r"save_checkpoint|saving checkpoint|checkpoint saved|global_step_\d+|checkpoints?/",
            text,
            re.I,
        )
    )
    if found:
        return GateResult("checkpoint_saved_when_update_counter_changed", "PASS", "found checkpoint-save evidence")
    return GateResult(
        "checkpoint_saved_when_update_counter_changed",
        "FAIL",
        "update counter changed but checkpoint-save evidence is missing",
    )


def gate_early_skip_cause(text: str) -> GateResult:
    found = bool(
        re.search(
            r"skip_actor_update_due_empty_sdpo|update_skipped_empty_sdpo_target|optimizer_step_skipped|"
            r"early skip cause|skip_cause|skipped_empty_target",
            text,
            re.I,
        )
    )
    if found:
        return GateResult("early_skip_cause_logged", "PASS", "found actor/EMA skip-cause evidence")
    return GateResult("early_skip_cause_logged", "FAIL", "missing early skip-cause metric or log line")


def gate_same_uid_homogeneity(text: str, metric_keys: set[str]) -> GateResult:
    found_metric = has_any_metric(
        metric_keys,
        [
            r"self_distillation/same_uid_group_count$",
            r"self_distillation/same_uid_strict_mixed_fraction$",
            r"self_distillation/same_uid_mixed_fraction$",
        ],
    )
    found_log = bool(
        re.search(
            r"same_uid_(strict_)?mixed_fraction|same_uid_group_count|sdpo_same_uid_warning",
            text,
            re.I,
        )
    )
    if found_metric or found_log:
        return GateResult(
            "same_uid_homogeneity_metrics_present",
            "PASS",
            "found same-UID all-fail/all-success/mixed group evidence",
        )
    return GateResult(
        "same_uid_homogeneity_metrics_present",
        "FAIL",
        "missing same-UID homogeneity metrics or no-mixed warning",
    )


def gate_cuda_memory_phases(text: str, metric_keys: set[str]) -> GateResult:
    missing = []
    for phase in CUDA_PHASES:
        phase_in_log = bool(re.search(r"\[sdpo_cuda_memory\].*" + re.escape(phase), text))
        phase_metric = has_any_metric(metric_keys, [rf"(^|/)cuda_memory/{re.escape(phase)}/"])
        if not phase_in_log and not phase_metric:
            missing.append(phase)
    if not missing:
        return GateResult("cuda_memory_phase_metrics_present", "PASS", "found CUDA memory evidence for required phases")
    return GateResult("cuda_memory_phase_metrics_present", "FAIL", "missing phases: " + ", ".join(missing))


def gate_teacher_scheduling(text: str, metric_keys: set[str], enabled: bool) -> GateResult:
    if not enabled:
        return GateResult(
            "teacher_scheduling_equivalence",
            "SKIP",
            "teacher scheduling equivalence was not requested for this QC run",
        )
    explicit = bool(re.search(r"teacher scheduling equivalence|teacher_scheduling_equivalence", text, re.I))
    backend_metric = has_any_metric(
        metric_keys,
        [
            r"self_distillation/teacher_backend_actor_snapshot$",
            r"self_distillation/teacher_backend_ema_ref$",
        ],
    )
    if explicit or backend_metric:
        return GateResult("teacher_scheduling_equivalence", "PASS", "found teacher-backend/scheduling evidence")
    return GateResult("teacher_scheduling_equivalence", "FAIL", "teacher scheduling enabled but no equivalence evidence found")


def evaluate_gates(
    text: str,
    metric_keys: set[str],
    *,
    mask_debug_enabled: bool = False,
    update_counter_changed: bool = False,
    teacher_scheduling_enabled: bool = False,
) -> list[GateResult]:
    return [
        gate_strict_reward_metrics(text, metric_keys),
        gate_mask_debug_sample(text, mask_debug_enabled),
        gate_checkpoint_saved(text, update_counter_changed),
        gate_early_skip_cause(text),
        gate_same_uid_homogeneity(text, metric_keys),
        gate_cuda_memory_phases(text, metric_keys),
        gate_teacher_scheduling(text, metric_keys, teacher_scheduling_enabled),
    ]


def format_gate_report(results: list[GateResult]) -> str:
    lines = ["# SDPO P1 Stabilization QC Gate Results", ""]
    for result in results:
        lines.append(f"- [{result.status}] {result.name}: {result.detail}")
    return "\n".join(lines) + "\n"


def report_template() -> str:
    gates = "\n".join(f"- [ ] {gate}" for gate in GATE_ORDER)
    commands = "\n".join(f"- `{cmd}`" for cmd in LOCAL_TEST_COMMANDS)
    return f"""# Tau3 SDPO P1 Stabilization QC Report

## Scope
- Branch: `codex/sdpo-p1-stabilization`
- Approved stabilization changes only: strict reward overlay, skip/checkpoint control flow, diagnostics, and QC hooks.
- Smoke target: P1 stabilization on P5 with strict peer-only SDPO.

## Local One-Command Test Set
{commands}

## P5 Smoke Command
```bash
{P5_SMOKE_COMMAND.rstrip()}
```

## Required Gates
{gates}

## Global Safety Sign-Off
- [ ] No dynamic sampling.
- [ ] No new live hindsight context beyond the strict successful-peer SDPO baseline.
- [ ] No NL assertion.
- [ ] No synthetic feedback.

## Evidence Links / Paths
- Run log:
- Metrics artifact:
- Checkpoint directory:
- W&B run:

## Notes
- Mask-debug sample is required only when mask debug is enabled for the smoke.
- Checkpoint-save evidence is required when the update counter changed.
- Teacher scheduling equivalence is required only when teacher scheduling is enabled.
- Same-UID homogeneity metrics should be present in every strict peer-only SDPO smoke.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-tests", action="store_true", help="Print the local one-command test set")
    parser.add_argument("--print-runbook", action="store_true", help="Print the P5 smoke runbook/report template")
    parser.add_argument("--log", type=Path, action="append", default=[], help="P5 smoke log file to scan")
    parser.add_argument("--metrics-json", type=Path, action="append", default=[], help="JSON/JSONL metrics artifact to scan")
    parser.add_argument("--mask-debug-enabled", action="store_true", help="Require a mask-debug sample")
    parser.add_argument("--update-counter-changed", action="store_true", help="Require checkpoint-save evidence")
    parser.add_argument("--teacher-scheduling-enabled", action="store_true", help="Require teacher scheduling equivalence evidence")
    parser.add_argument("--output-md", type=Path, help="Write QC gate results to this Markdown file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_tests:
        print("\n".join(LOCAL_TEST_COMMANDS))
    if args.print_runbook:
        print(report_template())

    if not args.log and not args.metrics_json:
        return 0

    text, metric_keys = collect_text(args.log, args.metrics_json)
    results = evaluate_gates(
        text,
        metric_keys,
        mask_debug_enabled=args.mask_debug_enabled,
        update_counter_changed=args.update_counter_changed,
        teacher_scheduling_enabled=args.teacher_scheduling_enabled,
    )
    report = format_gate_report(results)
    if args.output_md:
        args.output_md.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
