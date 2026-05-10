#!/usr/bin/env python3
"""Build offline teacher-prompt probes for Tau3 Memory-SDPO.

This script does not call a model. It turns failed rollout JSONLs into prompt
variants that can be scored by a local/vLLM teacher probe:

T0: original SDPO context with feedback only
T1: original SDPO context + relevant compact memory card
T2: original SDPO context + random compact memory card
T3: original prompt only, no feedback/memory lower bound
T4: matched irrelevant compact memory card
T5: same-UID successful peer demonstration when available

The actor/student rollout context is unchanged in training; these prompts are
only for teacher-quality checks before spending a full P5 run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MEMORY_MODULE_PATH = ROOT / "verl" / "utils" / "tau3_sdpo_memory.py"
_MEMORY_SPEC = importlib.util.spec_from_file_location("tau3_sdpo_memory_local", _MEMORY_MODULE_PATH)
if _MEMORY_SPEC is None or _MEMORY_SPEC.loader is None:
    raise RuntimeError(f"Failed to load memory utility from {_MEMORY_MODULE_PATH}")
_MEMORY_MODULE = importlib.util.module_from_spec(_MEMORY_SPEC)
sys.modules[_MEMORY_SPEC.name] = _MEMORY_MODULE
_MEMORY_SPEC.loader.exec_module(_MEMORY_MODULE)

load_memory_bank = _MEMORY_MODULE.load_memory_bank
render_memory_section = _MEMORY_MODULE.render_memory_section
strip_thinking = _MEMORY_MODULE.strip_thinking
is_live_eligible_card = _MEMORY_MODULE._is_live_eligible_card
blocked_by_query_entities = _MEMORY_MODULE._blocked_by_query_entities


DEFAULT_REPROMPT_TEMPLATE = """{prompt}{solution}{memory}{feedback}

Correctly solve the original task."""

DEFAULT_FEEDBACK_TEMPLATE = """

The following is feedback from your unsuccessful earlier attempt:

{feedback_raw}"""

DEFAULT_SOLUTION_TEMPLATE = """

Correct solution:

{successful_previous_attempt}"""


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_rollout_rows(paths: list[Path]):
    for path in paths:
        if path.is_dir():
            files = sorted(path.rglob("*.jsonl"))
        else:
            files = [path]
        for file in files:
            for idx, row in enumerate(iter_jsonl(file)):
                row["_source_file"] = str(file)
                row["_source_index"] = idx
                yield row


def get_feedback(row: dict[str, Any]) -> str:
    for key in ("feedback", "teacher_feedback", "reward_feedback"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return ""


def row_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score", row.get("reward", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def compact_query(row: dict[str, Any], *, failed_response_chars: int) -> str:
    prompt_tail = str(row.get("input", ""))[-2400:]
    return "\n\n".join(
        part
        for part in (
            get_feedback(row),
            strip_thinking(str(row.get("output", "")))[:failed_response_chars],
            prompt_tail,
        )
        if part
    )


def build_prompt(
    *,
    prompt: str,
    feedback: str,
    solution: str = "",
    memory_section: str = "",
    reprompt_template: str = DEFAULT_REPROMPT_TEMPLATE,
    feedback_template: str = DEFAULT_FEEDBACK_TEMPLATE,
    solution_template: str = DEFAULT_SOLUTION_TEMPLATE,
) -> str:
    feedback_section = feedback_template.format(feedback_raw=feedback) if feedback else ""
    solution_section = solution_template.format(successful_previous_attempt=solution) if solution else ""
    return reprompt_template.format(
        prompt=prompt,
        solution=solution_section,
        memory=memory_section,
        feedback=feedback_section,
    )


def looks_suspicious(text: str) -> bool:
    lowered = str(text or "").lower()
    if lowered.count("<think>") > lowered.count("</think>"):
        return True
    if re.search(r"(\bassistant\s*<think>\b|\bthrough a workaround\b|prices\s+prices)", lowered):
        return True
    return False


def choose_irrelevant_card(memory_bank, query: str, relevant, random_card, rng_key: str):
    relevant_id = relevant.card_id if relevant else None
    random_id = random_card.card_id if random_card else None
    blocked = {card_id for card_id in (relevant_id, random_id) if card_id}
    reference = relevant or random_card
    reference_len = len(render_memory_section(reference)) if reference else len(query)
    relevant_cluster = str(relevant.payload.get("task_cluster", "")) if relevant else ""
    relevant_state = str(relevant.payload.get("decision_state", "")) if relevant else ""
    candidates = []
    for card in memory_bank.cards:
        if card.card_id in blocked:
            continue
        if not is_live_eligible_card(card.payload) or blocked_by_query_entities(card.payload, query):
            continue
        same_cluster = relevant_cluster and str(card.payload.get("task_cluster", "")) == relevant_cluster
        same_state = relevant_state and str(card.payload.get("decision_state", "")) == relevant_state
        if same_cluster and same_state:
            continue
        candidates.append((abs(len(render_memory_section(card)) - reference_len), card.card_id, card))
    if candidates:
        candidates.sort()
        return candidates[0][2]
    for offset in range(16):
        card = memory_bank.retrieve(query, mode="random", rng_key=f"{rng_key}:irrelevant:{offset}")
        if card is not None and card.card_id not in blocked:
            return card
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", action="append", required=True, help="Rollout JSONL file or directory.")
    parser.add_argument("--memory-path", required=True, help="JSONL/JSON compact memory cards.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--failure-threshold", type=float, default=1.0)
    parser.add_argument("--failed-response-chars", type=int, default=1400)
    parser.add_argument("--peer-solution-chars", type=int, default=2500)
    parser.add_argument("--include-suspicious", action="store_true", help="Keep obvious runaway outputs in the probe.")
    parser.add_argument(
        "--require-no-successful-peer",
        action="store_true",
        help="If rollout rows include uid, keep only failed rows whose uid group has no successful sample.",
    )
    args = parser.parse_args()

    memory_bank = load_memory_bank(args.memory_path)
    rollout_paths = [Path(item).expanduser() for item in args.rollout]
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_rollout_rows(rollout_paths))
    uid_has_success: dict[str, bool] = {}
    for row in rows:
        uid = row.get("uid")
        if uid is None:
            continue
        key = str(uid)
        uid_has_success[key] = uid_has_success.get(key, False) or row_score(row) >= args.failure_threshold
    success_peer_by_uid: dict[str, str] = {}
    for row in rows:
        uid = row.get("uid")
        if uid is None or row_score(row) < args.failure_threshold:
            continue
        output = strip_thinking(str(row.get("output", "")))
        if output and str(uid) not in success_peer_by_uid:
            success_peer_by_uid[str(uid)] = output[: args.peer_solution_chars]

    written = 0
    arm_counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            if written >= args.max_samples:
                break
            if row_score(row) >= args.failure_threshold:
                continue
            uid = row.get("uid")
            if args.require_no_successful_peer and uid is not None and uid_has_success.get(str(uid), False):
                continue
            output_text = str(row.get("output", ""))
            if not args.include_suspicious and looks_suspicious(output_text):
                continue
            prompt = str(row.get("input", ""))
            feedback = get_feedback(row)
            query = compact_query(row, failed_response_chars=args.failed_response_chars)
            rng_key = f"{row.get('_source_file')}:{row.get('_source_index')}"
            relevant = memory_bank.retrieve(query, mode="relevant", rng_key=rng_key)
            random_card = memory_bank.retrieve(query, mode="random", rng_key=f"{rng_key}:random")
            irrelevant_card = choose_irrelevant_card(memory_bank, query, relevant, random_card, rng_key)
            peer_solution = success_peer_by_uid.get(str(uid)) if uid is not None else None
            variants = [
                ("T0_original", "", "", None),
                (
                    "T1_relevant_memory",
                    render_memory_section(relevant) if relevant else "",
                    "",
                    relevant.card_id if relevant else None,
                ),
                (
                    "T2_random_memory",
                    render_memory_section(random_card) if random_card else "",
                    "",
                    random_card.card_id if random_card else None,
                ),
                ("T3_no_context", "", "", None),
                (
                    "T4_matched_irrelevant_memory",
                    render_memory_section(irrelevant_card) if irrelevant_card else "",
                    "",
                    irrelevant_card.card_id if irrelevant_card else None,
                ),
            ]
            if peer_solution:
                variants.append(("T5_peer_success", "", peer_solution, None))
            sample_id = f"{Path(row.get('_source_file', '')).name}:{row.get('_source_index')}"
            for arm, memory_section, solution, card_id in variants:
                prompt_feedback = "" if arm == "T3_no_context" or solution else feedback
                payload = {
                    "sample_id": sample_id,
                    "arm": arm,
                    "memory_card_id": card_id,
                    "score": row_score(row),
                    "source_file": row.get("_source_file"),
                    "source_index": row.get("_source_index"),
                    "prompt": build_prompt(
                        prompt=prompt,
                        feedback=prompt_feedback,
                        solution=solution,
                        memory_section=memory_section,
                    ),
                    "feedback_available": bool(prompt_feedback),
                    "peer_solution_available": bool(peer_solution),
                    "uid": row.get("uid"),
                    "no_successful_peer_filter_applied": bool(args.require_no_successful_peer),
                    "output_chars": len(output_text),
                }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                arm_counts[arm] = arm_counts.get(arm, 0) + 1
            written += 1

    print(json.dumps({"failed_samples": written, "prompt_variants": arm_counts, "output_jsonl": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
