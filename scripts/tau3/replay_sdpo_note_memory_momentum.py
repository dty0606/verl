#!/usr/bin/env python3
"""Offline replay for Note-SDPO memory-as-teacher-momentum.

This script is intentionally model-free. It does not call the actor, teacher,
Bedrock, vLLM, or any embedding service. It answers a cheap question first:

    If we treat early successful rollouts as a tiny note memory, how much
    coverage/retrieval signal do we get on later risky SDPO failures?

The generated "notes" are safe proxy notes, not final teacher-written notes.
They avoid raw hidden/task-specific content and preserve only mechanical fields
such as tool names and redacted high-level response snippets. A future teacher
writer can consume the emitted sanitized packets to create real decision notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*$", flags=re.DOTALL | re.IGNORECASE)
_TOOL_XML_RE = re.compile(r"<function=([^>]+)>")
_TOOL_JSON_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_USER_ID_RE = re.compile(r"\b[a-z]+_[a-z]+_[0-9]{3,}\b", flags=re.IGNORECASE)
_RESERVATION_RE = re.compile(r"\b[A-Z0-9]{6}\b")
_FLIGHT_RE = re.compile(r"\b[A-Z]{2,4}[0-9]{2,5}\b")
_DATE_RE = re.compile(r"\b(?:20[0-9]{2}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}/[0-9]{1,2}/20[0-9]{2})\b")
_TIMESTAMP_RE = re.compile(r"\b20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9:.:-]+\b")
_MONEY_RE = re.compile(r"\$?\b[0-9]+(?:\.[0-9]{1,2})?\b")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
    "you",
    "your",
}


@dataclass
class RolloutRow:
    source_file: str
    source_index: int
    step: int
    row: dict[str, Any]
    score: float
    output: str
    input_text: str
    feedback: str
    tools: list[str]
    pred: str
    task_id: str | None


@dataclass
class MemoryUnit:
    memory_id: str
    source_step: int
    source_file: str
    source_index: int
    task_id: str | None
    text: str
    display_text: str
    tokens: frozenset[str]


@dataclass
class QueryUnit:
    query_id: str
    step: int
    source_file: str
    source_index: int
    task_id: str | None
    text: str
    tokens: frozenset[str]
    risk_flags: list[str]
    output_chars: int
    repeated_tool_max: int
    score: float


def strip_thinking(text: str) -> str:
    text = _THINK_RE.sub(" ", str(text or ""))
    text = _OPEN_THINK_RE.sub(" ", text)
    return " ".join(text.split())


def redact(text: str) -> str:
    text = strip_thinking(text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _TIMESTAMP_RE.sub("[TIMESTAMP]", text)
    text = _DATE_RE.sub("[DATE]", text)
    text = _USER_ID_RE.sub("[USER_ID]", text)
    text = _FLIGHT_RE.sub("[FLIGHT]", text)
    text = _RESERVATION_RE.sub("[RESERVATION]", text)
    # Keep this late so it does not break fixed placeholders.
    text = _MONEY_RE.sub("[NUM]", text)
    return " ".join(text.split())


def tokens(text: str) -> frozenset[str]:
    return frozenset(
        tok.lower()
        for tok in _TOKEN_RE.findall(redact(text))
        if len(tok) > 2 and tok.lower() not in _STOPWORDS
    )


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:n]


def stable_random(items: list[Any], key: str) -> Any | None:
    if not items:
        return None
    digest = hashlib.sha256(str(key or "").encode("utf-8")).hexdigest()
    return items[int(digest[:16], 16) % len(items)]


def json_text(value: Any, max_chars: int = 20_000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:max_chars]
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)[:max_chars]
    except Exception:
        return str(value)[:max_chars]


def parse_score(row: dict[str, Any]) -> float:
    for key in ("score", "reward", "acc", "accuracy", "final_reward"):
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def parse_feedback(row: dict[str, Any]) -> str:
    for key in ("feedback", "teacher_feedback", "reward_feedback"):
        value = row.get(key)
        if value is not None:
            return json_text(value, max_chars=30_000)
    return ""


def parse_tools(output: str) -> list[str]:
    tools = [item for item in _TOOL_XML_RE.findall(output or "")]
    tools.extend(item for item in _TOOL_JSON_RE.findall(output or ""))
    return [tool.strip() for tool in tools if tool.strip()]


def parse_task_id(row: dict[str, Any]) -> str | None:
    for key in ("task_id", "tau3_task_id"):
        if row.get(key) is not None:
            return str(row[key])
    gts = row.get("gts")
    if gts:
        try:
            payload = json.loads(gts) if isinstance(gts, str) else gts
            if isinstance(payload, dict) and payload.get("id") is not None:
                return str(payload["id"])
        except Exception:
            pass
    return None


def parse_step(path_name: str, row: dict[str, Any]) -> int:
    if row.get("step") is not None:
        try:
            return int(row["step"])
        except (TypeError, ValueError):
            pass
    stem = Path(path_name).stem
    if stem.isdigit():
        return int(stem)
    return -1


def iter_archive_lines(path: Path) -> Iterable[tuple[str, int, dict[str, Any]]]:
    if path.is_dir():
        for file in sorted(path.rglob("*.jsonl")):
            yield from iter_archive_lines(file)
        return
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith(".jsonl"):
                    continue
                with zf.open(name) as handle:
                    for idx, raw in enumerate(handle):
                        line = raw.decode("utf-8", errors="replace").strip()
                        if line:
                            yield name, idx, json.loads(line)
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as tf:
            for member in sorted(tf.getmembers(), key=lambda m: m.name):
                if not member.isfile() or not member.name.endswith(".jsonl"):
                    continue
                handle = tf.extractfile(member)
                if handle is None:
                    continue
                for idx, raw in enumerate(handle):
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        yield member.name, idx, json.loads(line)
        return
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if line:
                yield str(path), idx, json.loads(line)


def load_rows(path: Path) -> list[RolloutRow]:
    rows: list[RolloutRow] = []
    for source_file, source_index, row in iter_archive_lines(path):
        output = json_text(row.get("output") or row.get("response") or row.get("completion"), max_chars=300_000)
        input_text = json_text(row.get("input") or row.get("prompt") or row.get("messages"), max_chars=160_000)
        feedback = parse_feedback(row)
        rows.append(
            RolloutRow(
                source_file=source_file,
                source_index=source_index,
                step=parse_step(source_file, row),
                row=row,
                score=parse_score(row),
                output=output,
                input_text=input_text,
                feedback=feedback,
                tools=parse_tools(output),
                pred=str(row.get("pred") or ""),
                task_id=parse_task_id(row),
            )
        )
    return rows


def repeated_tool_max(tools: list[str]) -> int:
    if not tools:
        return 0
    counts = Counter(tools)
    return max(counts.values())


def risk_flags(row: RolloutRow, *, long_output_chars: int, repeated_tool_threshold: int) -> list[str]:
    flags: list[str] = []
    lowered = row.output.lower()
    if len(row.output) >= long_output_chars:
        flags.append("long_output")
    if lowered.count("<think>") > lowered.count("</think>"):
        flags.append("open_think")
    if repeated_tool_max(row.tools) >= repeated_tool_threshold:
        flags.append("repeated_tool")
    if re.search(r"through a workaround|prices\s+prices|<tool_call>\s*<tool_call>|</think>\s*</think>", lowered):
        flags.append("known_repetition_signature")
    feedback_lower = row.feedback.lower()
    for key in (
        "read_and_stall",
        "incomplete_workflow",
        "unsupported_final_claim",
        "unverified_write_action",
        "state_tracking_error",
        "invalid_tool_arguments",
    ):
        if key in feedback_lower:
            flags.append(key)
    if not flags and row.score < 1.0:
        flags.append("plain_failure")
    return flags


def final_assistant_snippet(output: str, max_chars: int = 320) -> str:
    text = redact(output)
    # Keep only the tail after tool noise as a very rough final-response proxy.
    parts = re.split(r"(?:</tool_call>|tooltool:)", text)
    tail = parts[-1] if parts else text
    return tail[-max_chars:].strip()


def build_memory(row: RolloutRow, *, max_display_chars: int = 900, include_final_snippet: bool = False) -> MemoryUnit:
    tool_counts = Counter(row.tools)
    unique_tools = list(dict.fromkeys(row.tools))
    final_snippet = final_assistant_snippet(row.output) if include_final_snippet else ""
    text = "\n".join(
        part
        for part in (
            "Proxy teacher note from a successful train rollout.",
            f"Source step: {row.step}.",
            f"Successful terminal action/prediction: {redact(row.pred) if row.pred else '[UNKNOWN]'}",
            f"Tool sequence names only: {' -> '.join(unique_tools) if unique_tools else '[NO_TOOL_CALLS]'}",
            "Repeated tool names: "
            + ", ".join(f"{tool} x{count}" for tool, count in tool_counts.items() if count > 1),
            f"Redacted final-response pattern: {final_snippet}" if final_snippet else "",
            "Use as a reusable decision lesson only. Do not copy entity values.",
        )
        if part
    )
    display = text[:max_display_chars].rstrip()
    return MemoryUnit(
        memory_id=f"mem_s{row.step}_{Path(row.source_file).stem}_{row.source_index}_{stable_hash(text, 8)}",
        source_step=row.step,
        source_file=row.source_file,
        source_index=row.source_index,
        task_id=row.task_id,
        text=text,
        display_text=display,
        tokens=tokens(text),
    )


def build_query(row: RolloutRow, flags: list[str], *, output_chars: int = 1400) -> QueryUnit:
    tool_counts = Counter(row.tools)
    repeated = repeated_tool_max(row.tools)
    query_text = "\n".join(
        part
        for part in (
            "Risky failed rollout needing a reusable decision note.",
            f"Step: {row.step}.",
            f"Risk flags: {', '.join(flags)}.",
            f"Predicted/last action: {redact(row.pred) if row.pred else '[UNKNOWN]'}",
            f"Tool sequence names only: {' -> '.join(row.tools) if row.tools else '[NO_TOOL_CALLS]'}",
            "Repeated tool names: "
            + ", ".join(f"{tool} x{count}" for tool, count in tool_counts.items() if count > 1),
            f"Environment feedback: {redact(row.feedback)[:2200]}" if row.feedback else "",
            f"Failed response excerpt: {redact(strip_thinking(row.output))[:output_chars]}",
        )
        if part
    )
    query_id = f"q_s{row.step}_{Path(row.source_file).stem}_{row.source_index}_{stable_hash(query_text, 8)}"
    return QueryUnit(
        query_id=query_id,
        step=row.step,
        source_file=row.source_file,
        source_index=row.source_index,
        task_id=row.task_id,
        text=query_text,
        tokens=tokens(query_text),
        risk_flags=flags,
        output_chars=len(row.output),
        repeated_tool_max=repeated,
        score=row.score,
    )


def lexical_score(query: QueryUnit, memory: MemoryUnit) -> float:
    if not query.tokens or not memory.tokens:
        return 0.0
    overlap = len(query.tokens & memory.tokens)
    union = len(query.tokens | memory.tokens)
    query_coverage = overlap / max(len(query.tokens), 1)
    memory_coverage = overlap / max(len(memory.tokens), 1)
    jaccard = overlap / max(union, 1)
    return 0.55 * memory_coverage + 0.30 * query_coverage + 0.15 * jaccard


def retrieve_topk(query: QueryUnit, memories: list[MemoryUnit], top_k: int) -> list[tuple[MemoryUnit, float]]:
    scored = [(memory, lexical_score(query, memory)) for memory in memories]
    scored.sort(key=lambda item: (item[1], item[0].memory_id), reverse=True)
    return scored[:top_k]


def parse_window(spec: str) -> tuple[str, int, int]:
    label, raw = spec.split("=", 1) if "=" in spec else (spec, spec)
    start_s, end_s = raw.split("-", 1)
    return label, int(start_s), int(end_s)


def in_window(step: int, start: int, end: int) -> bool:
    return start <= step <= end


def summarize_steps(rows: list[RolloutRow]) -> dict[str, Any]:
    by_step: dict[int, list[RolloutRow]] = defaultdict(list)
    for row in rows:
        by_step[row.step].append(row)
    out: dict[str, Any] = {}
    for step in sorted(by_step):
        step_rows = by_step[step]
        lengths = [len(row.output) for row in step_rows]
        out[str(step)] = {
            "n": len(step_rows),
            "successes": sum(row.score >= 1.0 for row in step_rows),
            "mean_score": sum(row.score for row in step_rows) / max(len(step_rows), 1),
            "output_chars_mean": statistics.mean(lengths) if lengths else 0,
            "output_chars_max": max(lengths) if lengths else 0,
            "open_think": sum(row.output.lower().count("<think>") > row.output.lower().count("</think>") for row in step_rows),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-bundle", required=True, help="Zip/tar/jsonl/directory rollout artifact.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--memory-window", action="append", default=["M5=1-5", "M10=1-10", "M20=1-20"])
    parser.add_argument("--query-window", action="append", default=["Q6_10=6-10", "Q11_20=11-20", "Q21_30=21-30", "Q40_47=40-47"])
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--long-output-chars", type=int, default=12_000)
    parser.add_argument("--repeated-tool-threshold", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries-per-window", type=int, default=80)
    parser.add_argument(
        "--include-final-snippet-in-memory",
        action="store_true",
        help="Debug-only: include redacted final response snippets in proxy notes. Off by default to reduce leakage.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    rows = load_rows(Path(args.rollout_bundle).expanduser())
    memory_windows = [parse_window(spec) for spec in args.memory_window]
    query_windows = [parse_window(spec) for spec in args.query_window]

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "rollout_bundle": str(args.rollout_bundle),
        "num_rows": len(rows),
        "step_summary": summarize_steps(rows),
        "windows": {},
        "settings": {
            "success_threshold": args.success_threshold,
            "long_output_chars": args.long_output_chars,
            "repeated_tool_threshold": args.repeated_tool_threshold,
            "top_k": args.top_k,
            "max_queries_per_window": args.max_queries_per_window,
            "memory_text_mode": "safe_proxy_no_gts",
            "include_final_snippet_in_memory": bool(args.include_final_snippet_in_memory),
            "analysis_task_id_note": "task_id is used only for overlap diagnostics, never in memory/query text",
        },
    }

    with output_path.open("w", encoding="utf-8") as out:
        for mem_label, mem_start, mem_end in memory_windows:
            source_rows = [
                row
                for row in rows
                if in_window(row.step, mem_start, mem_end) and row.score >= args.success_threshold
            ]
            memories = [
                build_memory(row, include_final_snippet=bool(args.include_final_snippet_in_memory))
                for row in source_rows
            ]
            memory_task_counts = Counter(memory.task_id for memory in memories if memory.task_id is not None)
            for query_label, query_start, query_end in query_windows:
                query_rows = [
                    row
                    for row in rows
                    if in_window(row.step, query_start, query_end) and row.score < args.success_threshold
                ]
                queries: list[QueryUnit] = []
                for row in query_rows:
                    flags = risk_flags(
                        row,
                        long_output_chars=args.long_output_chars,
                        repeated_tool_threshold=args.repeated_tool_threshold,
                    )
                    if flags:
                        queries.append(build_query(row, flags))
                queries = queries[: args.max_queries_per_window]

                top1_counter: Counter[str] = Counter()
                top1_task_counter: Counter[str | None] = Counter()
                exact_task_hits = 0
                random_exact_task_hits = 0
                scores: list[float] = []
                random_scores: list[float] = []
                risk_counter: Counter[str] = Counter()
                examples: list[dict[str, Any]] = []
                for query in queries:
                    risk_counter.update(query.risk_flags)
                    top = retrieve_topk(query, memories, args.top_k)
                    random_memory = stable_random(memories, query.query_id + ":random")
                    random_score = lexical_score(query, random_memory) if random_memory else 0.0
                    if top:
                        top_memory, top_score = top[0]
                        top1_counter[top_memory.memory_id] += 1
                        top1_task_counter[top_memory.task_id] += 1
                        scores.append(top_score)
                        if query.task_id is not None and top_memory.task_id == query.task_id:
                            exact_task_hits += 1
                    if random_memory:
                        random_scores.append(random_score)
                        if query.task_id is not None and random_memory.task_id == query.task_id:
                            random_exact_task_hits += 1

                    payload = {
                        "memory_window": mem_label,
                        "query_window": query_label,
                        "query_id": query.query_id,
                        "query_step": query.step,
                        "query_source_file": query.source_file,
                        "query_source_index": query.source_index,
                        "query_task_id_analysis_only": query.task_id,
                        "query_risk_flags": query.risk_flags,
                        "query_output_chars": query.output_chars,
                        "query_repeated_tool_max": query.repeated_tool_max,
                        "query_preview": query.text[:1200],
                        "top_k": [
                            {
                                "memory_id": memory.memory_id,
                                "score": score,
                                "source_step": memory.source_step,
                                "source_task_id_analysis_only": memory.task_id,
                                "display_preview": memory.display_text[:900],
                            }
                            for memory, score in top
                        ],
                        "random": {
                            "memory_id": random_memory.memory_id if random_memory else None,
                            "score": random_score,
                            "source_step": random_memory.source_step if random_memory else None,
                            "source_task_id_analysis_only": random_memory.task_id if random_memory else None,
                        },
                    }
                    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    if len(examples) < 8:
                        examples.append(payload)
                    all_results.append(payload)

                key = f"{mem_label}__{query_label}"
                summary["windows"][key] = {
                    "memory_window": {"label": mem_label, "start": mem_start, "end": mem_end},
                    "query_window": {"label": query_label, "start": query_start, "end": query_end},
                    "num_source_success_rows": len(source_rows),
                    "num_memory_units": len(memories),
                    "memory_task_counts_analysis_only": dict(memory_task_counts.most_common(20)),
                    "num_query_failure_rows": len(query_rows),
                    "num_risky_queries_used": len(queries),
                    "risk_flag_counts": dict(risk_counter.most_common()),
                    "top1_memory_counts": dict(top1_counter.most_common(10)),
                    "top1_task_counts_analysis_only": {
                        str(key): value for key, value in top1_task_counter.most_common(10)
                    },
                    "mean_top1_score": statistics.mean(scores) if scores else 0.0,
                    "mean_random_score": statistics.mean(random_scores) if random_scores else 0.0,
                    "exact_task_hit_fraction_analysis_only": exact_task_hits / max(len(queries), 1),
                    "random_exact_task_hit_fraction_analysis_only": random_exact_task_hits / max(len(queries), 1),
                    "examples": examples,
                }

    summary["timings"] = {
        "total_seconds": time.perf_counter() - started,
        "results": len(all_results),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), "output_jsonl": str(output_path), "results": len(all_results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
