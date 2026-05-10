#!/usr/bin/env python3
"""Build teacher-written decision-note memory banks for Tau3 SDPO.

This pipeline is designed for the real Memory/Note-SDPO path:

1. Read train rollout bundles and emit sanitized note-writer prompts.
2. Let a teacher model write compact reusable decision notes.
3. Compile, validate, redact, and merge those notes into a JSONL bank that can
   be used directly by ``tau3.sdpo.memory.path``.

The deterministic code here only handles provenance, sanitization, filtering,
and safety. It deliberately does not hard-code the semantic lesson; the writer
model is responsible for converting a trajectory into a reusable note.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


NOTE_SCHEMA_VERSION = "note_sdpo.v1"
DEFAULT_SYSTEM_PROMPT = """You write reusable decision notes for a tool-using customer-service agent.

Return JSON only. Do not include raw chain-of-thought, names, IDs, exact dates,
reservation numbers, payment numbers, flight numbers, exact prices, hidden task
IDs, or raw tool JSON.

The note should be general enough to help a future similar decision, not a copy
of this trajectory.
"""
DEFAULT_USER_TEMPLATE = """Summarize this successful training trajectory as one reusable decision note.

Write JSON with exactly these fields:
- note_type: "decision_lesson"
- situation: one sentence describing the general situation.
- known_evidence: list of evidence that should be verified before acting.
- decision_boundary: one of sufficient_to_act, need_more_tool_evidence, need_user_clarification, evidence_exhausted_negative, must_refuse, must_transfer.
- correct_action_pattern: list of high-level action steps; use tool names only when useful.
- avoid: list of bad continuations to avoid.
- stop_condition: one sentence saying when the agent should stop searching/thinking and act/refuse/transfer.
- why_reusable: one sentence explaining why this transfers to future tasks.
- applicability: list of conditions that should hold before using this note.
- missing_state_required_before_action: list of evidence that would still need to be collected before acting.
- unsafe_if: list of conditions where this note should not be applied.
- task_cluster: short snake_case cluster name.
- tool_family: list of high-level tool families or tool names.
- confidence: number from 0 to 1.

Keep the whole note concise.

Sanitized training packet:
{packet_json}
"""

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*$", flags=re.DOTALL | re.IGNORECASE)
_TOOL_XML_RE = re.compile(r"<function=([^>]+)>")
_TOOL_JSON_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_TIMESTAMP_RE = re.compile(r"\b20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9:.+-]+\b")
_DATE_RE = re.compile(
    r"\b(?:20[0-9]{2}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}/[0-9]{1,2}/20[0-9]{2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+[0-9]{1,2},?\s+20[0-9]{2})\b",
    flags=re.IGNORECASE,
)
_USER_ID_RE = re.compile(r"\b[a-z]+_[a-z]+_[0-9]{3,}\b", flags=re.IGNORECASE)
_ENTITY_ID_RE = re.compile(
    r"\b(?:reservation|booking|payment|certificate|gift_card|credit_card|user|passenger)[_-]"
    r"(?:id[_-]?)?(?=[A-Za-z0-9_-]*[0-9])[A-Za-z0-9_-]{3,}\b",
    flags=re.IGNORECASE,
)
_RESERVATION_RE = re.compile(r"\b(?=[A-Z0-9]{6,8}\b)(?=[A-Z0-9]*[0-9])[A-Z0-9]{6,8}\b")
_FLIGHT_RE = re.compile(r"\b[A-Z]{2,4}[0-9]{2,5}\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}\b")
_MONEY_RE = re.compile(r"\$[0-9][0-9,]*(?:\.[0-9]{1,2})?")
_NUMBER_RE = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\b")
_JSON_VALUE_RE = re.compile(
    r'"(?P<key>first_name|last_name|user_id|reservation_id|flight_number|payment_id|certificate_id|'
    r'gift_card_id|credit_card_id|email|phone|date_of_birth|dob)"\s*:\s*"(?P<value>[^"]{2,80})"',
    flags=re.IGNORECASE,
)
_BAD_NOTE_RE = re.compile(
    r"<think>|</think>|<tool_call>|</tool_call>|<function=|tooltool:|\"arguments\"\s*:|\"parameters\"\s*:",
    flags=re.IGNORECASE,
)
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
_VALID_BOUNDARIES = {
    "sufficient_to_act",
    "need_more_tool_evidence",
    "need_user_clarification",
    "evidence_exhausted_negative",
    "must_refuse",
    "must_transfer",
}
_REQUIRED_NOTE_FIELDS = (
    "note_type",
    "situation",
    "applicability",
    "known_evidence",
    "missing_state_required_before_action",
    "decision_boundary",
    "correct_action_pattern",
    "avoid",
    "unsafe_if",
    "stop_condition",
    "why_reusable",
    "task_cluster",
    "tool_family",
    "confidence",
)
_ALLOWED_NOTE_FIELDS = set(_REQUIRED_NOTE_FIELDS)
_REQUIRED_LIST_FIELDS = (
    "applicability",
    "known_evidence",
    "missing_state_required_before_action",
    "correct_action_pattern",
    "avoid",
    "unsafe_if",
    "tool_family",
)


@dataclass(frozen=True)
class SourceRow:
    source_file: str
    source_index: int
    step: int
    row: dict[str, Any]
    score: float
    input_text: str
    output_text: str
    feedback_text: str
    pred_text: str
    task_id: str | None
    uid: str | None
    tools: tuple[str, ...]
    sensitive_terms: tuple[str, ...]
    source_hash: str


def stable_hash(value: Any, n: int = 16) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def strip_thinking(text: str) -> str:
    text = _THINK_RE.sub(" ", str(text or ""))
    text = _OPEN_THINK_RE.sub(" ", text)
    return " ".join(text.split())


def compact_json(value: Any, *, max_chars: int = 20_000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:max_chars]
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)[:max_chars]
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


def parse_step(path_name: str, row: dict[str, Any]) -> int:
    if row.get("step") is not None:
        try:
            return int(row["step"])
        except (TypeError, ValueError):
            pass
    stem = Path(path_name).stem
    if stem.isdigit():
        return int(stem)
    match = re.search(r"(?:step|global_step)[_-]?([0-9]+)", path_name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


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


def parse_uid(row: dict[str, Any]) -> str | None:
    for key in ("uid", "uuid", "sample_id", "index"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def parse_text_field(row: dict[str, Any], keys: tuple[str, ...], *, max_chars: int = 40_000) -> str:
    for key in keys:
        if row.get(key) is not None:
            return compact_json(row[key], max_chars=max_chars)
    return ""


def parse_feedback(row: dict[str, Any]) -> str:
    return parse_text_field(row, ("feedback", "teacher_feedback", "reward_feedback"), max_chars=40_000)


def parse_tools(text: str) -> tuple[str, ...]:
    tools: list[str] = []
    tools.extend(item.strip() for item in _TOOL_XML_RE.findall(text or "") if item.strip())
    tools.extend(item.strip() for item in _TOOL_JSON_RE.findall(text or "") if item.strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for tool in tools:
        key = tool.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(tool)
    return tuple(ordered)


def extract_sensitive_terms(text: str, row: dict[str, Any]) -> tuple[str, ...]:
    terms: set[str] = set()
    haystack = "\n".join([text, compact_json(row, max_chars=80_000)])
    for pattern in (_EMAIL_RE, _USER_ID_RE, _ENTITY_ID_RE, _RESERVATION_RE, _FLIGHT_RE, _PHONE_RE):
        for match in pattern.finditer(haystack):
            value = match.group(0)
            terms.add(value)
            if pattern is _USER_ID_RE and not value.lower().startswith(("gift_card_", "credit_card_")):
                for part in re.split(r"[_\s-]+", value):
                    if len(part) >= 3 and not part.isdigit():
                        terms.add(part)
    for match in _JSON_VALUE_RE.finditer(haystack):
        key = match.group("key").lower()
        value = match.group("value").strip()
        if len(value) >= 3:
            terms.add(value)
        if key in {"first_name", "last_name", "user_id"}:
            for part in re.split(r"[\s_,-]+", value):
                clean = part.strip()
                if len(clean) >= 3 and clean.lower() not in _STOPWORDS and not clean.isdigit():
                    terms.add(clean)
    for key in ("task_id", "tau3_task_id", "uid", "user_id", "reservation_id"):
        if row.get(key) is not None:
            terms.add(str(row[key]))
    task_id = parse_task_id(row)
    if task_id:
        terms.add(task_id)
    return tuple(sorted(terms, key=lambda item: (-len(item), item.lower())))


def redact(text: str, sensitive_terms: Iterable[str] = ()) -> str:
    text = strip_thinking(text)
    for term in sorted((str(t).strip() for t in sensitive_terms), key=len, reverse=True):
        if not term or len(term) < 3:
            continue
        text = re.sub(rf"\b{re.escape(term)}\b", "[REDACTED]", text, flags=re.IGNORECASE)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _TIMESTAMP_RE.sub("[TIMESTAMP]", text)
    text = _DATE_RE.sub("[DATE]", text)
    text = _USER_ID_RE.sub("[USER_ID]", text)
    text = _ENTITY_ID_RE.sub("[ENTITY_ID]", text)
    text = _FLIGHT_RE.sub("[FLIGHT]", text)
    text = _RESERVATION_RE.sub("[ID]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _MONEY_RE.sub("[MONEY]", text)
    text = _NUMBER_RE.sub("[NUM]", text)
    return " ".join(text.split())


def has_env_error(row: dict[str, Any]) -> bool:
    for key in ("env_error", "is_env_error"):
        if bool(row.get(key)):
            return True
    text = compact_json(row.get("reward_extra_info") or row.get("extra_info") or row.get("feedback"), max_chars=20_000)
    return "ServiceUnavailableError" in text or "ThrottlingException" in text or "litellm." in text


def iter_json_rows(path: Path) -> Iterable[tuple[str, int, dict[str, Any]]]:
    if path.is_dir():
        for file in sorted(path.rglob("*.jsonl")):
            yield from iter_json_rows(file)
        return
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield str(path), idx, payload
        return
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload if isinstance(payload, list) else payload.get("rows", []) if isinstance(payload, dict) else []
        for idx, row in enumerate(rows):
            if isinstance(row, dict):
                yield str(path), idx, row
        return
    if ".zip" in suffixes:
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith(".jsonl"):
                    continue
                with zf.open(name) as f:
                    for idx, raw in enumerate(f):
                        line = raw.decode("utf-8").strip()
                        if not line:
                            continue
                        payload = json.loads(line)
                        if isinstance(payload, dict):
                            yield f"{path}!{name}", idx, payload
        return
    if ".tgz" in suffixes or ".tar" in suffixes or ".gz" in suffixes:
        with tarfile.open(path) as tf:
            members = sorted((m for m in tf.getmembers() if m.isfile() and m.name.endswith(".jsonl")), key=lambda m: m.name)
            for member in members:
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                for idx, raw in enumerate(extracted):
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield f"{path}!{member.name}", idx, payload
        return
    raise ValueError(f"Unsupported rollout source: {path}")


def load_source_rows(paths: list[Path]) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for path in paths:
        for source_file, source_index, row in iter_json_rows(path):
            input_text = parse_text_field(row, ("input", "prompt", "raw_prompt", "messages"), max_chars=80_000)
            output_text = parse_text_field(row, ("output", "response", "responses", "generated_text"), max_chars=120_000)
            pred_text = parse_text_field(row, ("pred", "prediction"), max_chars=120_000) or output_text
            feedback_text = parse_feedback(row)
            full_text = "\n".join([input_text, output_text, pred_text, feedback_text])
            source_hash = stable_hash({"source": source_file, "idx": source_index, "row": row}, n=20)
            rows.append(
                SourceRow(
                    source_file=source_file,
                    source_index=source_index,
                    step=parse_step(source_file, row),
                    row=row,
                    score=parse_score(row),
                    input_text=input_text,
                    output_text=output_text,
                    feedback_text=feedback_text,
                    pred_text=pred_text,
                    task_id=parse_task_id(row),
                    uid=parse_uid(row),
                    tools=parse_tools(output_text + "\n" + pred_text),
                    sensitive_terms=extract_sensitive_terms(full_text, row),
                    source_hash=source_hash,
                )
            )
    return rows


def filter_source_rows(
    rows: list[SourceRow],
    *,
    min_step: int | None,
    max_step: int | None,
    score_threshold: float,
    include_env_errors: bool,
    max_rows: int | None,
) -> list[SourceRow]:
    selected: list[SourceRow] = []
    seen_hashes: set[str] = set()
    for row in rows:
        if row.source_hash in seen_hashes:
            continue
        if min_step is not None and row.step >= 0 and row.step < min_step:
            continue
        if max_step is not None and row.step >= 0 and row.step > max_step:
            continue
        if row.score < score_threshold:
            continue
        if not include_env_errors and has_env_error(row.row):
            continue
        seen_hashes.add(row.source_hash)
        selected.append(row)
        if max_rows is not None and len(selected) >= max_rows:
            break
    return selected


def tail(text: str, n: int) -> str:
    text = str(text or "")
    return text[-n:] if len(text) > n else text


def head_tail(text: str, n: int) -> str:
    text = str(text or "")
    if len(text) <= n:
        return text
    half = max(n // 2 - 20, 1)
    return text[:half] + "\n...[middle omitted]...\n" + text[-half:]


def build_sanitized_packet(
    row: SourceRow,
    *,
    prompt_tail_chars: int,
    output_excerpt_chars: int,
    feedback_chars: int,
    source_split: str,
    domain: str,
) -> dict[str, Any]:
    redacted_prompt = redact(tail(row.input_text, prompt_tail_chars), row.sensitive_terms)
    redacted_output = redact(head_tail(row.output_text or row.pred_text, output_excerpt_chars), row.sensitive_terms)
    redacted_feedback = redact(tail(row.feedback_text, feedback_chars), row.sensitive_terms)
    return {
        "domain": domain,
        "source_split": source_split,
        "outcome": "success" if row.score >= 1.0 else "non_success",
        "score": row.score,
        "source_step": row.step,
        "tool_sequence": list(row.tools),
        "unique_tools": sorted(set(row.tools)),
        "sanitized_prompt_tail": redacted_prompt,
        "sanitized_assistant_excerpt": redacted_output,
        "sanitized_feedback": redacted_feedback,
        "instructions": [
            "Use this packet to infer a reusable decision lesson.",
            "Do not copy raw wording, names, IDs, dates, prices, or hidden task facts.",
            "Prefer general workflow/action-boundary guidance over task-specific details.",
        ],
    }


def build_prompt_row(
    row: SourceRow,
    *,
    source_run_id: str,
    prompt_tail_chars: int,
    output_excerpt_chars: int,
    feedback_chars: int,
    source_split: str,
    domain: str,
    system_prompt: str,
    user_template: str,
) -> dict[str, Any]:
    packet = build_sanitized_packet(
        row,
        prompt_tail_chars=prompt_tail_chars,
        output_excerpt_chars=output_excerpt_chars,
        feedback_chars=feedback_chars,
        source_split=source_split,
        domain=domain,
    )
    packet_json = json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2)
    prompt_id = f"note_prompt_{row.step if row.step >= 0 else 'x'}_{row.source_index}_{row.source_hash[:10]}"
    return {
        "prompt_id": prompt_id,
        "schema_version": NOTE_SCHEMA_VERSION,
        "source": {
            "source_file": row.source_file,
            "source_index": row.source_index,
            "source_step": row.step,
            "source_run_id": source_run_id,
            "source_hash": row.source_hash,
            "score": row.score,
            "task_id": row.task_id,
            "uid": row.uid,
            "split": source_split,
        },
        "safety": {
            "sensitive_terms_hashes": [stable_hash(term, n=12) for term in row.sensitive_terms],
            "blocked_task_ids": [row.task_id] if row.task_id else [],
            "blocked_entity_ids": list(row.sensitive_terms),
        },
        "packet": packet,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(packet_json=packet_json)},
        ],
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def extract_note_payload(writer_row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if isinstance(writer_row.get("note"), dict):
        return writer_row["note"], ""
    if any(key in writer_row for key in ("situation", "decision_boundary", "correct_action_pattern")):
        return writer_row, ""
    raw = writer_row.get("raw_response") or writer_row.get("content") or writer_row.get("text")
    if raw is None and isinstance(writer_row.get("choices"), list) and writer_row["choices"]:
        try:
            raw = writer_row["choices"][0]["message"]["content"]
        except Exception:
            raw = None
    if not raw:
        return None, "missing_note"
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None, "note_not_json"
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, "note_not_json"
    if not isinstance(parsed, dict):
        return None, "note_not_object"
    return parsed, ""


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [" ".join(str(item).split()) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [" ".join(str(item).split()) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [text]


def normalize_boundary(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "act": "sufficient_to_act",
        "sufficient": "sufficient_to_act",
        "need_tool": "need_more_tool_evidence",
        "need_tool_evidence": "need_more_tool_evidence",
        "ask_user": "need_user_clarification",
        "need_user": "need_user_clarification",
        "refuse": "must_refuse",
        "transfer": "must_transfer",
        "exhausted": "evidence_exhausted_negative",
    }
    text = aliases.get(text, text)
    return text if text in _VALID_BOUNDARIES else None


def note_text_fields(note: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "note_type",
        "situation",
        "known_evidence",
        "applicability",
        "missing_state_required_before_action",
        "decision_boundary",
        "correct_action_pattern",
        "avoid",
        "unsafe_if",
        "stop_condition",
        "why_reusable",
        "task_cluster",
        "tool_family",
        "confidence",
    ):
        value = note.get(key)
        if value is not None:
            parts.append(compact_json(value, max_chars=5000))
    return "\n".join(parts)


def has_identifier_or_raw_trace(text: str) -> bool:
    return bool(
        _BAD_NOTE_RE.search(text)
        or _EMAIL_RE.search(text)
        or _USER_ID_RE.search(text)
        or _ENTITY_ID_RE.search(text)
        or _FLIGHT_RE.search(text)
        or _RESERVATION_RE.search(text)
    )


def parse_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return confidence if 0.0 <= confidence <= 1.0 else None


def validate_and_build_card(
    writer_row: dict[str, Any],
    prompt_by_id: dict[str, dict[str, Any]],
    *,
    writer_model: str,
    max_note_chars: int,
    source_split: str,
    initial_status: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    note, parse_error = extract_note_payload(writer_row)
    prompt_id = str(writer_row.get("prompt_id") or writer_row.get("id") or "")
    prompt = prompt_by_id.get(prompt_id, {})
    source = dict(prompt.get("source") or writer_row.get("source") or {})
    safety = dict(prompt.get("safety") or writer_row.get("safety") or {})
    if parse_error:
        return None, {"prompt_id": prompt_id, "reason": parse_error, "source": source}
    assert note is not None

    missing = [key for key in _REQUIRED_NOTE_FIELDS if key not in note or note.get(key) in (None, "")]
    missing.extend(key for key in _REQUIRED_LIST_FIELDS if key in note and not as_list(note.get(key)))
    if missing:
        return None, {"prompt_id": prompt_id, "reason": "missing_required_fields", "missing": missing, "source": source}
    unknown_fields = sorted(set(note) - _ALLOWED_NOTE_FIELDS)
    if unknown_fields:
        return None, {
            "prompt_id": prompt_id,
            "reason": "unknown_note_fields",
            "unknown_fields": unknown_fields,
            "source": source,
        }
    if str(note.get("note_type", "")).strip() != "decision_lesson":
        return None, {
            "prompt_id": prompt_id,
            "reason": "invalid_note_type",
            "note_type": note.get("note_type"),
            "source": source,
        }
    decision_state = normalize_boundary(note.get("decision_boundary"))
    if decision_state is None:
        return None, {
            "prompt_id": prompt_id,
            "reason": "invalid_decision_boundary",
            "decision_boundary": note.get("decision_boundary"),
            "source": source,
        }
    confidence = parse_confidence(note.get("confidence"))
    if confidence is None:
        return None, {
            "prompt_id": prompt_id,
            "reason": "invalid_confidence",
            "confidence": note.get("confidence"),
            "source": source,
        }

    raw_note_text = note_text_fields(note)
    blocked_entities = safety.get("blocked_entity_ids") or []
    if any(re.search(rf"\b{re.escape(str(term))}\b", raw_note_text, flags=re.IGNORECASE) for term in blocked_entities if str(term).strip()):
        return None, {"prompt_id": prompt_id, "reason": "source_leakage_in_note", "source": source}
    if has_identifier_or_raw_trace(raw_note_text):
        return None, {"prompt_id": prompt_id, "reason": "identifier_pattern_in_note", "source": source}

    clean_note_text = redact(raw_note_text, blocked_entities)
    if _BAD_NOTE_RE.search(clean_note_text):
        return None, {"prompt_id": prompt_id, "reason": "raw_trace_or_thinking_in_note", "source": source}
    if len(clean_note_text) > max_note_chars:
        return None, {
            "prompt_id": prompt_id,
            "reason": "note_too_long",
            "note_chars": len(clean_note_text),
            "source": source,
        }
    if has_identifier_or_raw_trace(clean_note_text):
        return None, {"prompt_id": prompt_id, "reason": "unredacted_identifier_pattern", "source": source}

    known_state = [redact(item, safety.get("blocked_entity_ids") or []) for item in as_list(note.get("known_evidence"))]
    applicability = [redact(item, safety.get("blocked_entity_ids") or []) for item in as_list(note.get("applicability"))]
    missing_state_required_before_action = [
        redact(item, safety.get("blocked_entity_ids") or [])
        for item in as_list(note.get("missing_state_required_before_action"))
    ]
    correct_actions = [
        redact(item, safety.get("blocked_entity_ids") or []) for item in as_list(note.get("correct_action_pattern"))
    ]
    avoid = [redact(item, safety.get("blocked_entity_ids") or []) for item in as_list(note.get("avoid"))]
    unsafe_if = [redact(item, safety.get("blocked_entity_ids") or []) for item in as_list(note.get("unsafe_if"))]
    task_cluster = re.sub(r"[^a-z0-9_]+", "_", str(note.get("task_cluster") or "teacher_written_note").lower()).strip("_")
    if not task_cluster:
        task_cluster = "teacher_written_note"
    tool_family = [re.sub(r"\s+", "_", item.strip().lower()) for item in as_list(note.get("tool_family"))]
    situation = redact(str(note.get("situation") or ""), safety.get("blocked_entity_ids") or [])
    stop_condition = redact(str(note.get("stop_condition") or ""), safety.get("blocked_entity_ids") or [])
    why_reusable = redact(str(note.get("why_reusable") or ""), safety.get("blocked_entity_ids") or [])

    display_payload = {
        "situation": situation,
        "applicability": applicability,
        "known_evidence": known_state,
        "missing_state_required_before_action": missing_state_required_before_action,
        "decision_boundary": decision_state,
        "correct_action_pattern": correct_actions,
        "avoid": avoid,
        "unsafe_if": unsafe_if,
        "stop_condition": stop_condition,
        "why_reusable": why_reusable,
        "confidence": confidence,
    }
    display_text = json.dumps(display_payload, ensure_ascii=False, sort_keys=True)
    embedding_text = "\n".join(
        [
            task_cluster,
            decision_state,
            situation,
            "\n".join(known_state),
            "\n".join(applicability),
            "\n".join(missing_state_required_before_action),
            "\n".join(correct_actions),
            "\n".join(avoid),
            "\n".join(unsafe_if),
            stop_condition,
            why_reusable,
            " ".join(tool_family),
        ]
    )
    card_id = "note_sdpo_" + stable_hash(
        {
            "source_hash": source.get("source_hash"),
            "task_cluster": task_cluster,
            "decision_state": decision_state,
            "embedding_text": embedding_text,
        },
        n=16,
    )
    card = {
        "schema_version": NOTE_SCHEMA_VERSION,
        "card_id": card_id,
        "status": initial_status,
        "split": source.get("split") or source_split,
        "source_split": source.get("split") or source_split,
        "domain": prompt.get("packet", {}).get("domain", ""),
        "unit_type": "teacher_written_decision_note",
        "task_cluster": task_cluster,
        "decision_state": decision_state,
        "failure_pattern": situation,
        "applicability": applicability,
        "known_state": known_state,
        "missing_state_required_before_action": missing_state_required_before_action,
        "correct_next_actions": correct_actions,
        "forbidden_continuations": avoid,
        "unsafe_if": unsafe_if,
        "stop_condition": stop_condition,
        "why_reusable": why_reusable,
        "tool_family": tool_family,
        "confidence": confidence,
        "embedding_text": redact(embedding_text, safety.get("blocked_entity_ids") or []),
        "display_text": redact(display_text, safety.get("blocked_entity_ids") or []),
        "blocked_task_ids": safety.get("blocked_task_ids") or ([source.get("task_id")] if source.get("task_id") else []),
        "blocked_uids": [source.get("uid")] if source.get("uid") else [],
        "blocked_entity_ids": safety.get("blocked_entity_ids") or [],
        "source": {
            "source_run_id": source.get("source_run_id"),
            "source_file": source.get("source_file"),
            "source_index": source.get("source_index"),
            "source_step": source.get("source_step"),
            "source_task_id": source.get("task_id"),
            "source_uid": source.get("uid"),
            "source_hash": source.get("source_hash"),
            "score": source.get("score"),
        },
        "compiler": {
            "name": "build_note_sdpo_bank.py",
            "schema_version": NOTE_SCHEMA_VERSION,
            "prompt_hash": stable_hash(prompt.get("messages") or prompt.get("packet") or {}, n=16),
        },
        "leakage_audit": {
            "passed": True,
            "version": "v1",
            "raw_thinking": False,
            "raw_identifier_pattern": False,
            "source_terms_checked": len(safety.get("blocked_entity_ids") or []),
        },
        "note_writer": {
            "model": writer_model,
            "prompt_id": prompt_id,
            "mode": writer_row.get("writer_mode", "external"),
        },
        "retrieval_eligibility": {
            "default_live": initial_status == "active",
            "requires_train_split": True,
            "min_support": 1,
        },
        "utility_stats": {
            "used_n": 0,
            "teacher_helped_n": 0,
            "teacher_hurt_n": 0,
            "student_helped_n": 0,
            "utility_mean": 0.0,
        },
        "memory_momentum": {
            "support_count": 1,
            "merged_card_ids": [],
            "source_hashes": [source.get("source_hash")] if source.get("source_hash") else [],
        },
    }
    if has_identifier_or_raw_trace(str(card["embedding_text"])) or has_identifier_or_raw_trace(str(card["display_text"])):
        return None, {"prompt_id": prompt_id, "reason": "compiled_text_failed_leakage_scan", "source": source}
    if str(card["split"]).lower() != "train":
        return None, {"prompt_id": prompt_id, "reason": "non_train_split", "source": source}
    return card, None


def dedupe_key(card: dict[str, Any]) -> str:
    text = "\n".join(
        [
            str(card.get("task_cluster", "")),
            str(card.get("decision_state", "")),
            str(card.get("embedding_text", "")),
        ]
    )
    tokens = [tok.lower() for tok in _TOKEN_RE.findall(text) if tok.lower() not in _STOPWORDS]
    return stable_hash(" ".join(tokens), n=20)


def merge_cards(existing: list[dict[str, Any]], new_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for card in existing + new_cards:
        key = dedupe_key(card)
        if key not in merged:
            merged[key] = dict(card)
            momentum = dict(merged[key].get("memory_momentum") or {})
            momentum.setdefault("support_count", 1)
            momentum.setdefault("merged_card_ids", [])
            momentum.setdefault("source_hashes", [])
            merged[key]["memory_momentum"] = momentum
            order.append(key)
            continue
        target = merged[key]
        target_momentum = dict(target.get("memory_momentum") or {})
        card_momentum = dict(card.get("memory_momentum") or {})
        target_momentum["support_count"] = int(target_momentum.get("support_count", 1)) + int(
            card_momentum.get("support_count", 1)
        )
        target_momentum["merged_card_ids"] = sorted(
            set(target_momentum.get("merged_card_ids") or []) | {str(card.get("card_id", ""))}
        )
        target_momentum["source_hashes"] = sorted(
            set(target_momentum.get("source_hashes") or []) | set(card_momentum.get("source_hashes") or [])
        )
        target["memory_momentum"] = target_momentum
        for block_key in ("blocked_task_ids", "blocked_entity_ids", "blocked_reservation_ids", "blocked_user_ids"):
            target[block_key] = sorted(set(target.get(block_key) or []) | set(card.get(block_key) or []))
    return [merged[key] for key in order]


def emit_prompts(args: argparse.Namespace) -> int:
    rows = load_source_rows([Path(item).expanduser() for item in args.rollout])
    selected = filter_source_rows(
        rows,
        min_step=args.source_step_min,
        max_step=args.source_step_max,
        score_threshold=args.score_threshold,
        include_env_errors=args.include_env_errors,
        max_rows=args.max_source_rows,
    )
    prompt_rows = [
        build_prompt_row(
            row,
            source_run_id=args.source_run_id,
            prompt_tail_chars=args.prompt_tail_chars,
            output_excerpt_chars=args.output_excerpt_chars,
            feedback_chars=args.feedback_chars,
            source_split=args.source_split,
            domain=args.domain,
            system_prompt=args.system_prompt,
            user_template=args.user_template,
        )
        for row in selected
    ]
    count = write_jsonl(Path(args.output_prompts), prompt_rows)
    manifest = {
        "mode": "emit-prompts",
        "schema_version": NOTE_SCHEMA_VERSION,
        "rollout_sources": args.rollout,
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "prompts_written": count,
        "step_counts": dict(sorted(Counter(row.step for row in selected).items())),
        "score_threshold": args.score_threshold,
        "source_step_min": args.source_step_min,
        "source_step_max": args.source_step_max,
        "source_run_id": args.source_run_id,
        "output_prompts": args.output_prompts,
    }
    if args.manifest_json:
        Path(args.manifest_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_json).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def openai_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def write_openai(args: argparse.Namespace) -> int:
    prompts = read_jsonl(Path(args.prompt_jsonl))
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key in ${args.api_key_env}")
    output_path = Path(args.writer_output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    errors = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, prompt in enumerate(prompts):
            if args.max_prompts is not None and idx >= args.max_prompts:
                break
            row = {
                "prompt_id": prompt.get("prompt_id"),
                "source": prompt.get("source"),
                "writer_mode": "openai_chat",
                "writer_model": args.writer_model,
            }
            for attempt in range(args.retries + 1):
                try:
                    response = openai_chat_completion(
                        api_base=args.api_base,
                        api_key=api_key,
                        model=args.writer_model,
                        messages=prompt["messages"],
                        temperature=args.temperature,
                        timeout=args.timeout,
                    )
                    row["raw_response"] = response["choices"][0]["message"]["content"]
                    row["response"] = response
                    break
                except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                    row["error"] = repr(exc)
                    if attempt >= args.retries:
                        errors += 1
                    else:
                        time.sleep(args.retry_sleep * (attempt + 1))
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    manifest = {
        "mode": "write-openai",
        "prompt_jsonl": args.prompt_jsonl,
        "writer_output_jsonl": args.writer_output_jsonl,
        "writer_model": args.writer_model,
        "written": written,
        "errors": errors,
    }
    if args.manifest_json:
        Path(args.manifest_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_json).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if errors == 0 or args.allow_writer_errors else 1


def compile_bank(args: argparse.Namespace) -> int:
    prompt_rows = read_jsonl(Path(args.prompt_jsonl)) if args.prompt_jsonl else []
    prompt_by_id = {str(row.get("prompt_id")): row for row in prompt_rows if row.get("prompt_id")}
    writer_rows = read_jsonl(Path(args.writer_output_jsonl))
    existing_cards = read_jsonl(Path(args.existing_bank)) if args.existing_bank else []
    new_cards: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for writer_row in writer_rows:
        card, rejection = validate_and_build_card(
            writer_row,
            prompt_by_id,
            writer_model=args.writer_model,
            max_note_chars=args.max_note_chars,
            source_split=args.source_split,
            initial_status=args.initial_status,
        )
        if card is not None:
            new_cards.append(card)
        elif rejection is not None:
            rejections.append(rejection)
    final_cards = merge_cards(existing_cards, new_cards) if args.merge_similar else existing_cards + new_cards
    output_count = write_jsonl(Path(args.output_bank), final_cards)
    if args.rejections_jsonl:
        write_jsonl(Path(args.rejections_jsonl), rejections)
    reject_counts = Counter(item.get("reason", "unknown") for item in rejections)
    manifest = {
        "mode": "compile",
        "schema_version": NOTE_SCHEMA_VERSION,
        "prompt_jsonl": args.prompt_jsonl,
        "writer_output_jsonl": args.writer_output_jsonl,
        "existing_bank": args.existing_bank,
        "output_bank": args.output_bank,
        "writer_rows": len(writer_rows),
        "accepted_new_cards": len(new_cards),
        "rejected_notes": len(rejections),
        "reject_counts": dict(sorted(reject_counts.items())),
        "existing_cards": len(existing_cards),
        "final_cards": output_count,
        "merge_similar": args.merge_similar,
        "initial_status": args.initial_status,
    }
    if args.manifest_json:
        Path(args.manifest_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_json).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if rejections and args.fail_on_rejection:
        return 1
    return 0


def add_common_emit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rollout", action="append", required=True, help="Rollout source: dir, jsonl, json, zip, tgz, or tar.")
    parser.add_argument("--output-prompts", required=True, help="JSONL path for teacher note-writer prompts.")
    parser.add_argument("--manifest-json", default="", help="Optional manifest JSON output.")
    parser.add_argument("--source-step-min", type=int, default=None)
    parser.add_argument("--source-step-max", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=1.0)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--source-run-id", default="unknown_run")
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--prompt-tail-chars", type=int, default=6000)
    parser.add_argument("--output-excerpt-chars", type=int, default=6000)
    parser.add_argument("--feedback-chars", type=int, default=2000)
    parser.add_argument("--include-env-errors", action="store_true")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser("emit-prompts", help="Emit sanitized teacher-writer prompts from rollout sources.")
    add_common_emit_args(emit)
    emit.set_defaults(func=emit_prompts)

    writer = subparsers.add_parser("write-openai", help="Call an OpenAI-compatible chat-completions endpoint for prompts.")
    writer.add_argument("--prompt-jsonl", required=True)
    writer.add_argument("--writer-output-jsonl", required=True)
    writer.add_argument("--manifest-json", default="")
    writer.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    writer.add_argument("--api-key-env", default="OPENAI_API_KEY")
    writer.add_argument("--writer-model", required=True)
    writer.add_argument("--temperature", type=float, default=0.0)
    writer.add_argument("--timeout", type=int, default=120)
    writer.add_argument("--retries", type=int, default=2)
    writer.add_argument("--retry-sleep", type=float, default=10.0)
    writer.add_argument("--max-prompts", type=int, default=None)
    writer.add_argument("--allow-writer-errors", action="store_true")
    writer.set_defaults(func=write_openai)

    compile_parser = subparsers.add_parser("compile", help="Compile teacher writer outputs into an SDPO memory bank.")
    compile_parser.add_argument("--writer-output-jsonl", required=True)
    compile_parser.add_argument("--output-bank", required=True)
    compile_parser.add_argument("--prompt-jsonl", required=True)
    compile_parser.add_argument("--existing-bank", default="")
    compile_parser.add_argument("--rejections-jsonl", default="")
    compile_parser.add_argument("--manifest-json", default="")
    compile_parser.add_argument("--writer-model", default="external_teacher")
    compile_parser.add_argument("--source-split", default="train")
    compile_parser.add_argument("--initial-status", choices=["active", "candidate", "quarantined"], default="active")
    compile_parser.add_argument("--max-note-chars", type=int, default=2200)
    compile_parser.add_argument("--merge-similar", action=argparse.BooleanOptionalAction, default=True)
    compile_parser.add_argument("--fail-on-rejection", action="store_true")
    compile_parser.set_defaults(func=compile_bank)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
