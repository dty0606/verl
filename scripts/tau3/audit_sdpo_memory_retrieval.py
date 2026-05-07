#!/usr/bin/env python3
"""Audit Tau3 Memory-SDPO retrieval before live training.

This script is intentionally offline. It answers two cheap questions before we
spend P5 time:

1. Which memories would different retrieval backends select for failed prefixes?
2. How expensive is each retrieval backend on this memory/query set?

It does not call the SDPO teacher model and it does not change training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
strip_thinking = _MEMORY_MODULE.strip_thinking
tokens_for_lexical = _MEMORY_MODULE._tokens


@dataclass
class JsonRow:
    row: dict[str, Any]
    source_file: str
    source_index: int


@dataclass
class MemoryUnit:
    memory_id: str
    unit_type: str
    text: str
    display_text: str
    metadata: dict[str, Any]
    tokens: set[str]


@dataclass
class QuerySample:
    query_id: str
    text: str
    query_mode: str
    parts: dict[str, str]
    part_lengths: dict[str, int]
    metadata: dict[str, Any]
    tokens: set[str]


class EmbeddingCache:
    def __init__(self, path: Path | None):
        self.path = path
        self.values: dict[str, list[float]] = {}
        self.dirty = False
        if path and path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    key = payload.get("key")
                    embedding = payload.get("embedding")
                    if isinstance(key, str) and isinstance(embedding, list):
                        self.values[key] = [float(x) for x in embedding]

    def get(self, key: str) -> list[float] | None:
        return self.values.get(key)

    def set(self, key: str, value: list[float]) -> None:
        self.values[key] = value
        self.dirty = True

    def flush(self) -> None:
        if not self.path or not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for key, embedding in self.values.items():
                f.write(json.dumps({"key": key, "embedding": embedding}, ensure_ascii=False) + "\n")
        self.dirty = False


def stable_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def compact_json(obj: Any, *, max_chars: int) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    text = strip_thinking(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def text_value(value: Any, *, max_chars: int = 100_000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:max_chars]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role", "unknown")
                content = item.get("content", "")
                parts.append(f"{role}: {text_value(content, max_chars=max_chars)}")
            else:
                parts.append(str(item))
        return "\n".join(parts)[:max_chars]
    if isinstance(value, dict):
        return compact_json(value, max_chars=max_chars)
    return str(value)[:max_chars]


def row_score(row: dict[str, Any]) -> float:
    for key in ("score", "reward", "final_reward", "accuracy"):
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        for key in ("score", "reward", "accuracy"):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    pass
    return 0.0


def row_task_id(row: dict[str, Any]) -> str | None:
    for key in ("task_id", "tau3_task_id", "task"):
        if row.get(key) is not None:
            return str(row[key])
    additional = row.get("additional")
    if isinstance(additional, dict) and additional.get("task_id") is not None:
        return str(additional["task_id"])
    return None


def row_uid(row: dict[str, Any]) -> str | None:
    for key in ("uid", "group_uid", "prompt_uid"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def row_env_error(row: dict[str, Any]) -> bool:
    if bool(row.get("env_error") or row.get("bedrock_error")):
        return True
    reward_source = str(row.get("reward_source", ""))
    return "env_error" in reward_source


def get_feedback(row: dict[str, Any]) -> str:
    for key in ("feedback", "teacher_feedback", "reward_feedback"):
        value = row.get(key)
        if value is None:
            continue
        return text_value(value, max_chars=12_000)
    return ""


def input_text(row: dict[str, Any]) -> str:
    for key in ("input", "prompt", "messages"):
        if row.get(key) is not None:
            return text_value(row[key], max_chars=120_000)
    return ""


def output_text(row: dict[str, Any]) -> str:
    for key in ("output", "response", "completion", "trajectory"):
        if row.get(key) is not None:
            return text_value(row[key], max_chars=250_000)
    return ""


def clipped_part(text: str, *, max_chars: int) -> str:
    text = str(text or "")
    if max_chars <= 0:
        return ""
    return text[:max_chars]


def read_json_payload(text: str, source_file: str) -> Iterable[JsonRow]:
    text = text.strip()
    if not text:
        return
    if "\n" not in text and text.startswith(("{", "[")):
        payload = json.loads(text)
        if isinstance(payload, list):
            for idx, row in enumerate(payload):
                if isinstance(row, dict):
                    yield JsonRow(row=row, source_file=source_file, source_index=idx)
            return
        if isinstance(payload, dict):
            for key in ("rows", "rollouts", "samples", "data"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    for idx, row in enumerate(rows):
                        if isinstance(row, dict):
                            yield JsonRow(row=row, source_file=source_file, source_index=idx)
                    return
            yield JsonRow(row=payload, source_file=source_file, source_index=0)
            return
    for idx, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            yield JsonRow(row=row, source_file=source_file, source_index=idx)


def iter_json_rows(paths: list[Path]) -> Iterable[JsonRow]:
    for path in paths:
        if path.is_dir():
            files = sorted(
                p
                for p in path.rglob("*")
                if p.is_file() and (p.suffix.lower() in {".json", ".jsonl"} or p.name.endswith(".jsonl"))
            )
            yield from iter_json_rows(files)
            continue
        if tarfile.is_tarfile(path):
            with tarfile.open(path, "r:*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    name = member.name
                    if not (name.endswith(".jsonl") or name.endswith(".json")):
                        continue
                    handle = tar.extractfile(member)
                    if handle is None:
                        continue
                    text = handle.read().decode("utf-8", errors="replace")
                    yield from read_json_payload(text, f"{path}!{name}")
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        yield from read_json_payload(text, str(path))


def maybe_strip_thinking(text: str, *, keep_thinking: bool) -> str:
    return str(text or "") if keep_thinking else strip_thinking(text)


def remove_prompt_boilerplate(text: str) -> str:
    """Drop obvious system/tool-schema boilerplate from a prompt tail.

    This is intentionally conservative. It is only for retrieval-query audits,
    not training-time prompt construction.
    """

    kept_lines: list[str] = []
    for line in str(text or "").splitlines():
        lowered = line.lower()
        if any(
            marker in lowered
            for marker in (
                "you are an airline customer service agent",
                "available tools",
                "tool schema",
                "\"type\": \"function\"",
                "\"parameters\"",
                "<tools>",
                "</tools>",
                "# tools",
            )
        ):
            continue
        kept_lines.append(line)
    compact = "\n".join(kept_lines).strip()
    return compact or str(text or "")


def chunk_text(text: str, *, chunk_chars: int, overlap_chars: int, keep_thinking: bool) -> list[str]:
    text = maybe_strip_thinking(text, keep_thinking=keep_thinking)
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    step = max(1, chunk_chars - overlap_chars)
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_chars].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_chars >= len(text):
            break
    return chunks


def event_chunks(text: str, *, max_chars: int, keep_thinking: bool) -> list[str]:
    text = maybe_strip_thinking(text, keep_thinking=keep_thinking)
    if not text:
        return []
    markers = ("assistant:", "user:", "tool:", "<tool_call>", "</tool_call>", "get_", "search_", "book_", "update_", "cancel_")
    lines = text.splitlines()
    events: list[str] = []
    current: list[str] = []
    for line in lines:
        lowered = line.lower().strip()
        starts_event = any(lowered.startswith(marker) for marker in markers)
        if starts_event and current:
            events.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        events.append("\n".join(current).strip())

    packed: list[str] = []
    current_text = ""
    for event in events or [text]:
        if len(current_text) + len(event) + 2 > max_chars and current_text:
            packed.append(current_text.strip())
            current_text = event
        else:
            current_text = f"{current_text}\n\n{event}".strip()
    if current_text:
        packed.append(current_text.strip())
    return packed


def build_query_samples(
    rows: list[JsonRow],
    *,
    failure_threshold: float,
    max_queries: int,
    failed_response_chars: int,
    prompt_tail_chars: int,
    include_suspicious: bool,
    include_env_errors: bool,
    keep_thinking_in_query: bool,
    query_mode: str,
) -> list[QuerySample]:
    samples: list[QuerySample] = []
    valid_query_modes = {
        "current",
        "feedback_only",
        "failed_output_only",
        "prompt_tail_only",
        "prompt_tail_no_schema",
    }
    if query_mode not in valid_query_modes:
        raise ValueError(f"Unsupported query mode: {query_mode}. Expected one of {sorted(valid_query_modes)}")
    for item in rows:
        row = item.row
        if row_score(row) >= failure_threshold:
            continue
        if row_env_error(row) and not include_env_errors:
            continue
        out = output_text(row)
        if not include_suspicious and looks_suspicious(out):
            continue
        feedback = get_feedback(row)
        prompt = input_text(row)
        prompt_tail = prompt[-prompt_tail_chars:]
        prompt_tail_no_schema = remove_prompt_boilerplate(prompt_tail)
        failed_output = maybe_strip_thinking(out, keep_thinking=keep_thinking_in_query)
        parts = {
            "feedback": clipped_part(feedback, max_chars=12_000),
            "failed_output": clipped_part(failed_output, max_chars=failed_response_chars),
            "prompt_tail": clipped_part(prompt_tail, max_chars=prompt_tail_chars),
            "prompt_tail_no_schema": clipped_part(prompt_tail_no_schema, max_chars=prompt_tail_chars),
        }
        if query_mode == "current":
            selected_parts = ("feedback", "failed_output", "prompt_tail")
        elif query_mode == "feedback_only":
            selected_parts = ("feedback",)
        elif query_mode == "failed_output_only":
            selected_parts = ("failed_output",)
        elif query_mode == "prompt_tail_only":
            selected_parts = ("prompt_tail",)
        elif query_mode == "prompt_tail_no_schema":
            selected_parts = ("prompt_tail_no_schema",)
        query_text = "\n\n".join(parts[key] for key in selected_parts if parts[key])
        if not query_text:
            continue
        query_id = f"{Path(item.source_file).name}:{item.source_index}"
        metadata = {
            "source_file": item.source_file,
            "source_index": item.source_index,
            "score": row_score(row),
            "task_id": row_task_id(row),
            "uid": row_uid(row),
            "env_error": row_env_error(row),
            "feedback_available": bool(feedback),
            "output_chars": len(out),
            "input_chars": len(input_text(row)),
        }
        samples.append(
            QuerySample(
                query_id=query_id,
                text=query_text,
                query_mode=query_mode,
                parts={key: value for key, value in parts.items() if value},
                part_lengths={key: len(value) for key, value in parts.items()},
                metadata=metadata,
                tokens=set(tokens_for_lexical(query_text)),
            )
        )
        if 0 < max_queries <= len(samples):
            break
    return samples


def build_card_memory(paths: list[Path], *, max_card_chars: int) -> list[MemoryUnit]:
    units: list[MemoryUnit] = []
    for path in paths:
        bank = load_memory_bank(str(path), max_card_chars=max_card_chars)
        for card in bank.cards:
            text = card.retrieval_text or card.display_text
            metadata = dict(card.payload)
            metadata.update({"source_file": str(path), "unit_source": "card"})
            units.append(
                MemoryUnit(
                    memory_id=card.card_id,
                    unit_type=str(card.payload.get("unit_type") or "card"),
                    text=text,
                    display_text=card.display_text,
                    metadata=metadata,
                    tokens=set(card.tokens),
                )
            )
    return units


def build_rollout_memory(
    rows: list[JsonRow],
    *,
    units: set[str],
    success_threshold: float,
    max_memory_rows: int,
    max_memory_chars: int,
    chunk_chars: int,
    overlap_chars: int,
    include_env_errors: bool,
    keep_thinking_in_memory: bool,
) -> list[MemoryUnit]:
    memory: list[MemoryUnit] = []
    used_rows = 0
    for item in rows:
        row = item.row
        if row_score(row) < success_threshold:
            continue
        if row_env_error(row) and not include_env_errors:
            continue
        prompt = input_text(row)
        out = output_text(row)
        full = "\n\n".join(part for part in (prompt[-4000:], out[:max_memory_chars]) if part)
        base_id = f"{Path(item.source_file).name}:{item.source_index}"
        base_meta = {
            "source_file": item.source_file,
            "source_index": item.source_index,
            "score": row_score(row),
            "task_id": row_task_id(row),
            "uid": row_uid(row),
            "unit_source": "successful_rollout",
        }
        if "full_trajectory" in units and full:
            text = maybe_strip_thinking(full, keep_thinking=keep_thinking_in_memory)[:max_memory_chars]
            memory.append(
                MemoryUnit(
                    memory_id=f"{base_id}:full",
                    unit_type="full_trajectory",
                    text=text,
                    display_text=text[:2000],
                    metadata=base_meta | {"chunk_index": 0},
                    tokens=set(tokens_for_lexical(text)),
                )
            )
        if "char_chunk" in units:
            for chunk_index, chunk in enumerate(
                chunk_text(
                    full,
                    chunk_chars=chunk_chars,
                    overlap_chars=overlap_chars,
                    keep_thinking=keep_thinking_in_memory,
                )
            ):
                memory.append(
                    MemoryUnit(
                        memory_id=f"{base_id}:char:{chunk_index}",
                        unit_type="char_chunk",
                        text=chunk,
                        display_text=chunk[:2000],
                        metadata=base_meta | {"chunk_index": chunk_index},
                        tokens=set(tokens_for_lexical(chunk)),
                    )
                )
        if "event_chunk" in units:
            for chunk_index, chunk in enumerate(
                event_chunks(out, max_chars=chunk_chars, keep_thinking=keep_thinking_in_memory)
            ):
                memory.append(
                    MemoryUnit(
                        memory_id=f"{base_id}:event:{chunk_index}",
                        unit_type="event_chunk",
                        text=chunk,
                        display_text=chunk[:2000],
                        metadata=base_meta | {"chunk_index": chunk_index},
                        tokens=set(tokens_for_lexical(chunk)),
                    )
                )
        used_rows += 1
        if 0 < max_memory_rows <= used_rows:
            break
    return memory


def looks_suspicious(text: str) -> bool:
    lowered = str(text or "").lower()
    if lowered.count("<think>") > lowered.count("</think>"):
        return True
    suspicious_phrases = (
        "assistant<think>",
        "assistant <think>",
        "through a workaround",
        "prices prices",
        "—i—i—i",
    )
    return any(phrase in lowered for phrase in suspicious_phrases)


def lexical_score(query: QuerySample, memory: MemoryUnit) -> float:
    if not query.tokens or not memory.tokens:
        return 0.0
    overlap = len(query.tokens & memory.tokens)
    union = len(query.tokens | memory.tokens) or 1
    jaccard = overlap / union
    memory_coverage = overlap / max(len(memory.tokens), 1)
    query_coverage = overlap / max(len(query.tokens), 1)
    return 0.55 * memory_coverage + 0.30 * jaccard + 0.15 * query_coverage


def stable_random_score(query_id: str, memory_id: str) -> float:
    digest = stable_hash(f"{query_id}::{memory_id}")
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def compatible(query: QuerySample, memory: MemoryUnit, *, block_same_task_id: bool, block_same_uid: bool) -> bool:
    q_task = query.metadata.get("task_id")
    m_task = memory.metadata.get("task_id")
    if block_same_task_id and q_task is not None and m_task is not None and str(q_task) == str(m_task):
        return False
    q_uid = query.metadata.get("uid")
    m_uid = memory.metadata.get("uid")
    if block_same_uid and q_uid is not None and m_uid is not None and str(q_uid) == str(m_uid):
        return False
    return True


def passes_memory_text_filters(
    memory: MemoryUnit,
    *,
    exclude_regex: re.Pattern[str] | None,
    require_regex: re.Pattern[str] | None,
    exclude_generic_opening: bool,
) -> bool:
    text = f"{memory.memory_id}\n{memory.unit_type}\n{memory.text}\n{memory.display_text}"
    if exclude_regex and exclude_regex.search(text):
        return False
    if require_regex and not require_regex.search(text):
        return False
    if exclude_generic_opening and memory.unit_type in {"event_chunk", "char_chunk"}:
        lowered = memory.text.lower()
        action_markers = (
            "book_reservation",
            "update_reservation",
            "cancel_reservation",
            "transfer_to_human_agents",
            "search_direct_flight",
            "search_onestop_flight",
        )
        generic_markers = ("get_user_details", "get_reservation_details")
        has_action = any(marker in lowered for marker in action_markers)
        only_generic = any(marker in lowered for marker in generic_markers) and not has_action
        if only_generic:
            return False
    return True


def topk_by_scores(
    query: QuerySample,
    memory_units: list[MemoryUnit],
    scores: list[float],
    *,
    top_k: int,
    block_same_task_id: bool,
    block_same_uid: bool,
    exclude_regex: re.Pattern[str] | None,
    require_regex: re.Pattern[str] | None,
    exclude_generic_opening: bool,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, MemoryUnit]] = []
    for score, memory in zip(scores, memory_units, strict=True):
        if compatible(query, memory, block_same_task_id=block_same_task_id, block_same_uid=block_same_uid) and passes_memory_text_filters(
            memory,
            exclude_regex=exclude_regex,
            require_regex=require_regex,
            exclude_generic_opening=exclude_generic_opening,
        ):
            ranked.append((score, memory))
    ranked.sort(key=lambda item: (item[0], item[1].memory_id), reverse=True)
    results: list[dict[str, Any]] = []
    for rank, (score, memory) in enumerate(ranked[:top_k], start=1):
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "memory_id": memory.memory_id,
                "unit_type": memory.unit_type,
                "memory_task_id": memory.metadata.get("task_id"),
                "memory_uid": memory.metadata.get("uid"),
                "memory_score": memory.metadata.get("score"),
                "memory_source_file": memory.metadata.get("source_file"),
                "display_preview": memory.display_text[:500],
            }
        )
    return results


def normalize_rows(rows: list[list[float]]) -> list[list[float]]:
    normalized = []
    for row in rows:
        norm = math.sqrt(sum(float(x) * float(x) for x in row))
        if norm <= 0:
            normalized.append([0.0 for _ in row])
        else:
            normalized.append([float(x) / norm for x in row])
    return normalized


class DenseHFBackend:
    def __init__(self, model_name: str, *, batch_size: int, device: str, cache: EmbeddingCache):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.cache = cache
        t0 = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "dense_hf backend requires sentence-transformers. "
                "Install it only in the probe environment, not as a training-path dependency."
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.load_seconds = time.perf_counter() - t0

    def cache_key(self, text: str) -> str:
        return stable_hash(f"dense_hf::{self.model_name}::{text}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        pending: list[str] = []
        pending_keys: list[str] = []
        embeddings: dict[str, list[float]] = {}
        for text in texts:
            key = self.cache_key(text)
            cached = self.cache.get(key)
            if cached is None:
                pending.append(text)
                pending_keys.append(key)
            else:
                embeddings[key] = cached
        if pending:
            vectors = self.model.encode(
                pending,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for key, vector in zip(pending_keys, vectors.tolist(), strict=True):
                value = [float(x) for x in vector]
                self.cache.set(key, value)
                embeddings[key] = value
        return [embeddings[self.cache_key(text)] for text in texts]


class BedrockTitanBackend:
    def __init__(
        self,
        *,
        model_id: str,
        region: str | None,
        dimensions: int,
        normalize: bool,
        retries: int,
        retry_sleep: float,
        cache: EmbeddingCache,
    ):
        self.model_id = model_id
        self.region = region
        self.dimensions = dimensions
        self.normalize = normalize
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.cache = cache
        t0 = time.perf_counter()
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("bedrock_titan backend requires boto3 in the probe environment.") from exc
        kwargs = {"service_name": "bedrock-runtime"}
        if region:
            kwargs["region_name"] = region
        self.client = boto3.client(**kwargs)
        self.load_seconds = time.perf_counter() - t0

    def cache_key(self, text: str) -> str:
        return stable_hash(f"bedrock::{self.model_id}::{self.dimensions}::{self.normalize}::{text}")

    def _embed_one_uncached(self, text: str) -> list[float]:
        body = {
            "inputText": text[:50_000],
            "dimensions": self.dimensions,
            "normalize": self.normalize,
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                    accept="application/json",
                    contentType="application/json",
                )
                payload = json.loads(response["body"].read())
                embedding = payload.get("embedding")
                if not isinstance(embedding, list):
                    raise RuntimeError(f"Bedrock response missing embedding: {payload.keys()}")
                return [float(x) for x in embedding]
            except Exception as exc:  # pragma: no cover - cloud-only branch
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_sleep)
        raise RuntimeError(f"Bedrock embedding failed after {self.retries + 1} attempts") from last_error

    def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            key = self.cache_key(text)
            cached = self.cache.get(key)
            if cached is None:
                cached = self._embed_one_uncached(text)
                self.cache.set(key, cached)
            result.append(cached)
        return normalize_rows(result) if self.normalize else result


def dense_scores(query_embeddings: list[list[float]], memory_embeddings: list[list[float]]) -> list[list[float]]:
    scores_by_query: list[list[float]] = []
    for q_vec in query_embeddings:
        q_scores = []
        for m_vec in memory_embeddings:
            q_scores.append(sum(float(a) * float(b) for a, b in zip(q_vec, m_vec, strict=False)))
        scores_by_query.append(q_scores)
    return scores_by_query


def summarize_results(rows: list[dict[str, Any]], *, timings: dict[str, Any], memory_units: list[MemoryUnit]) -> dict[str, Any]:
    by_backend = defaultdict(list)
    for row in rows:
        by_backend[row["backend"]].append(row)

    top1_by_backend_query: dict[str, dict[str, str | None]] = defaultdict(dict)
    backend_card_counts: dict[str, Counter[str]] = {}
    for backend, backend_rows in by_backend.items():
        counts: Counter[str] = Counter()
        for row in backend_rows:
            top = row["top_k"][0] if row["top_k"] else None
            memory_id = top["memory_id"] if top else None
            top1_by_backend_query[backend][row["query_id"]] = memory_id
            if memory_id:
                counts[memory_id] += 1
        backend_card_counts[backend] = counts

    backends = sorted(by_backend.keys())
    agreement: dict[str, float] = {}
    for i, left in enumerate(backends):
        for right in backends[i + 1 :]:
            shared = sorted(set(top1_by_backend_query[left]) & set(top1_by_backend_query[right]))
            if not shared:
                continue
            same = sum(top1_by_backend_query[left][qid] == top1_by_backend_query[right][qid] for qid in shared)
            agreement[f"{left}__vs__{right}"] = same / len(shared)

    return {
        "num_results": len(rows),
        "num_memory_units": len(memory_units),
        "memory_unit_types": dict(Counter(unit.unit_type for unit in memory_units)),
        "backend_card_counts_top1": {backend: dict(counts.most_common(20)) for backend, counts in backend_card_counts.items()},
        "backend_top1_agreement": agreement,
        "timings": timings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-rollout", action="append", required=True, help="Failed-query rollout JSONL/dir/tarball.")
    parser.add_argument("--memory-card", action="append", default=[], help="Memory-card JSONL/JSON.")
    parser.add_argument("--memory-rollout", action="append", default=[], help="Successful-rollout JSONL/dir/tarball.")
    parser.add_argument(
        "--memory-unit",
        action="append",
        choices=("cards", "full_trajectory", "char_chunk", "event_chunk"),
        default=[],
        help="Memory unit type. Defaults to cards when --memory-card is provided, full_trajectory when --memory-rollout is provided.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=("random", "lexical", "dense_hf", "bedrock_titan"),
        default=[],
        help="Retrieval backend. Defaults to random+lexical.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--max-memory-rows", type=int, default=2000)
    parser.add_argument("--failure-threshold", type=float, default=1.0)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--failed-response-chars", type=int, default=1600)
    parser.add_argument("--prompt-tail-chars", type=int, default=2400)
    parser.add_argument(
        "--query-mode",
        choices=("current", "feedback_only", "failed_output_only", "prompt_tail_only", "prompt_tail_no_schema"),
        default="current",
        help="Which parts form retrieval queries. Defaults to the legacy current mode.",
    )
    parser.add_argument("--max-card-chars", type=int, default=2200)
    parser.add_argument("--max-memory-chars", type=int, default=12000)
    parser.add_argument("--chunk-chars", type=int, default=3000)
    parser.add_argument("--overlap-chars", type=int, default=400)
    parser.add_argument("--include-suspicious", action="store_true")
    parser.add_argument("--include-env-errors", action="store_true")
    parser.add_argument(
        "--keep-thinking-in-query",
        action="store_true",
        help="Keep raw <think> spans in failed-prefix retrieval queries. Default strips them.",
    )
    parser.add_argument(
        "--keep-thinking-in-memory",
        action="store_true",
        help="Keep raw <think> spans in rollout-derived memory units. Default strips them.",
    )
    parser.add_argument("--block-same-task-id", action="store_true")
    parser.add_argument("--block-same-uid", action="store_true")
    parser.add_argument(
        "--exclude-memory-regex",
        default="",
        help="Regex over memory id/type/text/preview to exclude generic or unsafe memory units.",
    )
    parser.add_argument(
        "--require-memory-regex",
        default="",
        help="Regex over memory id/type/text/preview; memory units must match if set.",
    )
    parser.add_argument(
        "--exclude-generic-opening-memory",
        action="store_true",
        help="Drop event/char chunks that only contain generic get_user/get_reservation reads and no action/search/write tool.",
    )
    parser.add_argument("--query-preview-chars", type=int, default=700)
    parser.add_argument("--embedding-cache", default="", help="Optional JSONL embedding cache.")
    parser.add_argument("--hf-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--hf-device", default="cpu")
    parser.add_argument("--hf-batch-size", type=int, default=32)
    parser.add_argument("--bedrock-model-id", default="amazon.titan-embed-text-v2:0")
    parser.add_argument("--bedrock-region", default="")
    parser.add_argument("--bedrock-dimensions", type=int, default=1024)
    parser.add_argument("--bedrock-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bedrock-retries", type=int, default=3)
    parser.add_argument("--bedrock-retry-sleep", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t_start = time.perf_counter()
    query_paths = [Path(item).expanduser() for item in args.query_rollout]
    memory_card_paths = [Path(item).expanduser() for item in args.memory_card]
    memory_rollout_paths = [Path(item).expanduser() for item in args.memory_rollout]
    memory_unit_types = set(args.memory_unit)
    if not memory_unit_types:
        if memory_card_paths:
            memory_unit_types.add("cards")
        if memory_rollout_paths:
            memory_unit_types.add("full_trajectory")
    backends = args.backend or ["random", "lexical"]
    exclude_memory_regex = re.compile(args.exclude_memory_regex, flags=re.IGNORECASE | re.DOTALL) if args.exclude_memory_regex else None
    require_memory_regex = re.compile(args.require_memory_regex, flags=re.IGNORECASE | re.DOTALL) if args.require_memory_regex else None

    timings: dict[str, Any] = {}
    t0 = time.perf_counter()
    query_rows = list(iter_json_rows(query_paths))
    memory_rows = list(iter_json_rows(memory_rollout_paths)) if memory_rollout_paths else []
    timings["read_rows_seconds"] = time.perf_counter() - t0
    timings["query_rows"] = len(query_rows)
    timings["memory_rollout_rows"] = len(memory_rows)

    t0 = time.perf_counter()
    queries = build_query_samples(
        query_rows,
        failure_threshold=args.failure_threshold,
        max_queries=args.max_queries,
        failed_response_chars=args.failed_response_chars,
        prompt_tail_chars=args.prompt_tail_chars,
        include_suspicious=args.include_suspicious,
        include_env_errors=args.include_env_errors,
        keep_thinking_in_query=args.keep_thinking_in_query,
        query_mode=args.query_mode,
    )
    memory_units: list[MemoryUnit] = []
    if "cards" in memory_unit_types:
        memory_units.extend(build_card_memory(memory_card_paths, max_card_chars=args.max_card_chars))
    rollout_unit_types = memory_unit_types & {"full_trajectory", "char_chunk", "event_chunk"}
    if rollout_unit_types:
        memory_units.extend(
            build_rollout_memory(
                memory_rows,
                units=rollout_unit_types,
                success_threshold=args.success_threshold,
                max_memory_rows=args.max_memory_rows,
                max_memory_chars=args.max_memory_chars,
                chunk_chars=args.chunk_chars,
                overlap_chars=args.overlap_chars,
                include_env_errors=args.include_env_errors,
                keep_thinking_in_memory=args.keep_thinking_in_memory,
            )
        )
    timings["build_units_seconds"] = time.perf_counter() - t0
    timings["query_samples"] = len(queries)
    timings["memory_units"] = len(memory_units)

    if not queries:
        raise SystemExit("No failed query samples found. Check --failure-threshold or rollout format.")
    if not memory_units:
        raise SystemExit("No memory units found. Provide --memory-card and/or --memory-rollout.")

    cache_path = Path(args.embedding_cache).expanduser() if args.embedding_cache else None
    cache = EmbeddingCache(cache_path)
    rows: list[dict[str, Any]] = []

    def result_row(query: QuerySample, backend: str, top_k_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "query_id": query.query_id,
            "backend": backend,
            "query_mode": query.query_mode,
            "query_metadata": query.metadata,
            "query_hash": stable_hash(query.text),
            "query_chars": len(query.text),
            "query_token_count": len(query.tokens),
            "query_part_lengths": query.part_lengths,
            "query_preview": query.text[: args.query_preview_chars],
            "query_part_previews": {
                key: value[: args.query_preview_chars]
                for key, value in query.parts.items()
            },
            "top_k": top_k_rows,
        }

    dense_backend_objects: dict[str, Any] = {}
    for backend in backends:
        t_backend = time.perf_counter()
        if backend == "random":
            for query in queries:
                scores = [stable_random_score(query.query_id, memory.memory_id) for memory in memory_units]
                rows.append(
                    result_row(
                        query,
                        backend,
                        topk_by_scores(
                            query,
                            memory_units,
                            scores,
                            top_k=args.top_k,
                            block_same_task_id=args.block_same_task_id,
                            block_same_uid=args.block_same_uid,
                            exclude_regex=exclude_memory_regex,
                            require_regex=require_memory_regex,
                            exclude_generic_opening=args.exclude_generic_opening_memory,
                        ),
                    )
                )
        elif backend == "lexical":
            for query in queries:
                scores = [lexical_score(query, memory) for memory in memory_units]
                rows.append(
                    result_row(
                        query,
                        backend,
                        topk_by_scores(
                            query,
                            memory_units,
                            scores,
                            top_k=args.top_k,
                            block_same_task_id=args.block_same_task_id,
                            block_same_uid=args.block_same_uid,
                            exclude_regex=exclude_memory_regex,
                            require_regex=require_memory_regex,
                            exclude_generic_opening=args.exclude_generic_opening_memory,
                        ),
                    )
                )
        elif backend == "dense_hf":
            dense_backend_objects[backend] = DenseHFBackend(
                args.hf_model,
                batch_size=args.hf_batch_size,
                device=args.hf_device,
                cache=cache,
            )
            embedder = dense_backend_objects[backend]
            timings["dense_hf_load_seconds"] = embedder.load_seconds
            t_embed = time.perf_counter()
            memory_embeddings = normalize_rows(embedder.embed([unit.text for unit in memory_units]))
            query_embeddings = normalize_rows(embedder.embed([query.text for query in queries]))
            timings["dense_hf_embed_seconds"] = time.perf_counter() - t_embed
            t_search = time.perf_counter()
            scores_by_query = dense_scores(query_embeddings, memory_embeddings)
            timings["dense_hf_search_seconds"] = time.perf_counter() - t_search
            for query, scores in zip(queries, scores_by_query, strict=True):
                rows.append(
                    result_row(
                        query,
                        backend,
                        topk_by_scores(
                            query,
                            memory_units,
                            scores,
                            top_k=args.top_k,
                            block_same_task_id=args.block_same_task_id,
                            block_same_uid=args.block_same_uid,
                            exclude_regex=exclude_memory_regex,
                            require_regex=require_memory_regex,
                            exclude_generic_opening=args.exclude_generic_opening_memory,
                        ),
                    )
                )
        elif backend == "bedrock_titan":
            dense_backend_objects[backend] = BedrockTitanBackend(
                model_id=args.bedrock_model_id,
                region=args.bedrock_region or None,
                dimensions=args.bedrock_dimensions,
                normalize=args.bedrock_normalize,
                retries=args.bedrock_retries,
                retry_sleep=args.bedrock_retry_sleep,
                cache=cache,
            )
            embedder = dense_backend_objects[backend]
            timings["bedrock_titan_load_seconds"] = embedder.load_seconds
            t_embed = time.perf_counter()
            memory_embeddings = normalize_rows(embedder.embed([unit.text for unit in memory_units]))
            query_embeddings = normalize_rows(embedder.embed([query.text for query in queries]))
            timings["bedrock_titan_embed_seconds"] = time.perf_counter() - t_embed
            t_search = time.perf_counter()
            scores_by_query = dense_scores(query_embeddings, memory_embeddings)
            timings["bedrock_titan_search_seconds"] = time.perf_counter() - t_search
            for query, scores in zip(queries, scores_by_query, strict=True):
                rows.append(
                    result_row(
                        query,
                        backend,
                        topk_by_scores(
                            query,
                            memory_units,
                            scores,
                            top_k=args.top_k,
                            block_same_task_id=args.block_same_task_id,
                            block_same_uid=args.block_same_uid,
                            exclude_regex=exclude_memory_regex,
                            require_regex=require_memory_regex,
                            exclude_generic_opening=args.exclude_generic_opening_memory,
                        ),
                    )
                )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        timings[f"{backend}_total_seconds"] = time.perf_counter() - t_backend

    cache.flush()
    output_path = Path(args.output_jsonl).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    timings["total_seconds"] = time.perf_counter() - t_start
    timings["seconds_per_query_per_backend"] = timings["total_seconds"] / max(len(queries) * len(backends), 1)
    summary = summarize_results(rows, timings=timings, memory_units=memory_units)
    summary["output_jsonl"] = str(output_path)
    summary["backends"] = backends
    summary["memory_unit_types_requested"] = sorted(memory_unit_types)
    summary["query_sources"] = [str(path) for path in query_paths]
    summary["memory_card_sources"] = [str(path) for path in memory_card_paths]
    summary["memory_rollout_sources"] = [str(path) for path in memory_rollout_paths]
    summary["thinking_policy"] = {
        "keep_thinking_in_query": bool(args.keep_thinking_in_query),
        "keep_thinking_in_memory": bool(args.keep_thinking_in_memory),
    }
    summary["query_mode"] = args.query_mode
    summary["memory_filters"] = {
        "exclude_memory_regex": args.exclude_memory_regex,
        "require_memory_regex": args.require_memory_regex,
        "exclude_generic_opening_memory": bool(args.exclude_generic_opening_memory),
    }

    summary_path = Path(args.summary_json).expanduser() if args.summary_json else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
