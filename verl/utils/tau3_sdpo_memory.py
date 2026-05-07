from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
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
    "i",
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


@dataclass(frozen=True)
class Tau3MemoryCard:
    card_id: str
    payload: dict[str, Any]
    retrieval_text: str
    display_text: str
    tokens: frozenset[str]


def strip_thinking(text: str) -> str:
    return " ".join(_THINK_RE.sub(" ", str(text or "")).split())


def _tokens(text: str) -> set[str]:
    return {
        tok.lower()
        for tok in _TOKEN_RE.findall(strip_thinking(text))
        if len(tok) > 2 and tok.lower() not in _STOPWORDS
    }


def _stable_index(key: str, n: int) -> int:
    digest = hashlib.sha256(str(key or "").encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % n


def _compact_json(obj: Any, *, max_chars: int) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    text = strip_thinking(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _card_text(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    parts: list[str] = []
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(value)
        else:
            try:
                parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
            except Exception:
                parts.append(str(value))
    return strip_thinking("\n".join(parts))


def _blocked_by_query_entities(payload: dict[str, Any], query_text: str) -> bool:
    """Prevent exact known IDs from being used if a card declares leakage blocks."""

    query = str(query_text or "").lower()
    for key in ("blocked_task_ids", "blocked_entity_ids", "blocked_reservation_ids", "blocked_user_ids"):
        for value in payload.get(key) or []:
            text = str(value or "").strip().lower()
            if text and text in query:
                return True
    return False


class Tau3MemoryBank:
    def __init__(self, cards: list[Tau3MemoryCard]):
        self.cards = cards

    def retrieve(
        self,
        query_text: str,
        *,
        mode: str = "relevant",
        rng_key: str = "",
        max_chars: int = 2200,
    ) -> Tau3MemoryCard | None:
        candidates = [
            card
            for card in self.cards
            if str(card.payload.get("split", "train")).lower() == "train"
            and not _blocked_by_query_entities(card.payload, query_text)
        ]
        if not candidates:
            return None

        mode = str(mode or "relevant").lower()
        if mode in {"random", "shuffle", "shuffled"}:
            return candidates[_stable_index(rng_key or query_text, len(candidates))]

        query_tokens = _tokens(query_text)
        if not query_tokens:
            return candidates[_stable_index(rng_key or query_text, len(candidates))]

        def score(card: Tau3MemoryCard) -> tuple[float, str]:
            overlap = len(query_tokens & card.tokens)
            union = len(query_tokens | card.tokens) or 1
            jaccard = overlap / union
            card_coverage = overlap / max(len(card.tokens), 1)
            bonus = 0.0
            for key in ("task_cluster", "decision_state", "failure_pattern"):
                value = str(card.payload.get(key, "")).lower()
                if value and any(tok in value for tok in query_tokens):
                    bonus += 0.05
            length_penalty = min(len(card.display_text), max_chars) / max(max_chars, 1) * 0.01
            return (0.75 * card_coverage + 0.25 * jaccard + bonus - length_penalty, card.card_id)

        return max(candidates, key=score)


def _load_one_card(raw: dict[str, Any], idx: int, *, max_card_chars: int) -> Tau3MemoryCard:
    card_id = str(raw.get("card_id") or raw.get("id") or f"card_{idx:04d}")
    retrieval_text = raw.get("embedding_text") or _card_text(
        raw,
        (
            "task_cluster",
            "decision_state",
            "failure_pattern",
            "known_state",
            "correct_next_actions",
            "forbidden_continuations",
            "display_text",
        ),
    )
    display_text = raw.get("display_text") or _compact_json(
        {
            key: raw.get(key)
            for key in (
                "task_cluster",
                "decision_state",
                "failure_pattern",
                "known_state",
                "correct_next_actions",
                "forbidden_continuations",
            )
            if raw.get(key) is not None
        },
        max_chars=max_card_chars,
    )
    return Tau3MemoryCard(
        card_id=card_id,
        payload=raw,
        retrieval_text=strip_thinking(str(retrieval_text or "")),
        display_text=strip_thinking(str(display_text or ""))[:max_card_chars],
        tokens=frozenset(_tokens(str(retrieval_text or "") + "\n" + str(display_text or ""))),
    )


@lru_cache(maxsize=16)
def load_memory_bank(path: str, *, max_card_chars: int = 2200) -> Tau3MemoryBank:
    memory_path = Path(path).expanduser()
    if not memory_path.exists():
        raise FileNotFoundError(f"Tau3 SDPO memory path does not exist: {memory_path}")

    rows: list[dict[str, Any]] = []
    if memory_path.suffix.lower() == ".json":
        payload = json.loads(memory_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict) and isinstance(payload.get("cards"), list):
            rows = [row for row in payload["cards"] if isinstance(row, dict)]
        else:
            raise ValueError(f"Unsupported memory JSON shape in {memory_path}")
    else:
        with memory_path.open(encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)

    cards = [_load_one_card(row, idx, max_card_chars=max_card_chars) for idx, row in enumerate(rows)]
    return Tau3MemoryBank(cards)


def render_memory_section(card: Tau3MemoryCard, *, template: str | None = None) -> str:
    template = template or (
        "\n\nRelevant train-only correction memory:\n"
        "{memory_card}\n\n"
        "Use this memory only as decision guidance. Do not copy IDs, hidden facts, or raw wording."
    )
    return template.format(memory_card=card.display_text, memory_card_id=card.card_id)
