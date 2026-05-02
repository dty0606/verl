from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"
_DIGITS = "0123456789"

_PAYMENT_ID_RE = re.compile(r"\b(?:gift_card|certificate|credit_card)_\d+\b")
_USER_ID_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+_\d{4}\b")
_FLIGHT_ID_RE = re.compile(r"\b[A-Z]{3}\d{3}\b")
_RESERVATION_ID_RE = re.compile(r"\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b")


@dataclass(frozen=True)
class _Match:
    kind: str
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class Tau3IdTransformResult:
    value: Any
    mapping: dict[str, str]


def _is_identifier_char(ch: str) -> bool:
    return ch.isalpha() or ch.isdigit() or ch == "_"


def _overlaps(existing: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < other_end and other_start < end for other_start, other_end in existing)


def _charclass_signature(value: str) -> str:
    signature: list[str] = []
    for ch in value:
        if ch.isupper():
            signature.append("U")
        elif ch.islower():
            signature.append("L")
        elif ch.isdigit():
            signature.append("D")
        else:
            signature.append(ch)
    return "".join(signature)


def _deterministic_bytes(*parts: object) -> bytes:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _fill_template_from_digest(template: str, digest: bytes) -> str:
    out: list[str] = []
    digest_idx = 0
    for ch in template:
        if ch.isupper():
            out.append(_UPPER[digest[digest_idx % len(digest)] % len(_UPPER)])
            digest_idx += 1
        elif ch.islower():
            out.append(_LOWER[digest[digest_idx % len(digest)] % len(_LOWER)])
            digest_idx += 1
        elif ch.isdigit():
            out.append(_DIGITS[digest[digest_idx % len(digest)] % len(_DIGITS)])
            digest_idx += 1
        else:
            out.append(ch)
    return "".join(out)


def _fill_template_from_counter(template: str, counter: int) -> str:
    out = list(template)
    for idx in range(len(out) - 1, -1, -1):
        ch = out[idx]
        if ch.isupper():
            out[idx] = _UPPER[counter % len(_UPPER)]
            counter //= len(_UPPER)
        elif ch.islower():
            out[idx] = _LOWER[counter % len(_LOWER)]
            counter //= len(_LOWER)
        elif ch.isdigit():
            out[idx] = _DIGITS[counter % len(_DIGITS)]
            counter //= len(_DIGITS)
    return "".join(out)


def _preserve_non_id_text(original: str, replaced: str) -> bool:
    return len(original) == len(replaced) and _charclass_signature(original) == _charclass_signature(replaced)


def _normalize_payment_id(original: str, suffix: str) -> str:
    prefix, _ = original.rsplit("_", 1)
    return f"{prefix}_{suffix}"


def _normalize_flight_id(original: str, suffix: str) -> str:
    prefix = original[:3]
    return f"{prefix}{suffix}"


def _find_matches(text: str) -> list[_Match]:
    matches: list[_Match] = []
    occupied: list[tuple[int, int]] = []
    ordered_patterns = (
        ("payment", _PAYMENT_ID_RE),
        ("user", _USER_ID_RE),
        ("flight", _FLIGHT_ID_RE),
        ("reservation", _RESERVATION_ID_RE),
    )
    for kind, pattern in ordered_patterns:
        for match in pattern.finditer(text):
            start, end = match.span()
            if _overlaps(occupied, start, end):
                continue
            if start > 0 and _is_identifier_char(text[start - 1]):
                continue
            if end < len(text) and _is_identifier_char(text[end]):
                continue
            matches.append(_Match(kind=kind, value=match.group(0), start=start, end=end))
            occupied.append((start, end))
    return sorted(matches, key=lambda item: item.start)


class Tau3IdRandomizer:
    """Deterministically randomize tau3 IDs while keeping formats stable."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._replacements: dict[tuple[str, str], str] = {}
        self._used_replacements: dict[str, tuple[str, str]] = {}

    @property
    def replacements(self) -> dict[str, str]:
        return {original: replacement for (_, original), replacement in self._replacements.items()}

    def randomize(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.randomize_text(value)
        if isinstance(value, list):
            return [self.randomize(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.randomize(item) for item in value)
        if isinstance(value, dict):
            return {
                self.randomize_text(key) if isinstance(key, str) else key: self.randomize(item)
                for key, item in value.items()
            }
        return value

    def randomize_text(self, text: str) -> str:
        matches = _find_matches(text)
        if not matches:
            return text

        out: list[str] = []
        cursor = 0
        for match in matches:
            out.append(text[cursor : match.start])
            out.append(self._replacement_for(match.kind, match.value))
            cursor = match.end
        out.append(text[cursor:])
        return "".join(out)

    def _replacement_for(self, kind: str, original: str) -> str:
        key = (kind, original)
        cached = self._replacements.get(key)
        if cached is not None:
            return cached

        attempt = 0
        replacement = original
        while replacement == original or replacement in self._used_replacements:
            replacement = self._candidate_replacement(kind, original, attempt)
            attempt += 1

        self._replacements[key] = replacement
        self._used_replacements[replacement] = key
        return replacement

    def transform(self, value: Any) -> Tau3IdTransformResult:
        transformed = self.randomize(value)
        return Tau3IdTransformResult(value=transformed, mapping=self.replacements)

    def _candidate_replacement(self, kind: str, original: str, attempt: int) -> str:
        digest = _deterministic_bytes("tau3", self.seed, kind, original, attempt)

        if kind == "payment":
            digits = _fill_template_from_digest(original.rsplit("_", 1)[1], digest)
            return _normalize_payment_id(original, digits)
        if kind == "flight":
            digits = _fill_template_from_digest(original[3:], digest)
            return _normalize_flight_id(original, digits)
        replacement = _fill_template_from_digest(original, digest)
        if not _preserve_non_id_text(original, replacement):
            raise AssertionError(f"format drift for {original} -> {replacement}")
        return replacement


def collect_tau3_ids(value: Any) -> set[str]:
    ids_by_kind = _collect_ids(value)
    flattened: set[str] = set()
    for values in ids_by_kind.values():
        flattened.update(values)
    return flattened


def build_tau3_id_mapping(value: Any, seed: int = 0) -> dict[str, str]:
    return Tau3IdRandomizer(seed=seed).transform(value).mapping


def randomize_tau3_ids(value: Any, seed: int = 0) -> Tau3IdTransformResult:
    """Return a randomized copy of a nested transcript-like structure plus the ID mapping."""

    return Tau3IdRandomizer(seed=seed).transform(value)


def _collect_ids(value: Any) -> dict[str, set[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    if isinstance(value, str):
        for match in _find_matches(value):
            found[match.kind].add(match.value)
        return found
    if isinstance(value, list | tuple):
        for item in value:
            nested = _collect_ids(item)
            for kind, values in nested.items():
                found[kind].update(values)
        return found
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                nested = _collect_ids(key)
                for kind, values in nested.items():
                    found[kind].update(values)
            nested = _collect_ids(item)
            for kind, values in nested.items():
                found[kind].update(values)
    return found


def _canonical_replacement(kind: str, original: str, counter: int) -> str:
    if kind == "payment":
        digits = str(counter).zfill(len(original.rsplit("_", 1)[1]))[-len(original.rsplit("_", 1)[1]) :]
        return _normalize_payment_id(original, digits)
    if kind == "flight":
        digits = str(counter).zfill(len(original[3:]))[-len(original[3:]) :]
        return _normalize_flight_id(original, digits)
    replacement = _fill_template_from_counter(original, counter)
    if not _preserve_non_id_text(original, replacement):
        raise AssertionError(f"format drift for {original} -> {replacement}")
    return replacement


def canonicalize_tau3_ids(value: Any) -> Tau3IdTransformResult:
    """Return a deterministic canonicalized copy of a nested transcript-like structure plus the ID mapping."""

    ids_by_kind = _collect_ids(value)
    mapping: dict[tuple[str, str], str] = {}
    used_replacements: set[str] = set()
    counters: dict[str, int] = defaultdict(int)

    for kind in sorted(ids_by_kind):
        for original in sorted(ids_by_kind[kind]):
            replacement = original
            while replacement == original or replacement in used_replacements:
                replacement = _canonical_replacement(kind, original, counters[kind])
                counters[kind] += 1
            mapping[(kind, original)] = replacement
            used_replacements.add(replacement)

    def _apply(item: Any) -> Any:
        if isinstance(item, str):
            matches = _find_matches(item)
            if not matches:
                return item
            out: list[str] = []
            cursor = 0
            for match in matches:
                out.append(item[cursor : match.start])
                out.append(mapping[(match.kind, match.value)])
                cursor = match.end
            out.append(item[cursor:])
            return "".join(out)
        if isinstance(item, list):
            return [_apply(value) for value in item]
        if isinstance(item, tuple):
            return tuple(_apply(value) for value in item)
        if isinstance(item, dict):
            return {
                _apply(key) if isinstance(key, str) else key: _apply(value)
                for key, value in item.items()
            }
        return item

    flattened_mapping = {original: replacement for (_, original), replacement in mapping.items()}
    return Tau3IdTransformResult(value=_apply(value), mapping=flattened_mapping)


__all__ = [
    "Tau3IdRandomizer",
    "Tau3IdTransformResult",
    "build_tau3_id_mapping",
    "canonicalize_tau3_ids",
    "collect_tau3_ids",
    "randomize_tau3_ids",
]
