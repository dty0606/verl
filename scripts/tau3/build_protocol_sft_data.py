from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


try:
    from verl.utils.tau3_data import (  # noqa: E402
        canonicalize_tau3_ids,
        extract_messages_for_sft,
        get_openai_tool_schemas,
        randomize_tau3_ids,
        validate_sft_candidate,
    )
except (ImportError, ModuleNotFoundError):
    import types

    verl_pkg = sys.modules.setdefault("verl", types.ModuleType("verl"))
    if not hasattr(verl_pkg, "__path__"):
        verl_pkg.__path__ = [str(ROOT / "verl")]

    utils_pkg = sys.modules.setdefault("verl.utils", types.ModuleType("verl.utils"))
    if not hasattr(utils_pkg, "__path__"):
        utils_pkg.__path__ = [str(ROOT / "verl" / "utils")]
    setattr(verl_pkg, "utils", utils_pkg)

    tau3_pkg = sys.modules.setdefault("verl.utils.tau3_data", types.ModuleType("verl.utils.tau3_data"))
    if not hasattr(tau3_pkg, "__path__"):
        tau3_pkg.__path__ = [str(ROOT / "verl" / "utils" / "tau3_data")]
    setattr(utils_pkg, "tau3_data", tau3_pkg)

    schema_mod = _load_module_from_path(
        "verl.utils.tau3_tool_schema_export",
        ROOT / "verl" / "utils" / "tau3_tool_schema_export.py",
    )
    setattr(utils_pkg, "tau3_tool_schema_export", schema_mod)

    id_mod = _load_module_from_path(
        "verl.utils.tau3_data.id_randomizer",
        ROOT / "verl" / "utils" / "tau3_data" / "id_randomizer.py",
    )
    val_mod = _load_module_from_path(
        "verl.utils.tau3_data.validators",
        ROOT / "verl" / "utils" / "tau3_data" / "validators.py",
    )

    canonicalize_tau3_ids = id_mod.canonicalize_tau3_ids
    extract_messages_for_sft = val_mod.extract_messages_for_sft
    get_openai_tool_schemas = val_mod.get_openai_tool_schemas
    randomize_tau3_ids = id_mod.randomize_tau3_ids
    validate_sft_candidate = val_mod.validate_sft_candidate
    setattr(tau3_pkg, "canonicalize_tau3_ids", canonicalize_tau3_ids)
    setattr(tau3_pkg, "extract_messages_for_sft", extract_messages_for_sft)
    setattr(tau3_pkg, "get_openai_tool_schemas", get_openai_tool_schemas)
    setattr(tau3_pkg, "randomize_tau3_ids", randomize_tau3_ids)
    setattr(tau3_pkg, "validate_sft_candidate", validate_sft_candidate)


@dataclass(slots=True)
class CandidateRecord:
    payload: dict[str, Any]
    source_path: str
    source_index: int


def _iter_input_files(inputs: list[str], *, trajectory_only: bool = True) -> list[Path]:
    """Collect input files from paths, globs, or directories.

    When *trajectory_only* is True (default) and a directory is given, only
    files under ``*/trajectories/`` subdirectories are collected. This avoids
    scanning summary.json, run_config.json, and other non-trajectory files
    that would be rejected by the validator anyway.
    """
    paths: list[Path] = []
    for raw_input in inputs:
        if any(token in raw_input for token in "*?[]"):
            for matched in sorted(glob.glob(raw_input, recursive=True)):
                path = Path(matched).expanduser().resolve()
                if path.is_file():
                    paths.append(path)
            continue

        path = Path(raw_input).expanduser().resolve()
        if path.is_dir():
            if trajectory_only:
                # Only scan trajectory subdirectories
                for traj_dir in sorted(path.rglob("trajectories")):
                    if traj_dir.is_dir():
                        paths.extend(sorted(traj_dir.glob("*.json")))
                        paths.extend(sorted(traj_dir.glob("*.jsonl")))
            else:
                paths.extend(sorted(path.rglob("*.json")))
                paths.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {raw_input}")

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def _load_records_from_path(path: Path) -> list[CandidateRecord]:
    if path.suffix == ".jsonl":
        records: list[CandidateRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    records.append(CandidateRecord(payload=payload, source_path=str(path), source_index=line_index))
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [CandidateRecord(payload=payload, source_path=str(path), source_index=0)]
    if isinstance(payload, list):
        return [
            CandidateRecord(payload=item, source_path=str(path), source_index=index)
            for index, item in enumerate(payload)
            if isinstance(item, dict)
        ]
    return []


def _record_key(record: CandidateRecord) -> str:
    task_id = record.payload.get("task_id", "unknown")
    sample_id = record.payload.get("sample_id", record.source_index)
    return f"{record.source_path}::task={task_id}::sample={sample_id}::idx={record.source_index}"


def _bool_enable_thinking(payload: dict[str, Any]) -> bool:
    # Check top-level enable_thinking (set by Bedrock generation scripts)
    top_level = payload.get("enable_thinking")
    if isinstance(top_level, bool):
        return top_level
    if isinstance(top_level, str) and top_level.strip().lower() in ("1", "true", "yes"):
        return True
    chat_kwargs = payload.get("chat_template_kwargs")
    if isinstance(chat_kwargs, dict) and "enable_thinking" in chat_kwargs:
        return bool(chat_kwargs.get("enable_thinking"))
    return str(payload.get("thinking_mode", "")).lower() == "on"


def _messages_contain_reasoning_traces(messages: list[dict[str, Any]]) -> bool:
    return bool(re.search(r"<think\b|</think>", json.dumps(messages, ensure_ascii=False), flags=re.IGNORECASE))


def _extract_assistant_thinking_content(content: Any) -> tuple[str, str]:
    """Split a leading ``<think>...</think>`` block from assistant content."""
    if not isinstance(content, str):
        return "", "" if content is None else str(content)

    match = re.match(r"\s*<think>\s*(.*?)\s*</think>\s*(.*)\Z", content, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return "", content
    return match.group(1).strip(), match.group(2)


def _strip_assistant_thinking(message: dict[str, Any]) -> dict[str, Any]:
    """Remove historical reasoning from assistant messages used as context."""
    copied = {k: v for k, v in message.items() if v is not None}
    if copied.get("role") != "assistant":
        return copied
    thinking, remainder = _extract_assistant_thinking_content(copied.get("content"))
    if thinking:
        copied["content"] = remainder
    copied.pop("thinking", None)
    copied.pop("reasoning", None)
    return copied


def _assistant_message_to_answer(message: dict[str, Any]) -> dict[str, Any]:
    """Convert one assistant message into an AReaL-style SFT target answer."""
    content = message.get("content")
    thinking, content_without_thinking = _extract_assistant_thinking_content(content)
    answer: dict[str, Any] = {
        "role": "assistant",
        "content": content_without_thinking or "",
        "thinking": thinking,
    }
    if message.get("tool_calls"):
        answer["tool_calls"] = message["tool_calls"]
    return answer


def _expand_turn_sft_rows(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand one full trajectory into one row per non-greeting assistant turn."""
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    seen_non_system = False

    for message_index, message in enumerate(messages):
        role = message.get("role")

        # Tau3 traces include an assistant greeting before the first user turn.
        # It is not a policy target and Qwen templates require user-first
        # conversation structure after the optional system prompt.
        is_leading_assistant_greeting = (
            role == "assistant" and not seen_non_system and not message.get("tool_calls")
        )
        if role != "system":
            seen_non_system = True
        if is_leading_assistant_greeting:
            continue

        if role == "assistant":
            answer = _assistant_message_to_answer(message)
            if answer.get("thinking") or answer.get("content") or answer.get("tool_calls"):
                rows.append(
                    {
                        "messages": [_strip_assistant_thinking(history_message) for history_message in history],
                        "answer": answer,
                        "source_message_index": message_index,
                        "assistant_turn_index": len(rows),
                    }
                )

        history.append(_strip_assistant_thinking(message))

    return rows


_FIRST_ASSISTANT_USER_ECHO_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("my_user_id", re.compile(r"\bmy user id\b", re.IGNORECASE)),
    ("my_name_is", re.compile(r"\bmy name is\b", re.IGNORECASE)),
    ("i_am_named_user", re.compile(r"\bi am [A-Z][A-Za-z'-]+ [A-Z][A-Za-z'-]+\b")),
)


def _first_assistant_user_echo_details(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Detect reward-successful traces where the assistant starts by speaking as the user.

    The tau reward can still pass these trajectories if later tool actions solve
    the task, but they are bad imitation targets for protocol SFT. Keep this
    filter narrow: only reject the first assistant turn when it is plain text
    and contains explicit user/persona self-identification.
    """

    turns = payload.get("turns")
    if not isinstance(turns, list) or not turns or not isinstance(turns[0], dict):
        return None

    first_turn = turns[0]
    parsed = first_turn.get("parsed") or {}
    if not isinstance(parsed, dict) or parsed.get("action_type") != "plain_text":
        return None

    text = str(first_turn.get("raw_model_output") or "")
    matched = [name for name, pattern in _FIRST_ASSISTANT_USER_ECHO_PATTERNS if pattern.search(text)]
    if not matched:
        return None

    return {
        "reason": "first_assistant_user_echo",
        "turn_index": 0,
        "action_type": parsed.get("action_type"),
        "matched_patterns": matched,
        "raw_preview": text[:320],
    }


def _apply_id_transform(messages: list[dict[str, Any]], *, transform: str, seed: int, record_key: str) -> list[dict[str, Any]]:
    if transform == "none":
        return messages
    if transform == "canonicalize":
        return canonicalize_tau3_ids(messages).value
    if transform == "randomize":
        record_seed = int(hashlib.sha256(f"{seed}|{record_key}".encode("utf-8")).hexdigest()[:16], 16)
        return randomize_tau3_ids(messages, seed=record_seed).value
    raise ValueError(f"Unsupported id transform: {transform}")


def _id_variant_specs(*, id_transform: str, id_variants: int, include_original: bool) -> list[tuple[str, str, int]]:
    if id_variants < 1:
        raise ValueError(f"id_variants must be >= 1; got {id_variants}")

    if id_transform != "randomize" and id_variants != 1:
        raise ValueError("--id-variants > 1 is only supported with --id-transform randomize.")
    if include_original and id_transform != "randomize":
        raise ValueError("--include-original-id-variant is only supported with --id-transform randomize.")

    specs: list[tuple[str, str, int]] = []
    if include_original:
        specs.append(("original", "none", 0))

    for variant_index in range(id_variants):
        specs.append((f"{id_transform}_{variant_index}", id_transform, variant_index))
    return specs


def _stable_task_partition(task_ids: list[str], *, val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    if len(task_ids) < 2:
        raise ValueError("Need at least two unique task ids to create an automatic train/test split.")

    ranked = []
    for task_id in sorted(set(task_ids)):
        digest = hashlib.sha256(f"{seed}|{task_id}".encode("utf-8")).hexdigest()
        ranked.append((int(digest[:16], 16), task_id))
    ranked.sort()

    test_count = max(1, min(len(ranked) - 1, int(round(len(ranked) * val_fraction))))
    test_ids = {task_id for _, task_id in ranked[:test_count]}
    train_ids = {task_id for _, task_id in ranked[test_count:]}
    return train_ids, test_ids


def _stable_row_holdout(rows: list[dict[str, Any]], *, val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}")
    if len(rows) < 2:
        raise ValueError("Need at least two accepted rows to create a train-only validation holdout.")

    task_ids = sorted({str(row["task_id"]) for row in rows})
    if len(task_ids) >= 2:
        train_task_ids, eval_task_ids = _stable_task_partition(task_ids, val_fraction=val_fraction, seed=seed)
        train_rows = [row for row in rows if str(row["task_id"]) in train_task_ids]
        eval_rows = [row for row in rows if str(row["task_id"]) in eval_task_ids]
        return train_rows, eval_rows

    ranked: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        key = f"{seed}|{row.get('task_id')}|{row.get('sample_id')}|{row.get('id_variant')}|{index}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        ranked.append((int(digest[:16], 16), index))
    ranked.sort()
    eval_count = max(1, min(len(rows) - 1, int(round(len(rows) * val_fraction))))
    eval_indices = {index for _, index in ranked[:eval_count]}
    train_rows = [row for index, row in enumerate(rows) if index not in eval_indices]
    eval_rows = [row for index, row in enumerate(rows) if index in eval_indices]
    return train_rows, eval_rows


def _load_split_manifest(path: Path) -> tuple[set[str], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    train_ids = {str(task_id) for task_id in payload.get("train_task_ids", [])}
    test_ids = {str(task_id) for task_id in payload.get("test_task_ids", [])}
    if not train_ids or not test_ids:
        raise ValueError("Split manifest must contain non-empty train_task_ids and test_task_ids.")
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Split manifest contains overlapping train/test ids: {sorted(overlap)}")
    return train_ids, test_ids


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def build_protocol_sft_dataset(
    *,
    input_paths: list[str],
    output_dir: str,
    domain: str = "airline",
    id_transform: str = "none",
    id_seed: int = 0,
    id_variants: int = 1,
    include_original_id_variant: bool = False,
    split_manifest: str | None = None,
    val_fraction: float = 0.1,
    train_only_holdout: bool = False,
    allow_canonical_test_validation: bool = False,
    filter_first_assistant_user_echo: bool = True,
    include_system_prompt: bool = True,
    include_thinking_traces: bool = False,
    force_enable_thinking: bool = False,
    sft_format: str = "trajectory",
    max_rows_per_task: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if sft_format not in {"trajectory", "turn"}:
        raise ValueError(f"Unsupported sft_format: {sft_format}")
    if max_rows_per_task is not None and max_rows_per_task < 1:
        raise ValueError(f"max_rows_per_task must be >= 1 when set; got {max_rows_per_task}")

    out_dir = Path(output_dir).expanduser().resolve()
    if out_dir.exists() and not overwrite and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    input_files = _iter_input_files(input_paths)
    tool_schemas = get_openai_tool_schemas(domain)
    id_variant_specs = _id_variant_specs(
        id_transform=id_transform,
        id_variants=id_variants,
        include_original=include_original_id_variant,
    )

    accepted_rows: list[dict[str, Any]] = []
    accepted_audit_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for input_file in input_files:
        for record in _load_records_from_path(input_file):
            key = _record_key(record)
            validation = validate_sft_candidate(record.payload, domain=domain)
            base_metadata = {
                "source_path": record.source_path,
                "source_index": record.source_index,
                "task_id": str(record.payload.get("task_id", "")),
                "sample_id": record.payload.get("sample_id"),
                "model": record.payload.get("model"),
                "domain": record.payload.get("domain", domain),
                "task_split": record.payload.get("task_split"),
                "final_reward": record.payload.get("final_reward"),
                "validation_reason": validation.reason,
            }

            if not validation.accept:
                rejected_rows.append({**base_metadata, "accept": False, "details": validation.details})
                continue

            if filter_first_assistant_user_echo:
                user_echo_details = _first_assistant_user_echo_details(record.payload)
                if user_echo_details is not None:
                    rejected_rows.append(
                        {
                            **base_metadata,
                            "accept": False,
                            "validation_reason": "first_assistant_user_echo",
                            "details": user_echo_details,
                        }
                    )
                    continue

            try:
                base_messages = extract_messages_for_sft(
                    record.payload,
                    include_system_prompt=include_system_prompt,
                    include_thinking_traces=include_thinking_traces,
                )
            except Exception as exc:  # noqa: BLE001
                rejected_rows.append(
                    {
                        **base_metadata,
                        "accept": False,
                        "validation_reason": "builder_extract_failed",
                        "details": {"error": f"{type(exc).__name__}: {exc}"},
                    }
                )
                continue

            source_enable_thinking = _bool_enable_thinking(record.payload)

            for id_variant, variant_transform, variant_index in id_variant_specs:
                try:
                    messages = _apply_id_transform(
                        base_messages,
                        transform=variant_transform,
                        seed=id_seed + variant_index,
                        record_key=f"{key}::variant={id_variant}",
                    )
                    contains_reasoning_traces = _messages_contain_reasoning_traces(messages)
                    if include_thinking_traces and not contains_reasoning_traces:
                        rejected_rows.append(
                            {
                                **base_metadata,
                                "accept": False,
                                "validation_reason": "missing_thinking_traces",
                                "details": {
                                    "id_variant": id_variant,
                                    "source_enable_thinking": source_enable_thinking,
                                    "force_enable_thinking": force_enable_thinking,
                                },
                            }
                        )
                        continue

                    row_enable_thinking = bool(
                        force_enable_thinking or contains_reasoning_traces or source_enable_thinking
                    )
                    thinking_supervision_mode = (
                        "source_thinking_traces"
                        if include_thinking_traces and contains_reasoning_traces
                        else (
                            "template_override_only"
                            if force_enable_thinking
                            else ("source_thinking_metadata" if source_enable_thinking else "non_thinking")
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    rejected_rows.append(
                        {
                            **base_metadata,
                            "accept": False,
                            "validation_reason": "builder_id_transform_failed",
                            "details": {
                                "id_variant": id_variant,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        }
                    )
                    continue

                common_audit_metadata = {
                    "source_path": record.source_path,
                    "source_index": record.source_index,
                    "task_id": str(record.payload.get("task_id", "")),
                    "sample_id": record.payload.get("sample_id"),
                    "model": record.payload.get("model"),
                    "domain": record.payload.get("domain", domain),
                    "task_split": record.payload.get("task_split"),
                    "final_reward": record.payload.get("final_reward"),
                    "validation_reason": validation.reason,
                    "validation_details": validation.details,
                    "tool_names": validation.tool_names,
                    "id_transform": variant_transform,
                    "id_variant": id_variant,
                    "source_enable_thinking": source_enable_thinking,
                    "force_enable_thinking": force_enable_thinking,
                    "chat_template_enable_thinking": row_enable_thinking,
                    "thinking_supervision_mode": thinking_supervision_mode,
                    "contains_reasoning_traces": contains_reasoning_traces,
                }

                base_row = {
                    "tools": tool_schemas,
                    "enable_thinking": row_enable_thinking,
                    "source_path": record.source_path,
                    "source_index": record.source_index,
                    "task_id": str(record.payload.get("task_id", "")),
                    "sample_id": record.payload.get("sample_id"),
                    "model": record.payload.get("model"),
                    "domain": record.payload.get("domain", domain),
                    "task_split": record.payload.get("task_split"),
                    "final_reward": record.payload.get("final_reward"),
                    "request_id": record.payload.get("request_id"),
                    "validation_reason": validation.reason,
                    "id_transform": variant_transform,
                    "id_variant": id_variant,
                }

                emitted_rows: list[dict[str, Any]]
                if sft_format == "trajectory":
                    emitted_rows = [
                        {
                            **base_row,
                            "messages": [{k: v for k, v in m.items() if v is not None} for m in messages],
                            "audit_metadata": {**common_audit_metadata, "sft_format": sft_format},
                        }
                    ]
                else:
                    turn_rows = _expand_turn_sft_rows([{k: v for k, v in m.items() if v is not None} for m in messages])
                    if not turn_rows:
                        rejected_rows.append(
                            {
                                **base_metadata,
                                "accept": False,
                                "validation_reason": "no_turn_sft_rows",
                                "details": {"id_variant": id_variant},
                            }
                        )
                        continue
                    emitted_rows = []
                    for turn_row in turn_rows:
                        emitted_rows.append(
                            {
                                **base_row,
                                "messages": turn_row["messages"],
                                "answer": turn_row["answer"],
                                "assistant_turn_index": turn_row["assistant_turn_index"],
                                "source_message_index": turn_row["source_message_index"],
                                "audit_metadata": {
                                    **common_audit_metadata,
                                    "sft_format": sft_format,
                                    "assistant_turn_index": turn_row["assistant_turn_index"],
                                    "source_message_index": turn_row["source_message_index"],
                                },
                            }
                        )

                accepted_rows.extend(emitted_rows)
                for emitted_row in emitted_rows:
                    accepted_audit_rows.append(
                        {
                        **base_metadata,
                        "accept": True,
                        "message_count": len(emitted_row["messages"]),
                        "tool_names": validation.tool_names,
                        "id_transform": variant_transform,
                        "id_variant": id_variant,
                        "sft_format": sft_format,
                        "assistant_turn_index": emitted_row.get("assistant_turn_index"),
                        "source_message_index": emitted_row.get("source_message_index"),
                        "source_enable_thinking": source_enable_thinking,
                        "force_enable_thinking": force_enable_thinking,
                        "chat_template_enable_thinking": row_enable_thinking,
                        "thinking_supervision_mode": thinking_supervision_mode,
                        "contains_reasoning_traces": contains_reasoning_traces,
                        "details": validation.details,
                        }
                    )

    if not accepted_rows:
        raise RuntimeError("No valid tau3 success trajectories were accepted.")

    if split_manifest:
        train_ids, test_ids = _load_split_manifest(Path(split_manifest).expanduser().resolve())
        split_mode = "manifest"
    else:
        train_ids, test_ids = _stable_task_partition(
            [str(row["task_id"]) for row in accepted_rows],
            val_fraction=val_fraction,
            seed=id_seed,
        )
        split_mode = "hashed_task_id"

    train_rows = [row for row in accepted_rows if str(row["task_id"]) in train_ids]
    test_rows = [row for row in accepted_rows if str(row["task_id"]) in test_ids]
    held_out_rows = [row for row in accepted_rows if str(row["task_id"]) not in train_ids | test_ids]

    if split_manifest and test_rows and not allow_canonical_test_validation:
        raise RuntimeError(
            f"Refusing to use {len(test_rows)} canonical test-task trajectory row(s) as SFT validation. "
            "Build SFT data from train-split trajectories and pass --train-only-holdout, or pass "
            "--allow-canonical-test-validation only for non-claim/debug runs."
        )

    if held_out_rows:
        rejected_rows.extend(
            {
                "source_path": row["source_path"],
                "source_index": row["source_index"],
                "task_id": row["task_id"],
                "sample_id": row["sample_id"],
                "model": row["model"],
                "domain": row["domain"],
                "task_split": row["task_split"],
                "final_reward": row["final_reward"],
                "accept": False,
                "validation_reason": "not_in_requested_split",
                "details": {},
            }
            for row in held_out_rows
        )

    if not train_rows or not test_rows:
        if split_manifest and train_only_holdout and train_rows and not test_rows:
            train_rows, test_rows = _stable_row_holdout(train_rows, val_fraction=val_fraction, seed=id_seed)
            split_mode = "manifest_train_only_local_holdout"
            train_ids = {str(row["task_id"]) for row in train_rows}
            test_ids = {str(row["task_id"]) for row in test_rows}
        else:
            raise RuntimeError(
                f"Split produced empty partition(s): train={len(train_rows)} test={len(test_rows)}. "
                "Provide a split manifest with accepted task ids, more accepted task ids, or set "
                "--train-only-holdout when intentionally building from train-split-only trajectories."
            )

    if max_rows_per_task is not None:
        def _cap_rows_per_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                grouped.setdefault(str(row["task_id"]), []).append(row)
            capped_rows: list[dict[str, Any]] = []
            for task_id, task_rows in sorted(grouped.items()):
                ranked_rows = sorted(
                    task_rows,
                    key=lambda row: hashlib.sha256(
                        (
                            f"{id_seed}|{task_id}|{row.get('source_path')}|{row.get('source_index')}|"
                            f"{row.get('sample_id')}|{row.get('id_variant')}|{row.get('assistant_turn_index', '')}"
                        ).encode("utf-8")
                    ).hexdigest(),
                )
                capped_rows.extend(ranked_rows[:max_rows_per_task])
            return capped_rows

        train_rows = _cap_rows_per_task(train_rows)
        test_rows = _cap_rows_per_task(test_rows)

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(test_rows).to_parquet(test_path, index=False)

    accepted_jsonl_path = out_dir / "accepted.jsonl"
    rejected_jsonl_path = out_dir / "rejected.jsonl"
    _write_jsonl(accepted_jsonl_path, accepted_audit_rows)
    _write_jsonl(rejected_jsonl_path, rejected_rows)

    manifest = {
        "manifest_version": 1,
        "dataset_type": "tau3_protocol_sft",
        "sft_format": sft_format,
        "domain": domain,
        "id_transform": id_transform,
        "id_seed": id_seed,
        "id_variants": id_variants,
        "include_original_id_variant": include_original_id_variant,
        "include_system_prompt": include_system_prompt,
        "dataset_config": {
            "messages_key": "messages",
            "answer_key": "answer" if sft_format == "turn" else None,
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
        },
        "split_mode": split_mode,
        "split_manifest": split_manifest,
        "val_fraction": val_fraction,
        "train_only_holdout": train_only_holdout,
        "allow_canonical_test_validation": allow_canonical_test_validation,
        "filter_first_assistant_user_echo": filter_first_assistant_user_echo,
        "include_thinking_traces": include_thinking_traces,
        "force_enable_thinking": force_enable_thinking,
        "max_rows_per_task": max_rows_per_task,
        "thinking_supervision_note": (
            "enable_thinking controls Qwen chat-template rendering. "
            "--include-thinking-traces stitches turns[].thinking_text into assistant messages as <think> blocks. "
            "--force-enable-thinking does not add supervised reasoning traces."
        ),
        "accepted_rows_with_source_enable_thinking": sum(
            1 for row in accepted_rows if row["audit_metadata"]["source_enable_thinking"]
        ),
        "accepted_rows_with_forced_enable_thinking": sum(
            1
            for row in accepted_rows
            if row["audit_metadata"]["force_enable_thinking"]
            and not row["audit_metadata"]["source_enable_thinking"]
        ),
        "accepted_rows_containing_reasoning_traces": sum(
            1 for row in accepted_rows if row["audit_metadata"]["contains_reasoning_traces"]
        ),
        "accepted_rows_per_task": dict(sorted(Counter(str(row["task_id"]) for row in accepted_rows).items())),
        "train_rows_per_task": dict(sorted(Counter(str(row["task_id"]) for row in train_rows).items())),
        "test_rows_per_task": dict(sorted(Counter(str(row["task_id"]) for row in test_rows).items())),
        "thinking_supervision_modes": dict(
            sorted(Counter(row["audit_metadata"]["thinking_supervision_mode"] for row in accepted_rows).items())
        ),
        "input_files": [str(path) for path in input_files],
        "train_parquet": str(train_path),
        "test_parquet": str(test_path),
        "parquet_files": [str(train_path), str(test_path)],
        "accepted_jsonl": str(accepted_jsonl_path),
        "rejected_jsonl": str(rejected_jsonl_path),
        "accepted_rows": len(accepted_rows),
        "rejected_rows": len(rejected_rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_task_ids": sorted(train_ids),
        "test_task_ids": sorted(test_ids),
        "rejected_reasons": dict(sorted(Counter(row["validation_reason"] for row in rejected_rows).items())),
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build verified tau3 protocol SFT parquet datasets from structured trajectory artifacts."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Trajectory file, directory, or glob. Repeatable.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write train/test parquet and manifests.")
    parser.add_argument("--domain", default="airline")
    parser.add_argument(
        "--id-transform",
        choices=["none", "randomize", "canonicalize"],
        default="none",
        help="Optional ID normalization applied to emitted messages.",
    )
    parser.add_argument("--id-seed", type=int, default=0)
    parser.add_argument(
        "--id-variants",
        type=int,
        default=1,
        help="Number of deterministic same-format ID-randomized variants per accepted row. "
        "Only valid with --id-transform randomize.",
    )
    parser.add_argument(
        "--include-original-id-variant",
        action="store_true",
        help="With --id-transform randomize, also include one unrandomized copy of each accepted row.",
    )
    parser.add_argument(
        "--split-manifest",
        default=str(ROOT / "datasets" / "tau3_live_airline_canonical_split.json"),
        help="Explicit train/test task-id manifest. Empty string disables canonical split loading.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Task-level held-out fraction when --split-manifest is not provided.",
    )
    parser.add_argument(
        "--train-only-holdout",
        action="store_true",
        help=(
            "Allow a manifest-filtered train-only candidate root and create a local validation holdout "
            "from train-split successes. This never uses canonical test-task trajectories."
        ),
    )
    parser.add_argument(
        "--allow-canonical-test-validation",
        action="store_true",
        help=(
            "Permit manifest test-task trajectories to become test.parquet. "
            "Use only for smoke/debug runs; final SFT should use train-only trajectories plus --train-only-holdout."
        ),
    )
    parser.add_argument(
        "--allow-first-assistant-user-echo",
        action="store_true",
        help=(
            "Do not reject successful trajectories whose first assistant turn appears to echo the user persona. "
            "Default is to filter these out because they are bad protocol-SFT imitation targets."
        ),
    )
    parser.add_argument(
        "--include-thinking-traces",
        action="store_true",
        help=(
            "Stitch turns[].thinking_text from trajectory data into assistant messages as "
            "<think>...</think> blocks. Requires trajectories generated with thinking enabled. "
            "Rows without stitched reasoning traces are rejected."
        ),
    )
    parser.add_argument(
        "--force-enable-thinking",
        action="store_true",
        help=(
            "Emit enable_thinking=True for all accepted parquet rows regardless of source rollout metadata. "
            "This re-renders verified action trajectories under a thinking-on Qwen chat template, but does not "
            "add supervised reasoning traces."
        ),
    )
    parser.add_argument(
        "--sft-format",
        choices=["trajectory", "turn"],
        default="trajectory",
        help=(
            "trajectory emits one full multi-turn row. turn emits one row per assistant target with "
            "prior history in messages and current target in answer, matching AReaL tau2-style SFT."
        ),
    )
    parser.add_argument(
        "--max-rows-per-task",
        type=int,
        default=None,
        help=(
            "Optional deterministic cap applied after train/test split, useful for balancing turn-expanded "
            "SFT rows across tau3 task ids."
        ),
    )
    parser.add_argument("--omit-system-prompt", action="store_true", help="Do not prepend the stored system prompt.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    args = parser.parse_args()

    manifest = build_protocol_sft_dataset(
        input_paths=args.input,
        output_dir=args.output_dir,
        domain=args.domain,
        id_transform=args.id_transform,
        id_seed=args.id_seed,
        id_variants=args.id_variants,
        include_original_id_variant=args.include_original_id_variant,
        split_manifest=args.split_manifest or None,
        val_fraction=args.val_fraction,
        train_only_holdout=args.train_only_holdout,
        allow_canonical_test_validation=args.allow_canonical_test_validation,
        filter_first_assistant_user_echo=not args.allow_first_assistant_user_echo,
        include_system_prompt=not args.omit_system_prompt,
        include_thinking_traces=args.include_thinking_traces,
        force_enable_thinking=args.force_enable_thinking,
        sft_format=args.sft_format,
        max_rows_per_task=args.max_rows_per_task,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
