#!/usr/bin/env python3
"""Curate a balanced Tau3 turn-per-row SFT subset.

This script is intentionally separate from the trajectory builder. The builder
maximizes accepted data; this curator creates a smaller, reproducible training
slice for fast VLM-format SFT experiments.

Input format:
    A dataset directory with train.parquet and/or test.parquet emitted by
    scripts/tau3/build_protocol_sft_data.py --sft-format turn.

Output format:
    train.parquet, test.parquet, and manifest.json with the same nested SFT
    columns plus simple curation metadata columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SPLIT_MANIFEST = ROOT / "datasets" / "tau3_live_airline_canonical_split.json"


def _json_clean(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    return value


def _stable_digest(value: Any, *, seed: int, salt: str = "") -> str:
    payload = json.dumps(_json_clean(value), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(f"{seed}|{salt}|{payload}".encode("utf-8")).hexdigest()


def _norm_task_id(task_id: Any) -> str:
    text = "" if task_id is None else str(task_id).strip()
    match = re.search(r"(\d+)\Z", text)
    return match.group(1) if match else text


def _load_split_manifest(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    train_ids = {_norm_task_id(task_id) for task_id in payload.get("train_task_ids", [])}
    test_ids = {_norm_task_id(task_id) for task_id in payload.get("test_task_ids", [])}
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Split manifest contains train/test overlap: {sorted(overlap)}")
    return train_ids, test_ids


def _read_input_dataset(input_dir: Path, splits: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in splits:
        path = input_dir / f"{split}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame.copy()
        frame["_input_split"] = split
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No parquet splits found under {input_dir} for splits={splits}")
    return pd.concat(frames, ignore_index=True)


def _answer_dict(row: pd.Series) -> dict[str, Any]:
    answer = _json_clean(row.get("answer"))
    if not isinstance(answer, dict):
        return {}
    return answer


def _tool_name_from_call(tool_call: Any) -> str:
    call = _json_clean(tool_call)
    if not isinstance(call, dict):
        return "unknown_tool"
    if isinstance(call.get("name"), str):
        return call["name"]
    function = call.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    return "unknown_tool"


def _target_profile(row: pd.Series) -> dict[str, Any]:
    answer = _answer_dict(row)
    tool_calls = answer.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    content = answer.get("content") or ""
    thinking = answer.get("thinking") or answer.get("reasoning") or ""

    tool_name = "no_tool"
    if tool_calls:
        tool_name = _tool_name_from_call(tool_calls[0])

    if tool_calls and str(content).strip():
        target_kind = "tool_call_with_text"
    elif tool_calls:
        target_kind = "tool_call"
    elif str(content).strip():
        target_kind = "final_or_clarify_text"
    else:
        target_kind = "empty_text"

    try:
        turn_index = int(row.get("assistant_turn_index", -1))
    except (TypeError, ValueError):
        turn_index = -1
    if turn_index < 0:
        turn_bucket = "unknown"
    elif turn_index <= 1:
        turn_bucket = "early"
    elif turn_index <= 5:
        turn_bucket = "middle"
    else:
        turn_bucket = "late"

    return {
        "target_kind": target_kind,
        "target_tool_name": tool_name,
        "turn_bucket": turn_bucket,
        "has_thinking": bool(str(thinking).strip()),
        "assistant_turn_index_int": turn_index,
    }


def _source_turn_key(row: pd.Series) -> tuple[str, ...]:
    return (
        _norm_task_id(row.get("task_id")),
        str(row.get("source_path", "")),
        str(row.get("source_index", "")),
        str(row.get("sample_id", "")),
        str(row.get("assistant_turn_index", "")),
        str(row.get("source_message_index", "")),
    )


def _source_trajectory_key(row: pd.Series) -> tuple[str, ...]:
    return (
        _norm_task_id(row.get("task_id")),
        str(row.get("source_path", "")),
        str(row.get("source_index", "")),
        str(row.get("sample_id", "")),
        str(row.get("id_variant", "")),
    )


def _exact_row_key(row: pd.Series, *, seed: int) -> str:
    # Exact duplicate protection. ID-augmented variants are handled by
    # _source_turn_key, so this intentionally preserves literal IDs.
    return _stable_digest(
        {
            "task_id": _norm_task_id(row.get("task_id")),
            "messages": row.get("messages"),
            "answer": row.get("answer"),
        },
        seed=seed,
        salt="exact_row",
    )


def _selection_key(row: pd.Series, *, seed: int, salt: str) -> str:
    return _stable_digest(
        {
            "task_id": _norm_task_id(row.get("task_id")),
            "source_path": row.get("source_path"),
            "source_index": row.get("source_index"),
            "sample_id": row.get("sample_id"),
            "id_variant": row.get("id_variant"),
            "assistant_turn_index": row.get("assistant_turn_index"),
            "source_message_index": row.get("source_message_index"),
            "answer": row.get("answer"),
        },
        seed=seed,
        salt=salt,
    )


def _deduplicate_rows(df: pd.DataFrame, *, seed: int, max_variants_per_source_turn: int) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(df)

    ranked_indices: list[int] = []
    for _, group in df.groupby("_source_turn_key", sort=False):
        group = group.copy()
        group["_variant_rank"] = group.apply(
            lambda row: (
                0 if str(row.get("id_variant", "")).lower() == "original" else 1,
                _selection_key(row, seed=seed, salt="variant_rank"),
            ),
            axis=1,
        )
        ranked = group.sort_values("_variant_rank").head(max_variants_per_source_turn)
        ranked_indices.extend(ranked.index.tolist())

    df = df.loc[sorted(ranked_indices)].copy()
    after_variant_cap = len(df)
    df["_exact_row_key"] = df.apply(lambda row: _exact_row_key(row, seed=seed), axis=1)
    df = df.drop_duplicates("_exact_row_key", keep="first").copy()

    return df, {
        "input_rows": before,
        "dropped_by_source_turn_variant_cap": before - after_variant_cap,
        "dropped_by_exact_duplicate": after_variant_cap - len(df),
        "rows_after_dedup": len(df),
    }


def _round_robin_select(task_df: pd.DataFrame, *, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or task_df.empty:
        return task_df.iloc[0:0].copy()

    buckets: dict[str, list[int]] = {}
    for stratum, group in task_df.groupby("curation_stratum", sort=True):
        ranked = group.copy()
        ranked["_rank_key"] = ranked.apply(lambda row: _selection_key(row, seed=seed, salt=f"stratum:{stratum}"), axis=1)
        buckets[str(stratum)] = ranked.sort_values("_rank_key").index.tolist()

    selected: list[int] = []
    while len(selected) < count:
        progressed = False
        for stratum in sorted(buckets):
            indices = buckets[stratum]
            if not indices:
                continue
            selected.append(indices.pop(0))
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break

    return task_df.loc[selected].copy()


def _curate_task_rows(
    task_df: pd.DataFrame,
    *,
    train_rows_per_task: int,
    val_rows_per_task: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    desired = train_rows_per_task + val_rows_per_task
    selected = _round_robin_select(task_df, count=desired, seed=seed)

    if selected.empty:
        return selected.copy(), selected.copy(), {"available": len(task_df), "selected": 0, "shortfall": desired}

    selected["_split_rank_key"] = selected.apply(lambda row: _selection_key(row, seed=seed, salt="split"), axis=1)
    selected = selected.sort_values("_split_rank_key").copy()

    # Keep all turns from the same source trajectory on the same side of the
    # local train/validation split. This avoids validation leakage while still
    # allowing row-level task-balanced selection.
    val_indices: set[int] = set()
    if val_rows_per_task > 0 and len(selected) > train_rows_per_task:
        group_ranks: list[tuple[str, tuple[str, ...], list[int]]] = []
        for trajectory_key, group in selected.groupby("_source_trajectory_key", sort=False):
            rank = _stable_digest(trajectory_key, seed=seed, salt="trajectory_split")
            group_ranks.append((rank, trajectory_key, group.index.tolist()))
        for _, _, indices in sorted(group_ranks):
            if len(selected) - (len(val_indices) + len(indices)) < train_rows_per_task:
                continue
            val_indices.update(indices)
            if len(val_indices) >= val_rows_per_task:
                break

    val_df = selected.loc[sorted(val_indices)].copy()
    train_df = selected.loc[[idx for idx in selected.index if idx not in val_indices]].copy()

    if len(train_df) > train_rows_per_task:
        train_df = train_df.sort_values("_split_rank_key").head(train_rows_per_task).copy()

    for frame, split_name in [(train_df, "train"), (val_df, "test")]:
        frame["curation_selected_split"] = split_name

    return train_df, val_df, {
        "available": len(task_df),
        "selected": len(train_df) + len(val_df),
        "train_rows": len(train_df),
        "test_rows": len(val_df),
        "shortfall": max(0, desired - len(selected)),
    }


def _counter_dict(series: pd.Series) -> dict[str, int]:
    return dict(sorted(Counter(str(v) for v in series.tolist()).items()))


def curate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = input_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8")) if source_manifest_path.exists() else {}

    split_manifest_path = None if not args.split_manifest else Path(args.split_manifest).expanduser().resolve()
    canonical_train_ids, canonical_test_ids = _load_split_manifest(split_manifest_path)
    deny_task_ids = {_norm_task_id(task_id) for task_id in args.deny_task_ids}

    df = _read_input_dataset(input_dir, args.input_splits)
    required_columns = {"task_id", "messages", "answer"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Input dataset is missing required raw turn-SFT columns {missing_columns}. "
            "Run this curator before pretokenize_turn_sft.py, or pre-tokenize the curated output."
        )
    df = df.copy()
    df["task_id"] = df["task_id"].apply(_norm_task_id)
    profiles = df.apply(_target_profile, axis=1, result_type="expand")
    for column in profiles.columns:
        df[column] = profiles[column]
    df["_source_turn_key"] = df.apply(_source_turn_key, axis=1)
    df["_source_trajectory_key"] = df.apply(_source_trajectory_key, axis=1)
    df["curation_stratum"] = (
        df["target_kind"].astype(str)
        + "|"
        + df["target_tool_name"].astype(str)
        + "|"
        + df["turn_bucket"].astype(str)
        + "|thinking="
        + df["has_thinking"].astype(str)
    )

    initial_rows = len(df)
    dropped: dict[str, int] = {}

    if canonical_train_ids and args.restrict_to_canonical_train:
        mask = df["task_id"].isin(canonical_train_ids)
        if canonical_test_ids and not args.allow_canonical_test_rows:
            mask &= ~df["task_id"].isin(canonical_test_ids)
        if not args.allow_unknown_task_ids:
            mask &= df["task_id"].isin(canonical_train_ids)
        dropped["not_in_canonical_train"] = int((~mask).sum())
        df = df[mask].copy()
    elif canonical_test_ids and not args.allow_canonical_test_rows:
        mask = ~df["task_id"].isin(canonical_test_ids)
        dropped["canonical_test_task_rows"] = int((~mask).sum())
        df = df[mask].copy()

    if deny_task_ids:
        mask = ~df["task_id"].isin(deny_task_ids)
        dropped["denied_task_rows"] = int((~mask).sum())
        df = df[mask].copy()

    if "final_reward" in df.columns and args.require_final_reward is not None:
        rewards = pd.to_numeric(df["final_reward"], errors="coerce")
        mask = rewards >= float(args.require_final_reward)
        dropped["below_required_final_reward"] = int((~mask).sum())
        df = df[mask].copy()

    if args.require_thinking:
        mask = df["has_thinking"].astype(bool)
        dropped["missing_target_thinking"] = int((~mask).sum())
        df = df[mask].copy()

    if args.drop_empty_targets:
        mask = df["target_kind"] != "empty_text"
        dropped["empty_targets"] = int((~mask).sum())
        df = df[mask].copy()

    df, dedup_stats = _deduplicate_rows(
        df,
        seed=args.seed,
        max_variants_per_source_turn=args.max_id_variants_per_source_turn,
    )

    task_stats: dict[str, Any] = {}
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    for task_id, task_df in sorted(df.groupby("task_id", sort=True), key=lambda item: item[0]):
        train_df, test_df, stats = _curate_task_rows(
            task_df,
            train_rows_per_task=args.train_rows_per_task,
            val_rows_per_task=args.val_rows_per_task,
            seed=args.seed,
        )
        task_stats[str(task_id)] = stats
        train_frames.append(train_df)
        test_frames.append(test_df)

    train_out = pd.concat(train_frames, ignore_index=True) if train_frames else df.iloc[0:0].copy()
    test_out = pd.concat(test_frames, ignore_index=True) if test_frames else df.iloc[0:0].copy()

    helper_columns = [
        "_source_turn_key",
        "_source_trajectory_key",
        "_exact_row_key",
        "_split_rank_key",
        "_rank_key",
        "_variant_rank",
    ]
    for frame in [train_out, test_out]:
        for column in helper_columns:
            if column in frame.columns:
                frame.drop(columns=[column], inplace=True)

    if train_out.empty:
        raise RuntimeError("Curated train split is empty. Check denylist and split-manifest filters.")
    if test_out.empty and args.val_rows_per_task > 0:
        raise RuntimeError("Curated test split is empty. Set --val-rows-per-task 0 only for train-only debug runs.")

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    train_out.to_parquet(train_path, index=False)
    test_out.to_parquet(test_path, index=False)

    manifest = {
        "manifest_version": 1,
        "dataset_type": "tau3_turn_sft_curated_subset",
        "source": str(input_dir),
        "source_manifest": source_manifest,
        "input_splits": args.input_splits,
        "initial_rows": initial_rows,
        "rows_after_filters": len(df),
        "dropped_rows": dropped,
        "dedup": dedup_stats,
        "split_manifest": str(split_manifest_path) if split_manifest_path else None,
        "restrict_to_canonical_train": args.restrict_to_canonical_train,
        "allow_canonical_test_rows": args.allow_canonical_test_rows,
        "allow_unknown_task_ids": args.allow_unknown_task_ids,
        "deny_task_ids": sorted(deny_task_ids),
        "require_final_reward": args.require_final_reward,
        "require_thinking": args.require_thinking,
        "drop_empty_targets": args.drop_empty_targets,
        "target_train_rows_per_task": args.train_rows_per_task,
        "target_val_rows_per_task": args.val_rows_per_task,
        "seed": args.seed,
        "max_id_variants_per_source_turn": args.max_id_variants_per_source_turn,
        "train_rows": len(train_out),
        "test_rows": len(test_out),
        "train_rows_per_task": _counter_dict(train_out["task_id"]),
        "test_rows_per_task": _counter_dict(test_out["task_id"]) if not test_out.empty else {},
        "train_target_kind_counts": _counter_dict(train_out["target_kind"]),
        "test_target_kind_counts": _counter_dict(test_out["target_kind"]) if not test_out.empty else {},
        "train_tool_counts": _counter_dict(train_out["target_tool_name"]),
        "train_turn_bucket_counts": _counter_dict(train_out["turn_bucket"]),
        "train_stratum_counts": _counter_dict(train_out["curation_stratum"]),
        "task_stats": task_stats,
        "shortfall_tasks": {
            task_id: stats
            for task_id, stats in task_stats.items()
            if int(stats.get("shortfall", 0)) > 0 or int(stats.get("train_rows", 0)) < args.train_rows_per_task
        },
        "parquet_files": [str(train_path), str(test_path)],
        "notes": [
            "Rows are sampled deterministically from canonical train tasks unless explicitly overridden.",
            "Use this raw curated subset as input to pretokenize_turn_sft.py before VERL SFT.",
            "If task 39 is re-confirmed assertion-bad, rerun with --deny-task-ids 7 39.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input turn-SFT dataset dir with train/test parquet.")
    parser.add_argument("--output", required=True, help="Output curated dataset dir.")
    parser.add_argument(
        "--input-splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help=(
            "Input parquet splits to sample from. Default uses both train and local validation "
            "because build_protocol_sft_data.py may create a train-only holdout from canonical train tasks."
        ),
    )
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT_MANIFEST))
    parser.add_argument("--restrict-to-canonical-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-canonical-test-rows", action="store_true")
    parser.add_argument("--allow-unknown-task-ids", action="store_true")
    parser.add_argument("--deny-task-ids", nargs="*", default=[])
    parser.add_argument("--train-rows-per-task", type=int, default=170)
    parser.add_argument("--val-rows-per-task", type=int, default=20)
    parser.add_argument("--max-id-variants-per-source-turn", type=int, default=1)
    parser.add_argument("--require-final-reward", type=float, default=1.0)
    parser.add_argument("--require-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-empty-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_rows_per_task < 1:
        raise ValueError("--train-rows-per-task must be >= 1")
    if args.val_rows_per_task < 0:
        raise ValueError("--val-rows-per-task must be >= 0")
    if args.max_id_variants_per_source_turn < 1:
        raise ValueError("--max-id-variants-per-source-turn must be >= 1")

    manifest = curate_dataset(args)
    print(json.dumps({k: manifest[k] for k in ["train_rows", "test_rows", "train_rows_per_task", "shortfall_tasks"]}, indent=2))
    print(f"Manifest: {Path(args.output).expanduser().resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
