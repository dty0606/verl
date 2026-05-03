#!/usr/bin/env python3
"""Pre-tokenize turn-per-row SFT parquet for fast VERL training.

Reads the turn-format parquet (messages + answer), applies Qwen3.5 chat
template, and writes input_ids + loss_mask directly into a new parquet.
Training then uses the standard VERL SFT path with pre-tokenized data
instead of on-the-fly template rendering.

Usage:
    python3 scripts/tau3/pretokenize_turn_sft.py \
        --input datasets/tau3_sft_thinking_train_only \
        --output datasets/tau3_sft_thinking_pretokenized \
        --model Qwen/Qwen3.5-4B \
        --max-length 32768 \
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - local lightweight fallback
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_METADATA_COLUMNS = [
    "task_id",
    "sample_id",
    "source_path",
    "source_index",
    "id_variant",
    "assistant_turn_index",
    "source_message_index",
    "target_kind",
    "target_tool_name",
    "turn_bucket",
    "curation_stratum",
    "curation_selected_split",
]


def _tokenize_row(row_dict: dict, tokenizer_path: str, max_length: int, truncation: str) -> dict | None:
    """Tokenize a single turn-per-row SFT example. Runs in worker process."""
    # Lazy import so each worker loads its own tokenizer
    from transformers import AutoTokenizer
    from verl.utils.dataset.turn_sft_dataset import (
        _answer_to_message,
        _as_token_list,
        normalize_messages_for_qwen_template,
        normalize_tool_call,
        normalize_tools_for_qwen_template,
    )
    from verl.utils.py_functional import convert_nested_value_to_list_recursive
    from verl.utils.tokenizer import normalize_token_ids

    # Cache tokenizer per worker process
    if not hasattr(_tokenize_row, "_tokenizer"):
        _tokenize_row._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer = _tokenize_row._tokenizer

    messages_raw = convert_nested_value_to_list_recursive(row_dict["messages"])
    answer_raw = convert_nested_value_to_list_recursive(row_dict["answer"])
    tools_raw = row_dict.get("tools")
    if tools_raw is not None:
        tools_raw = convert_nested_value_to_list_recursive(tools_raw)
        tools = normalize_tools_for_qwen_template(tools_raw)
    else:
        tools = None

    enable_thinking = row_dict.get("enable_thinking", True)
    if enable_thinking is not None:
        enable_thinking = bool(enable_thinking)

    messages = normalize_messages_for_qwen_template(messages_raw)
    answer = _answer_to_message(answer_raw)

    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    full_messages = [*messages, answer]

    # Try nested tool-call format first, then flat
    def render(msgs, add_gen_prompt=False):
        ids = tokenizer.apply_chat_template(
            msgs, tools=tools, add_generation_prompt=add_gen_prompt,
            tokenize=True, **template_kwargs,
        )
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return normalize_token_ids(ids)

    try:
        full_ids = render(full_messages)
        prompt_ids = render(messages, add_gen_prompt=True)
    except Exception:
        # Try flat tool-call format
        from verl.utils.dataset.turn_sft_dataset import flatten_message_tool_calls
        flat_messages = flatten_message_tool_calls(full_messages)
        flat_prompt = flatten_message_tool_calls([*messages])
        try:
            full_ids = render(flat_messages)
            prompt_ids = render([*flat_prompt, {"role": "user", "content": ""}] if not flat_prompt else flat_prompt, add_gen_prompt=True)
            # Re-render prompt properly
            prompt_ids = render(flat_messages[:-1], add_gen_prompt=True)
        except Exception as e:
            return {"error": str(e)}

    prompt_len = len(prompt_ids)
    loss_mask = [0] * min(prompt_len, len(full_ids)) + [1] * max(0, len(full_ids) - prompt_len)
    loss_mask = loss_mask[:len(full_ids)]

    original_seq_len = len(full_ids)
    truncated = False
    if original_seq_len > max_length:
        if truncation == "error":
            return {"error": f"Sequence length {original_seq_len} > max_length {max_length}"}
        elif truncation == "right":
            full_ids = full_ids[:max_length]
            loss_mask = loss_mask[:max_length]
            truncated = True
        elif truncation == "left":
            full_ids = full_ids[-max_length:]
            loss_mask = loss_mask[-max_length:]
            truncated = True
        else:
            return {"error": f"Unknown truncation method: {truncation}"}

    if len(full_ids) != len(loss_mask):
        return {"error": f"input_ids/loss_mask length mismatch: {len(full_ids)} vs {len(loss_mask)}"}
    if loss_mask and loss_mask[0]:
        # VERL SFT shifts the flattened loss mask by one token. A label on the
        # first token of a jagged sample can cross sample boundaries, so drop it.
        loss_mask[0] = 0
    labeled_tokens = int(sum(loss_mask))
    if labeled_tokens <= 0:
        return {
            "error": (
                "Empty loss mask after tokenization/truncation. "
                f"original_seq_len={original_seq_len}, max_length={max_length}, truncation={truncation}"
            )
        }

    return {
        "input_ids": full_ids,
        "loss_mask": loss_mask,
        "seq_len": len(full_ids),
        "original_seq_len": original_seq_len,
        "labeled_tokens": labeled_tokens,
        "truncated": truncated,
    }


def pretokenize_split(
    input_path: Path,
    output_path: Path,
    tokenizer_path: str,
    max_length: int,
    truncation: str,
    workers: int,
    metadata_columns: list[str],
) -> dict:
    df = pd.read_parquet(input_path)
    rows = df.to_dict(orient="records")
    preserved_metadata_columns = [column for column in metadata_columns if column in df.columns]

    results = []
    errors = 0

    # Process rows — use multiprocessing for speed
    if workers > 1:
        worker_args = ((row, tokenizer_path, max_length, truncation) for row in rows)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            mapped = executor.map(_tokenize_row_from_args, worker_args, chunksize=16)
            for idx, result in enumerate(tqdm(mapped, total=len(rows), desc=str(input_path.name))):
                if result is None or "error" in result:
                    errors += 1
                    continue
                results.append((idx, result))
    else:
        for i, row in enumerate(tqdm(rows, desc=str(input_path.name))):
            result = _tokenize_row(row, tokenizer_path, max_length, truncation)
            if result is None or "error" in result:
                errors += 1
                continue
            results.append((i, result))

    # Sort by original index
    results.sort(key=lambda x: x[0])

    # Build output dataframe
    out_rows = []
    for idx, result in results:
        out_row = {
            "input_ids": result["input_ids"],
            "loss_mask": result["loss_mask"],
        }
        for column in preserved_metadata_columns:
            out_row[column] = rows[idx].get(column)
        out_rows.append(out_row)

    out_df = pd.DataFrame(out_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    stats = {
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "errors": errors,
        "avg_seq_len": np.mean([r[1]["seq_len"] for r in results]) if results else 0,
        "avg_labeled_tokens": np.mean([r[1]["labeled_tokens"] for r in results]) if results else 0,
        "max_seq_len": max(r[1]["seq_len"] for r in results) if results else 0,
        "max_original_seq_len": max(r[1]["original_seq_len"] for r in results) if results else 0,
        "truncated_rows": sum(1 for _, result in results if result.get("truncated")),
        "preserved_metadata_columns": preserved_metadata_columns,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input dataset dir with train.parquet/test.parquet")
    parser.add_argument("--output", required=True, help="Output dir for pre-tokenized parquet")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Tokenizer model path")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument(
        "--truncation",
        default="left",
        choices=["right", "left", "error"],
        help="Use left truncation to preserve the current assistant answer at the end of the sequence.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-errors", action="store_true", help="Write output even if some rows fail tokenization.")
    parser.add_argument(
        "--metadata-columns",
        nargs="*",
        default=DEFAULT_METADATA_COLUMNS,
        help="Optional row metadata columns to preserve in the pre-tokenized parquet for audits/debugging.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in ["train", "test"]:
        input_path = input_dir / f"{split}.parquet"
        if not input_path.exists():
            print(f"Skipping {split}: {input_path} not found")
            continue
        output_path = output_dir / f"{split}.parquet"
        print(f"\n=== Pre-tokenizing {split} ===")
        stats = pretokenize_split(
            input_path,
            output_path,
            args.model,
            args.max_length,
            args.truncation,
            args.workers,
            args.metadata_columns,
        )
        all_stats[split] = stats
        print(f"  {stats}")
        if stats["errors"] and not args.allow_errors:
            output_path.unlink(missing_ok=True)
            raise SystemExit(
                f"Pre-tokenization failed for {stats['errors']} {split} row(s). "
                "Inspect the input data or rerun with --allow-errors only for debugging."
            )

    manifest = {
        "source": str(input_dir),
        "model": args.model,
        "max_length": args.max_length,
        "truncation": args.truncation,
        **all_stats,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest: {output_dir / 'manifest.json'}")
    print("Done.")


def _tokenize_row_from_args(args):
    return _tokenize_row(*args)


if __name__ == "__main__":
    main()
