#!/usr/bin/env python3
"""Pre-tokenize full-trajectory Tau3 thinking SFT parquet.

Input rows are produced by ``build_protocol_sft_data.py --sft-format trajectory``:
one successful multi-turn trajectory per row with assistant thinking stitched
into message content. This script patches the Qwen3.5 tokenizer chat template
in-memory so historical assistant ``<think>`` blocks are preserved, then writes
``input_ids`` and ``loss_mask`` for ``PretokenizedSFTDataset``.

The default truncation mode is ``error``. Full trajectories are meant to carry
causal history, so overlength rows should be inspected rather than silently
cropped during the first smoke.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

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
    "domain",
    "task_split",
    "final_reward",
    "validation_reason",
]


def _contiguous_spans(mask: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _extract_tool_name(tool_call: dict[str, Any]) -> str | None:
    function_entry = tool_call.get("function")
    if isinstance(function_entry, dict):
        return str(function_entry.get("name") or "") or None
    return str(tool_call.get("name") or "") or None


def _tokenize_row(row_dict: dict, tokenizer_path: str, max_length: int, truncation: str, audit: bool) -> dict:
    """Tokenize one full-trajectory row. Runs inside worker processes."""

    from transformers import AutoTokenizer
    from verl.utils.dataset.qwen35_preserve_thinking_template import (
        install_preserve_thinking_chat_template,
        preserve_thinking_template_sha256,
    )
    from verl.utils.dataset.simple_sft_dataset import (
        flatten_message_tool_calls,
        normalize_messages_for_qwen_template,
        normalize_tools_for_qwen_template,
    )
    from verl.utils.py_functional import convert_nested_value_to_list_recursive
    from verl.utils.tokenizer import normalize_token_ids

    if not hasattr(_tokenize_row, "_tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        install_preserve_thinking_chat_template(tokenizer)
        _tokenize_row._tokenizer = tokenizer
    tokenizer = _tokenize_row._tokenizer

    messages_raw = convert_nested_value_to_list_recursive(row_dict["messages"])
    messages = normalize_messages_for_qwen_template(messages_raw)
    tools_raw = row_dict.get("tools")
    tools = None
    if tools_raw is not None:
        tools = normalize_tools_for_qwen_template(convert_nested_value_to_list_recursive(tools_raw))

    enable_thinking = row_dict.get("enable_thinking", True)
    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = bool(enable_thinking)

    def render(msgs, *, add_generation_prompt=False) -> list[int]:
        ids = tokenizer.apply_chat_template(
            msgs,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            **template_kwargs,
        )
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return normalize_token_ids(ids)

    template_shape = "nested"
    try:
        full_ids = render(messages)
    except Exception:
        messages = flatten_message_tool_calls(messages)
        template_shape = "flat"
        full_ids = render(messages)

    assistant_indices = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    loss_mask = [0] * len(full_ids)
    for assistant_index in assistant_indices:
        prefix_messages = messages[:assistant_index]
        prefix_len = len(render(prefix_messages, add_generation_prompt=True)) if prefix_messages else 0
        suffix_len = len(render(messages[: assistant_index + 1], add_generation_prompt=False))
        for token_index in range(prefix_len, min(suffix_len, len(loss_mask))):
            loss_mask[token_index] = 1

    original_seq_len = len(full_ids)
    truncated = False
    if original_seq_len > max_length:
        if truncation == "error":
            return {"error": f"Sequence length {original_seq_len} > max_length {max_length}"}
        if truncation == "right":
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
        loss_mask[0] = 0
    labeled_tokens = int(sum(loss_mask))
    if labeled_tokens <= 0:
        return {
            "error": (
                "Empty loss mask after tokenization/truncation. "
                f"original_seq_len={original_seq_len}, max_length={max_length}, truncation={truncation}"
            )
        }

    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    audit_report = None
    if audit:
        spans = _contiguous_spans(loss_mask)
        decoded_spans = [
            tokenizer.decode(full_ids[start:end], skip_special_tokens=False)[:600] for start, end in spans
        ]
        failures: list[str] = []
        if not truncated and len(spans) != len(assistant_messages):
            failures.append(f"assistant_span_count_mismatch: spans={len(spans)} assistants={len(assistant_messages)}")
        for span_index, message in enumerate(assistant_messages[: len(decoded_spans)]):
            decoded = decoded_spans[span_index]
            content = str(message.get("content") or "")
            if "<think" in content and "<think" not in decoded:
                failures.append(f"missing_think_in_span_{span_index}")
            for tool_call in message.get("tool_calls") or []:
                tool_name = _extract_tool_name(tool_call)
                if tool_name and tool_name not in decoded:
                    failures.append(f"missing_tool_name_in_span_{span_index}: {tool_name}")
        audit_report = {
            "assistant_count": len(assistant_messages),
            "masked_span_count": len(spans),
            "labeled_tokens": labeled_tokens,
            "decoded_spans": decoded_spans,
            "failures": failures,
        }
        if failures:
            return {"error": f"Audit failed: {failures}", "audit_report": audit_report}

    return {
        "input_ids": full_ids,
        "loss_mask": loss_mask,
        "seq_len": len(full_ids),
        "original_seq_len": original_seq_len,
        "labeled_tokens": labeled_tokens,
        "assistant_count": len(assistant_messages),
        "truncated": truncated,
        "template_shape": template_shape,
        "chat_template_sha256": preserve_thinking_template_sha256(),
        "audit_report": audit_report,
    }


def _tokenize_row_from_args(args):
    return _tokenize_row(*args)


def pretokenize_split(
    input_path: Path,
    output_path: Path,
    tokenizer_path: str,
    max_length: int,
    truncation: str,
    workers: int,
    metadata_columns: list[str],
    audit_samples: int,
) -> dict:
    df = pd.read_parquet(input_path)
    rows = df.to_dict(orient="records")
    preserved_metadata_columns = [column for column in metadata_columns if column in df.columns]

    results: list[tuple[int, dict]] = []
    error_examples: list[dict[str, Any]] = []
    errors = 0

    if workers > 1:
        worker_args = (
            (row, tokenizer_path, max_length, truncation, index < audit_samples)
            for index, row in enumerate(rows)
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            mapped = executor.map(_tokenize_row_from_args, worker_args, chunksize=8)
            for index, result in enumerate(tqdm(mapped, total=len(rows), desc=str(input_path.name))):
                if result is None or "error" in result:
                    errors += 1
                    if len(error_examples) < 10:
                        error_examples.append({"index": index, "error": None if result is None else result["error"]})
                    continue
                results.append((index, result))
    else:
        for index, row in enumerate(tqdm(rows, desc=str(input_path.name))):
            result = _tokenize_row(row, tokenizer_path, max_length, truncation, index < audit_samples)
            if result is None or "error" in result:
                errors += 1
                if len(error_examples) < 10:
                    error_examples.append({"index": index, "error": None if result is None else result["error"]})
                continue
            results.append((index, result))

    results.sort(key=lambda item: item[0])

    out_rows = []
    audit_reports = []
    for index, result in results:
        out_row = {
            "input_ids": result["input_ids"],
            "loss_mask": result["loss_mask"],
            "seq_len": result["seq_len"],
            "original_seq_len": result["original_seq_len"],
            "labeled_tokens": result["labeled_tokens"],
            "assistant_count": result["assistant_count"],
            "truncated": result["truncated"],
            "template_shape": result["template_shape"],
        }
        for column in preserved_metadata_columns:
            out_row[column] = rows[index].get(column)
        out_rows.append(out_row)
        if result.get("audit_report") is not None:
            audit_reports.append({"index": index, **result["audit_report"]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(output_path, index=False)
    if audit_reports:
        audit_path = output_path.with_suffix(".audit.jsonl")
        audit_path.write_text(
            "".join(json.dumps(report, ensure_ascii=False) + "\n" for report in audit_reports),
            encoding="utf-8",
        )

    seq_lens = [result["seq_len"] for _, result in results]
    labeled_tokens = [result["labeled_tokens"] for _, result in results]
    assistant_counts = [result["assistant_count"] for _, result in results]
    template_shape_counts = {}
    if results:
        shape_series = pd.Series([result["template_shape"] for _, result in results]).value_counts()
        template_shape_counts = {str(key): int(value) for key, value in shape_series.items()}

    stats = {
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "errors": errors,
        "error_examples": error_examples,
        "avg_seq_len": float(np.mean(seq_lens)) if seq_lens else 0,
        "p90_seq_len": float(np.percentile(seq_lens, 90)) if seq_lens else 0,
        "p95_seq_len": float(np.percentile(seq_lens, 95)) if seq_lens else 0,
        "max_seq_len": int(max(seq_lens)) if seq_lens else 0,
        "max_original_seq_len": int(max(result["original_seq_len"] for _, result in results)) if results else 0,
        "avg_labeled_tokens": float(np.mean(labeled_tokens)) if labeled_tokens else 0,
        "avg_assistant_count": float(np.mean(assistant_counts)) if assistant_counts else 0,
        "truncated_rows": int(sum(1 for _, result in results if result.get("truncated"))),
        "template_shapes": template_shape_counts,
        "preserved_metadata_columns": preserved_metadata_columns,
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input trajectory-format dataset dir")
    parser.add_argument("--output", required=True, help="Output pre-tokenized dataset dir")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Tokenizer model path")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--truncation", default="error", choices=["error", "right", "left"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--audit-samples", type=int, default=3)
    parser.add_argument("--allow-errors", action="store_true", help="Write output even if some rows fail")
    parser.add_argument(
        "--metadata-columns",
        nargs="*",
        default=DEFAULT_METADATA_COLUMNS,
        help="Optional row metadata columns to preserve for audits/debugging.",
    )
    args = parser.parse_args()

    from verl.utils.dataset.qwen35_preserve_thinking_template import (
        preserve_thinking_template_sha256,
        write_preserve_thinking_template,
    )

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_preserve_thinking_template(output_dir / "chat_template_preserve_thinking.jinja")

    all_stats = {}
    for split in ["train", "test"]:
        input_path = input_dir / f"{split}.parquet"
        if not input_path.exists():
            print(f"Skipping {split}: {input_path} not found")
            continue
        output_path = output_dir / f"{split}.parquet"
        print(f"\n=== Pre-tokenizing full trajectories: {split} ===")
        stats = pretokenize_split(
            input_path=input_path,
            output_path=output_path,
            tokenizer_path=args.model,
            max_length=args.max_length,
            truncation=args.truncation,
            workers=args.workers,
            metadata_columns=args.metadata_columns,
            audit_samples=args.audit_samples,
        )
        all_stats[split] = stats
        print(f"  {stats}")
        if stats["errors"] and not args.allow_errors:
            output_path.unlink(missing_ok=True)
            raise SystemExit(
                f"Full-traj pre-tokenization failed for {stats['errors']} {split} row(s). "
                "Inspect manifest error_examples or rerun with --allow-errors only for debugging."
            )

    manifest = {
        "source": str(input_dir),
        "format": "full_trajectory",
        "model": args.model,
        "max_length": args.max_length,
        "truncation": args.truncation,
        "chat_template": "qwen35_preserve_historical_thinking",
        "chat_template_sha256": preserve_thinking_template_sha256(),
        **all_stats,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
