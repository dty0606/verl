#!/usr/bin/env python3
"""Analyze token budget for full-trajectory Tau3 SFT/RL contexts.

This script reads pre-tokenized full-trajectory SFT parquet files produced by
``pretokenize_full_traj_sft.py``. It uses the stored ``segments`` metadata for
exact role/message/turn boundaries, then optionally decodes assistant spans to
estimate how much of each assistant turn is thinking, tool-call markup, or
user-facing text.

The exact segment stats answer "how long is the rollout context?". The decoded
component estimates answer "what is consuming the assistant tokens?".
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call\b[^>]*>.*?</tool_call>", re.IGNORECASE | re.DOTALL)


def _to_builtin(value: Any) -> Any:
    """Convert pandas/pyarrow/numpy nested values into plain Python values."""

    if value is None:
        return None
    if hasattr(value, "as_py"):
        return _to_builtin(value.as_py())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _to_builtin(value.tolist())
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    # Pandas may represent absent nested scalars as nan. Leave normal numbers
    # alone, but normalize NaN to None for cleaner JSON.
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _as_int_list(value: Any) -> list[int]:
    value = _to_builtin(value)
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _as_segments(value: Any) -> list[dict[str, Any]]:
    value = _to_builtin(value)
    if not isinstance(value, list):
        return []
    segments = []
    for item in value:
        if isinstance(item, dict):
            segments.append(item)
    return segments


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def _stats(values: list[int | float]) -> dict[str, float | int]:
    clean = [float(value) for value in values if value is not None]
    clean.sort()
    if not clean:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "count": len(clean),
        "mean": float(sum(clean) / len(clean)),
        "p50": _percentile(clean, 50),
        "p90": _percentile(clean, 90),
        "p95": _percentile(clean, 95),
        "p99": _percentile(clean, 99),
        "max": int(max(clean)),
    }


def _safe_segment_len(segment: dict[str, Any]) -> int:
    return max(0, int(segment.get("token_end") or 0) - int(segment.get("token_start") or 0))


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def _count_piece_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(len(ids))


def _estimate_assistant_components(tokenizer, input_ids: list[int], segment: dict[str, Any]) -> dict[str, int]:
    start = int(segment.get("token_start") or 0)
    end = int(segment.get("token_end") or 0)
    segment_ids = input_ids[start:end]
    total = len(segment_ids)
    if total <= 0:
        return {"assistant_total": 0, "thinking": 0, "tool_call": 0, "visible_text": 0, "template_overhead": 0}

    decoded = tokenizer.decode(segment_ids, skip_special_tokens=False)
    thinking_text = "".join(match.group(0) for match in THINK_RE.finditer(decoded))
    tool_call_text = "".join(match.group(0) for match in TOOL_CALL_RE.finditer(decoded))
    thinking_tokens = _count_piece_tokens(tokenizer, thinking_text)
    tool_call_tokens = _count_piece_tokens(tokenizer, tool_call_text)

    # Remove explicit thinking/tool markup to estimate visible assistant text.
    visible_text = TOOL_CALL_RE.sub("", THINK_RE.sub("", decoded))
    visible_tokens = _count_piece_tokens(tokenizer, visible_text.strip())
    overhead = max(0, total - thinking_tokens - tool_call_tokens - visible_tokens)
    return {
        "assistant_total": total,
        "thinking": thinking_tokens,
        "tool_call": tool_call_tokens,
        "visible_text": visible_tokens,
        "template_overhead": overhead,
    }


def _read_split(dataset_dir: Path, split: str) -> pd.DataFrame:
    path = dataset_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_parquet(path)


def analyze_split(
    df: pd.DataFrame,
    *,
    thresholds: list[int],
    tokenizer=None,
    component_sample_rows: int = 0,
) -> dict[str, Any]:
    seq_lens: list[int] = []
    labeled_tokens: list[int] = []
    assistant_counts: list[int] = []
    exact_by_role: Counter[str] = Counter()
    exact_by_segment_type: Counter[str] = Counter()
    assistant_prompt_lens: list[int] = []
    assistant_response_lens: list[int] = []
    assistant_turn_index_stats: dict[int, dict[str, list[int]]] = defaultdict(lambda: {"prompt": [], "response": []})
    row_first_over_threshold: dict[int, list[int | None]] = {threshold: [] for threshold in thresholds}
    component_totals: Counter[str] = Counter()
    component_rows_used = 0
    max_prompt_len_by_row: list[int] = []

    for row_index, row in df.iterrows():
        input_ids = _as_int_list(row.get("input_ids"))
        segments = _as_segments(row.get("segments"))
        seq_len = int(row.get("seq_len") or len(input_ids))
        seq_lens.append(seq_len)
        labeled_tokens.append(int(row.get("labeled_tokens") or sum(_as_int_list(row.get("loss_mask")))))
        assistant_counts.append(int(row.get("assistant_count") or 0))

        assistant_token_starts: list[tuple[int, int]] = []
        row_max_prompt = 0
        for segment in segments:
            length = _safe_segment_len(segment)
            role = str(segment.get("role") or "unknown")
            segment_type = str(segment.get("segment_type") or role or "unknown")
            exact_by_role[role] += length
            exact_by_segment_type[segment_type] += length

            if role == "assistant":
                turn_index = int(segment.get("assistant_turn_index") or 0)
                prompt_len = int(segment.get("token_start") or 0)
                response_len = length
                assistant_prompt_lens.append(prompt_len)
                assistant_response_lens.append(response_len)
                assistant_turn_index_stats[turn_index]["prompt"].append(prompt_len)
                assistant_turn_index_stats[turn_index]["response"].append(response_len)
                assistant_token_starts.append((turn_index, prompt_len))
                row_max_prompt = max(row_max_prompt, prompt_len)

                if tokenizer is not None and component_rows_used < component_sample_rows:
                    components = _estimate_assistant_components(tokenizer, input_ids, segment)
                    component_totals.update(components)

        if tokenizer is not None and component_rows_used < component_sample_rows:
            component_rows_used += 1
        max_prompt_len_by_row.append(row_max_prompt)

        for threshold in thresholds:
            first_turn = None
            for turn_index, prompt_len in sorted(assistant_token_starts):
                if prompt_len > threshold:
                    first_turn = turn_index
                    break
            row_first_over_threshold[threshold].append(first_turn)

    row_thresholds = {}
    for threshold in thresholds:
        rows_over = sum(1 for value in seq_lens if value > threshold)
        turns_over = sum(1 for value in assistant_prompt_lens if value > threshold)
        first_turn_values = [value for value in row_first_over_threshold[threshold] if value is not None]
        row_thresholds[str(threshold)] = {
            "rows_with_total_seq_len_over_threshold": rows_over,
            "row_fraction_total_seq_len_over_threshold": rows_over / max(1, len(seq_lens)),
            "assistant_turns_with_prompt_len_over_threshold": turns_over,
            "assistant_turn_fraction_prompt_len_over_threshold": turns_over / max(1, len(assistant_prompt_lens)),
            "rows_where_any_assistant_prompt_exceeds_threshold": len(first_turn_values),
            "first_exceeding_assistant_turn_index_stats": _stats(first_turn_values),
        }

    turn_stats = {}
    for turn_index, values in sorted(assistant_turn_index_stats.items()):
        turn_stats[str(turn_index)] = {
            "prompt_len_before_turn": _stats(values["prompt"]),
            "assistant_response_len": _stats(values["response"]),
        }

    exact_total_tokens = sum(exact_by_segment_type.values()) or 1
    exact_segment_budget = {
        key: {
            "tokens": int(value),
            "fraction": float(value / exact_total_tokens),
        }
        for key, value in sorted(exact_by_segment_type.items())
    }
    exact_role_budget = {
        key: {
            "tokens": int(value),
            "fraction": float(value / exact_total_tokens),
        }
        for key, value in sorted(exact_by_role.items())
    }

    component_estimates = None
    if tokenizer is not None and component_rows_used:
        total = int(component_totals.get("assistant_total", 0)) or 1
        component_estimates = {
            "sampled_rows": component_rows_used,
            "note": "Decoded assistant-span estimate. Exact role/turn boundaries come from segments; subcomponent counts are approximate because they are re-tokenized from decoded substrings.",
            "tokens": {key: int(value) for key, value in sorted(component_totals.items())},
            "fractions_of_sampled_assistant_tokens": {
                key: float(value / total)
                for key, value in sorted(component_totals.items())
                if key != "assistant_total"
            },
        }

    return {
        "rows": int(len(df)),
        "seq_len": _stats(seq_lens),
        "labeled_tokens": _stats(labeled_tokens),
        "assistant_count": _stats(assistant_counts),
        "max_prompt_len_before_assistant_by_row": _stats(max_prompt_len_by_row),
        "assistant_prompt_len_before_turn": _stats(assistant_prompt_lens),
        "assistant_response_len": _stats(assistant_response_lens),
        "exact_role_budget": exact_role_budget,
        "exact_segment_budget": exact_segment_budget,
        "component_estimates": component_estimates,
        "thresholds": row_thresholds,
        "by_assistant_turn_index": turn_stats,
    }


def _print_summary(split: str, result: dict[str, Any]) -> None:
    print(f"\n=== {split} ===")
    print(
        "rows={rows} seq_len mean={mean:.0f} p90={p90:.0f} p95={p95:.0f} max={max}".format(
            rows=result["rows"],
            mean=result["seq_len"]["mean"],
            p90=result["seq_len"]["p90"],
            p95=result["seq_len"]["p95"],
            max=result["seq_len"]["max"],
        )
    )
    print(
        "assistant_count mean={mean:.2f} p90={p90:.0f} max={max}; prompt_before_assistant p90={prompt_p90:.0f} max={prompt_max}".format(
            mean=result["assistant_count"]["mean"],
            p90=result["assistant_count"]["p90"],
            max=result["assistant_count"]["max"],
            prompt_p90=result["assistant_prompt_len_before_turn"]["p90"],
            prompt_max=result["assistant_prompt_len_before_turn"]["max"],
        )
    )
    print("exact segment budget:")
    for key, value in result["exact_segment_budget"].items():
        print(f"  {key}: {value['tokens']} tokens ({value['fraction']:.1%})")

    if result.get("component_estimates"):
        print("assistant component estimate:")
        fractions = result["component_estimates"]["fractions_of_sampled_assistant_tokens"]
        for key, value in fractions.items():
            print(f"  {key}: {value:.1%}")

    print("threshold risks:")
    for threshold, value in result["thresholds"].items():
        print(
            "  {threshold}: seq>{threshold} rows={rows:.1%}; assistant prompt>{threshold} turns={turns:.1%}; rows any prompt>{threshold}={row_any:.1%}".format(
                threshold=threshold,
                rows=value["row_fraction_total_seq_len_over_threshold"],
                turns=value["assistant_turn_fraction_prompt_len_over_threshold"],
                row_any=value["rows_where_any_assistant_prompt_exceeds_threshold"] / max(1, result["rows"]),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Pretokenized full-traj dataset directory")
    parser.add_argument("--split", default="train", choices=["train", "test", "both"])
    parser.add_argument("--model", default=None, help="Optional tokenizer path for assistant component estimates")
    parser.add_argument("--component-sample-rows", type=int, default=1000)
    parser.add_argument("--thresholds", nargs="*", type=int, default=[4096, 8192, 16384, 24576, 32768])
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).expanduser()
    tokenizer = None
    if args.model:
        tokenizer = _load_tokenizer(args.model)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    output = {
        "dataset": str(dataset_dir),
        "model": args.model,
        "component_sample_rows": args.component_sample_rows if tokenizer is not None else 0,
        "thresholds": args.thresholds,
        "splits": {},
    }
    for split in splits:
        df = _read_split(dataset_dir, split)
        result = analyze_split(
            df,
            thresholds=args.thresholds,
            tokenizer=tokenizer,
            component_sample_rows=max(0, args.component_sample_rows),
        )
        output["splits"][split] = result
        _print_summary(split, result)

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {output_path}")
    else:
        print("\nJSON output omitted. Use --output-json to save the full report.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
