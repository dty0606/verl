#!/usr/bin/env python3
"""Audit Tau3 SFT parquet rows against the Qwen3.5 chat template.

This is a pre-flight check for SimpleSFTDataset. It renders real rows,
normalizes tool-call arguments to mappings, builds assistant loss masks, and
fails if masked spans do not contain expected thinking/tool-call content.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.utils.dataset.simple_sft_dataset import SimpleSFTDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="Directory containing train.parquet and test.parquet, or a parquet file.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split to inspect.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Tokenizer/model path.")
    parser.add_argument("--max-rows", type=int, default=20, help="Number of rows to audit.")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length used by the dataset.")
    parser.add_argument("--audit-max-chars", type=int, default=800, help="Max decoded chars per assistant span.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_dir).expanduser()
    if dataset_path.is_dir():
        dataset_path = dataset_path / f"{args.split}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = OmegaConf.create(
        {
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": True,
            "pad_mode": "no_padding",
            "max_length": args.max_length,
            "truncation": "error",
            "audit_max_chars": args.audit_max_chars,
        }
    )
    dataset = SimpleSFTDataset(
        parquet_files=[str(dataset_path)],
        tokenizer=tokenizer,
        config=config,
        processor=None,
        max_samples=args.max_rows,
    )

    failures = 0
    for index in range(len(dataset)):
        try:
            report = dataset.audit_item(index)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(json.dumps({"item": index, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
            continue
        if report["failures"]:
            failures += 1
        print(json.dumps(report, ensure_ascii=False))

    if failures:
        raise SystemExit(f"Template audit failed for {failures}/{len(dataset)} row(s).")
    print(f"Template audit passed for {len(dataset)} row(s).")


if __name__ == "__main__":
    main()
