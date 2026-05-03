#!/usr/bin/env python3
"""Patch a HF tokenizer directory to preserve historical assistant thinking.

Use this after a VERL SFT checkpoint is exported if the training data was
pre-tokenized with the preserve-thinking template. The patch writes both
``tokenizer_config.json`` and ``chat_template.jinja`` so later vLLM/VERL rollout
loads see the same template contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_template_module():
    path = ROOT / "verl" / "utils" / "dataset" / "qwen35_preserve_thinking_template.py"
    spec = importlib.util.spec_from_file_location("qwen35_preserve_thinking_template", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_template_module = _load_template_module()
PRESERVE_THINKING_CHAT_TEMPLATE = _template_module.PRESERVE_THINKING_CHAT_TEMPLATE
preserve_thinking_template_sha256 = _template_module.preserve_thinking_template_sha256
write_preserve_thinking_template = _template_module.write_preserve_thinking_template


def patch_tokenizer_dir(model_dir: Path, *, backup: bool) -> dict:
    tokenizer_config = model_dir / "tokenizer_config.json"
    if not tokenizer_config.exists():
        raise FileNotFoundError(f"Missing tokenizer_config.json under {model_dir}")

    payload = json.loads(tokenizer_config.read_text(encoding="utf-8-sig"))
    old_template = payload.get("chat_template")
    old_sha = None
    if isinstance(old_template, str):
        old_sha = __import__("hashlib").sha256(old_template.encode("utf-8")).hexdigest()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if backup:
        backup_path = tokenizer_config.with_suffix(f".json.bak-{timestamp}")
        shutil.copy2(tokenizer_config, backup_path)
    else:
        backup_path = None

    payload["chat_template"] = PRESERVE_THINKING_CHAT_TEMPLATE
    payload["tau3_preserve_thinking_template"] = {
        "enabled": True,
        "template_sha256": preserve_thinking_template_sha256(),
        "patched_at_utc": timestamp,
        "old_template_sha256": old_sha,
    }
    tokenizer_config.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    jinja_path = write_preserve_thinking_template(model_dir / "chat_template.jinja")
    return {
        "model_dir": str(model_dir),
        "tokenizer_config": str(tokenizer_config),
        "chat_template_jinja": str(jinja_path),
        "backup": str(backup_path) if backup_path else None,
        "template_sha256": preserve_thinking_template_sha256(),
        "old_template_sha256": old_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="HF checkpoint/tokenizer directory to patch")
    parser.add_argument("--no-backup", action="store_true", help="Do not save tokenizer_config backup")
    args = parser.parse_args()

    result = patch_tokenizer_dir(Path(args.model_dir).expanduser().resolve(), backup=not args.no_backup)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
