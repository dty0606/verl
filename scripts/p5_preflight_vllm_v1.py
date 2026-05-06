#!/usr/bin/env python3
"""Preflight checks for Tau3 latest-VERL runs on vLLM V1.

This script is intentionally conservative. It catches the failure modes we hit
while moving Qwen3.5 Tau3 runs to newer vLLM stacks: torch/vLLM ABI mismatch,
missing CUDA headers for FlashInfer/GDN JIT, accidentally running vLLM V0, and
the Tau3 numeric-argument parser regression.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version


def _print(key: str, value: Any) -> None:
    print(f"{key}={value}")


def _fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def _warn(message: str) -> None:
    print(f"WARN: {message}")


def _dist_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "<not-installed>"


def _parse_version(raw: str) -> Version | None:
    clean = raw.split("+", 1)[0]
    try:
        return Version(clean)
    except InvalidVersion:
        return None


def check_env(args: argparse.Namespace) -> None:
    _print("PYTHON", sys.version.replace("\n", " "))
    _print("EXECUTABLE", sys.executable)
    _print("VLLM_USE_V1", os.environ.get("VLLM_USE_V1", "<unset>"))
    _print("CUDA_HOME", os.environ.get("CUDA_HOME", "<unset>"))
    _print("LD_LIBRARY_PATH_SET", bool(os.environ.get("LD_LIBRARY_PATH")))

    if args.require_v1_env and os.environ.get("VLLM_USE_V1") != "1":
        _fail("VLLM_USE_V1 must be set to 1 for the vLLM V1 migration smoke.")


def check_cuda_header(args: argparse.Namespace) -> None:
    candidates: list[Path] = []
    if os.environ.get("CUDA_HOME"):
        candidates.append(Path(os.environ["CUDA_HOME"]) / "include" / "cuda_runtime.h")
    candidates.append(Path("/usr/local/cuda/include/cuda_runtime.h"))
    candidates.append(Path("/usr/local/cuda-12/include/cuda_runtime.h"))

    found = next((path for path in candidates if path.exists()), None)
    _print("CUDA_RUNTIME_HEADER", found if found else "<missing>")
    if found is None and not args.allow_missing_cuda_header:
        _fail(
            "cuda_runtime.h is missing. FlashInfer/GDN JIT can fail later with "
            "'fatal error: cuda_runtime.h: No such file or directory'."
        )


def check_runtime_versions(args: argparse.Namespace) -> None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runs on P5
        _fail(f"cannot import torch: {exc!r}")

    try:
        import vllm
    except Exception as exc:  # pragma: no cover - runs on P5
        _fail(
            "cannot import vllm. If this mentions an undefined symbol in "
            "vllm._C, the torch/vLLM wheel ABI is mismatched. "
            f"Original error: {exc!r}"
        )

    _print("TORCH_VERSION", torch.__version__)
    _print("TORCH_CUDA", getattr(torch.version, "cuda", None))
    _print("VLLM_VERSION", vllm.__version__)
    _print("FLASHINFER_VERSION", _dist_version("flashinfer-python"))
    _print("TRANSFORMERS_VERSION", _dist_version("transformers"))
    _print("TOKENIZERS_VERSION", _dist_version("tokenizers"))

    vllm_version = _parse_version(vllm.__version__)
    min_vllm = _parse_version(args.min_vllm)
    if vllm_version is None:
        _warn(f"Could not parse vLLM version {vllm.__version__!r}")
    elif min_vllm is not None and vllm_version < min_vllm and not args.allow_older_vllm:
        _fail(f"vLLM {vllm.__version__} is older than required {args.min_vllm}.")

    if args.require_vllm_extension:
        try:
            importlib.import_module("vllm._C")
        except Exception as exc:  # pragma: no cover - runs on P5
            _fail(
                "vllm._C failed to import. This is usually a torch/vLLM ABI "
                f"mismatch. Original error: {exc!r}"
            )
        _print("VLLM_C_EXTENSION", "PASS")

    if torch.cuda.is_available():
        _print("CUDA_DEVICE_0", torch.cuda.get_device_name(0))
        _print("CUDA_DEVICE_COUNT", torch.cuda.device_count())
    else:
        _warn("torch.cuda.is_available() is false; this preflight is expected to run on a GPU P5 node.")


def check_tau3_parser() -> None:
    try:
        from verl.utils.tau3_action_parser import parse_model_output_to_tau_action
    except Exception as exc:
        _fail(f"cannot import Tau3 parser from local verl package: {exc!r}")

    raw = (
        "<tool_call><function=update_reservation_baggages>"
        "<parameter=nonfree_baggages>0</parameter>"
        "<parameter=total_baggages>3</parameter>"
        "</function></tool_call>"
    )
    parsed = parse_model_output_to_tau_action(raw)
    args = json.loads(parsed.action_for_env)["arguments"]
    types = {key: type(value).__name__ for key, value in args.items()}
    _print("TAU3_PARSER_TYPES", types)
    if types.get("nonfree_baggages") != "int" or types.get("total_baggages") != "int":
        _fail("Tau3 parser is not preserving numeric XML parameters as ints.")
    _print("TAU3_PARSER", "PASS")


def check_paths(args: argparse.Namespace) -> None:
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).expanduser()
        _print("DATASET_DIR", dataset_dir)
        for name in ("train.parquet", "test.parquet"):
            path = dataset_dir / name
            _print(f"DATASET_{name.upper()}", path if path.exists() else "<missing>")
            if not path.exists():
                _fail(f"missing dataset file: {path}")

    if args.model_path:
        model_path = Path(args.model_path).expanduser()
        _print("MODEL_PATH", model_path)
        if not model_path.exists() and "/" not in args.model_path:
            _fail(f"model path does not exist: {model_path}")
        if not args.skip_model_config:
            check_model_config(args.model_path)


def _interesting_config_fields(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    needles = (
        "architectures",
        "model_type",
        "attention",
        "linear",
        "mamba",
        "gdn",
        "delta",
        "sliding",
        "hybrid",
    )
    found: dict[str, Any] = {}
    stack: list[tuple[str, Any]] = [("", cfg_dict)]
    while stack:
        prefix, value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if any(needle in key.lower() for needle in needles):
                    found[path] = child
                if isinstance(child, dict):
                    stack.append((path, child))
        elif isinstance(value, list) and any(needle in prefix.lower() for needle in needles):
            found[prefix] = value
    return found


def check_model_config(model_path: str) -> None:
    try:
        from transformers import AutoConfig
    except Exception as exc:
        _warn(f"cannot import transformers AutoConfig: {exc!r}")
        return

    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        _warn(f"could not load model config for {model_path}: {exc!r}")
        return

    cfg_dict = cfg.to_dict()
    _print("MODEL_TYPE", cfg_dict.get("model_type"))
    _print("MODEL_ARCHITECTURES", cfg_dict.get("architectures"))
    interesting = _interesting_config_fields(cfg_dict)
    for key in sorted(interesting)[:30]:
        value = interesting[key]
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, sort_keys=True)[:500]
        else:
            rendered = value
        _print(f"MODEL_CONFIG_{key}", rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--min-vllm", default="0.20.0")
    parser.add_argument("--allow-older-vllm", action="store_true")
    parser.add_argument("--allow-missing-cuda-header", action="store_true")
    parser.add_argument("--skip-model-config", action="store_true")
    parser.add_argument("--require-v1-env", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-vllm-extension", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    check_env(args)
    check_cuda_header(args)
    check_runtime_versions(args)
    check_tau3_parser()
    check_paths(args)
    _print("PREFLIGHT", "PASS")


if __name__ == "__main__":
    main()
