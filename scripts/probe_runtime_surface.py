#!/usr/bin/env python3
"""Emit a read-only JSON probe of the local or P5 runtime surface."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENV_KEYS = (
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "VLLM_USE_V1",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_LOGGING_LEVEL",
    "VLLM_LANGUAGE_MODEL_ONLY",
    "TAU3_VLLM_PROFILE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "RAY_ADDRESS",
    "WANDB_MODE",
    "WANDB_PROJECT",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "WSL_DISTRO_NAME",
)

IMPORT_TARGETS = (
    ("torch", ("torch",)),
    ("vllm", ("vllm",)),
    ("transformers", ("transformers",)),
    ("tokenizers", ("tokenizers",)),
    ("flashinfer", ("flashinfer-python", "flashinfer")),
    ("ray", ("ray",)),
)

IMPORT_PROBE_CODE = r"""
import importlib
import importlib.metadata as metadata
import json
import sys
import traceback

module_name = sys.argv[1]
dist_names = [name for name in sys.argv[2].split(",") if name]
out = {
    "module": module_name,
    "ok": False,
    "version": None,
    "dist_versions": {},
    "error": None,
    "traceback_tail": None,
    "torch_cuda": None,
}

try:
    module = importlib.import_module(module_name)
    out["ok"] = True
    version = getattr(module, "__version__", None)
    if version is not None:
        out["version"] = str(version)

    for dist_name in dist_names:
        try:
            out["dist_versions"][dist_name] = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            out["dist_versions"][dist_name] = None
        except Exception as exc:
            out["dist_versions"][dist_name] = f"<error: {exc!r}>"

    if module_name == "torch":
        cuda = {
            "torch_cuda_version": str(getattr(module.version, "cuda", None)),
            "available": bool(module.cuda.is_available()),
            "device_count": None,
            "devices": [],
        }
        try:
            cuda["device_count"] = int(module.cuda.device_count())
        except Exception as exc:
            cuda["device_count_error"] = repr(exc)

        if cuda.get("device_count"):
            for index in range(cuda["device_count"]):
                device = {"index": index}
                try:
                    device["name"] = module.cuda.get_device_name(index)
                except Exception as exc:
                    device["name_error"] = repr(exc)
                try:
                    props = module.cuda.get_device_properties(index)
                    device["total_memory_bytes"] = int(getattr(props, "total_memory", 0))
                    device["major"] = int(getattr(props, "major", 0))
                    device["minor"] = int(getattr(props, "minor", 0))
                    device["multi_processor_count"] = int(
                        getattr(props, "multi_processor_count", 0)
                    )
                except Exception as exc:
                    device["properties_error"] = repr(exc)
                cuda["devices"].append(device)
        out["torch_cuda"] = cuda
except BaseException as exc:
    out["error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "repr": repr(exc),
    }
    out["traceback_tail"] = traceback.format_exc().splitlines()[-8:]

print(json.dumps(out, sort_keys=True))
"""


def _truncate(text: str | None, limit: int = 12000) -> tuple[str, bool]:
    if text is None:
        return "", False
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...<truncated>...", True


def run_command(args: list[str], timeout: float) -> dict[str, Any]:
    start = time.monotonic()
    result: dict[str, Any] = {
        "command": args,
        "present": bool(shutil.which(args[0])),
        "timed_out": False,
    }
    if not result["present"]:
        result["status"] = "missing"
        return result

    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        stdout, stdout_truncated = _truncate(completed.stdout)
        stderr, stderr_truncated = _truncate(completed.stderr)
        result.update(
            {
                "status": "completed",
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
        )
    except subprocess.TimeoutExpired as exc:
        stdout, stdout_truncated = _truncate(
            exc.stdout.decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout
        )
        stderr, stderr_truncated = _truncate(
            exc.stderr.decode("utf-8", "replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr
        )
        result.update(
            {
                "status": "timeout",
                "timed_out": True,
                "returncode": None,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "returncode": None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
    finally:
        result["duration_ms"] = round((time.monotonic() - start) * 1000, 2)

    return result


def run_python_probe(module_name: str, dist_names: tuple[str, ...], timeout: float) -> dict[str, Any]:
    start = time.monotonic()
    args = [
        sys.executable,
        "-c",
        IMPORT_PROBE_CODE,
        module_name,
        ",".join(dist_names),
    ]
    raw = run_command(args, timeout=timeout)
    probe: dict[str, Any] = {
        "module": module_name,
        "ok": False,
        "status": raw.get("status"),
        "timed_out": raw.get("timed_out", False),
        "duration_ms": raw.get("duration_ms"),
    }

    if raw.get("status") != "completed":
        probe["command_result"] = raw
        return probe

    stdout = raw.get("stdout", "")
    json_line = next((line for line in reversed(stdout.splitlines()) if line.strip()), "")
    try:
        parsed = json.loads(json_line)
        parsed["duration_ms"] = round((time.monotonic() - start) * 1000, 2)
        if raw.get("stderr"):
            stderr, truncated = _truncate(raw["stderr"], limit=4000)
            parsed["stderr"] = stderr
            parsed["stderr_truncated"] = truncated
        return parsed
    except json.JSONDecodeError as exc:
        probe.update(
            {
                "status": "json_parse_error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "command_result": raw,
            }
        )
        return probe


def parse_nvidia_smi_csv(stdout: str) -> list[dict[str, str]]:
    keys = (
        "index",
        "name",
        "driver_version",
        "memory_total_mib",
        "memory_free_mib",
        "utilization_gpu_percent",
        "temperature_gpu_c",
    )
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        values = [part.strip() for part in line.split(",")]
        rows.append({key: values[index] if index < len(values) else "" for index, key in enumerate(keys)})
    return rows


def disk_usage(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    usage = shutil.disk_usage(resolved)
    return {
        "path": str(resolved),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "total_gib": round(usage.total / (1024**3), 2),
        "used_gib": round(usage.used / (1024**3), 2),
        "free_gib": round(usage.free / (1024**3), 2),
    }


def collect_env() -> dict[str, Any]:
    env: dict[str, Any] = {}
    path_like = {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH"}
    for key in ENV_KEYS:
        value = os.environ.get(key)
        if value is None:
            env[key] = None
        elif key in path_like:
            env[key] = [part for part in value.split(os.pathsep) if part]
        else:
            env[key] = value
    return env


def is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "microsoft" in version.lower() or "wsl" in version.lower()


def cuda_runtime_candidates() -> list[dict[str, Any]]:
    candidates: list[Path] = []

    for env_key in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX"):
        value = os.environ.get(env_key)
        if value:
            candidates.append(Path(value) / "include" / "cuda_runtime.h")

    candidates.extend(
        [
            Path("/usr/local/cuda/include/cuda_runtime.h"),
            Path("/usr/local/cuda-12/include/cuda_runtime.h"),
            Path("/usr/local/cuda-12.9/include/cuda_runtime.h"),
            Path("/usr/local/cuda/targets/x86_64-linux/include/cuda_runtime.h"),
            Path("/usr/local/cuda/targets/sbsa-linux/include/cuda_runtime.h"),
            Path(sys.prefix) / "include" / "cuda_runtime.h",
        ]
    )

    if platform.system() == "Windows":
        cuda_root = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"))
        if cuda_root.name.upper().startswith("V"):
            candidates.append(cuda_root / "include" / "cuda_runtime.h")
        else:
            candidates.extend(cuda_root.glob(r"v*\include\cuda_runtime.h"))

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        text = str(candidate)
        if text in seen:
            continue
        seen.add(text)
        out.append({"path": text, "exists": candidate.exists()})
    return out


def collect_commands(command_timeout: float) -> dict[str, Any]:
    commands: dict[str, Any] = {}

    nvidia_query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=command_timeout,
    )
    if nvidia_query.get("status") == "completed" and nvidia_query.get("returncode") == 0:
        nvidia_query["parsed_gpus"] = parse_nvidia_smi_csv(nvidia_query.get("stdout", ""))
    commands["nvidia_smi"] = {
        "path": shutil.which("nvidia-smi"),
        "query": nvidia_query,
        "list": run_command(["nvidia-smi", "-L"], timeout=command_timeout),
    }

    commands["nvcc"] = {
        "path": shutil.which("nvcc"),
        "version": run_command(["nvcc", "--version"], timeout=command_timeout),
    }

    commands["docker"] = {
        "path": shutil.which("docker"),
        "version": run_command(["docker", "version", "--format", "{{json .}}"], timeout=command_timeout),
        "info": run_command(["docker", "info", "--format", "{{json .}}"], timeout=command_timeout),
    }

    if platform.system() == "Windows":
        commands["wsl"] = {
            "applicable": True,
            "path": shutil.which("wsl"),
            "status": run_command(["wsl", "--status"], timeout=command_timeout),
            "list_verbose": run_command(["wsl", "-l", "-v"], timeout=command_timeout),
        }
    else:
        commands["wsl"] = {"applicable": False, "reason": "not_windows"}

    return commands


def collect_imports(import_timeout: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    imports: dict[str, Any] = {}
    torch_cuda: dict[str, Any] = {"available": None, "reason": "torch_not_imported"}

    for module_name, dist_names in IMPORT_TARGETS:
        result = run_python_probe(module_name, dist_names, timeout=import_timeout)
        imports[module_name] = result
        if module_name == "torch":
            torch_cuda = result.get("torch_cuda") or {
                "available": None,
                "reason": "torch_import_failed_or_timed_out",
            }

    vllm_c = run_python_probe("vllm._C", ("vllm",), timeout=import_timeout)
    return imports, vllm_c, torch_cuda


def collect_probe(command_timeout: float, import_timeout: float) -> dict[str, Any]:
    cwd = Path.cwd()
    home = Path.home()
    tmp = Path(tempfile.gettempdir())
    uname = platform.uname()
    imports, vllm_c, torch_cuda = collect_imports(import_timeout=import_timeout)

    return {
        "schema": "runtime_surface_probe.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "uname": uname._asdict(),
            "is_windows": platform.system() == "Windows",
            "is_linux": platform.system() == "Linux",
            "is_wsl": is_wsl(),
        },
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
        },
        "paths": {
            "cwd": str(cwd),
            "home": str(home),
            "tmp": str(tmp),
        },
        "env": collect_env(),
        "disk_space": {
            "cwd": disk_usage(cwd),
            "home": disk_usage(home),
            "tmp": disk_usage(tmp),
        },
        "commands": collect_commands(command_timeout=command_timeout),
        "imports": imports,
        "vllm_c_import": vllm_c,
        "torch_cuda": torch_cuda,
        "cuda_runtime_h_candidates": cuda_runtime_candidates(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only local/P5 runtime probe. Outputs JSON to stdout and does "
            "not write files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/probe_runtime_surface.py
              python scripts/probe_runtime_surface.py --indent 2 --command-timeout 5 --import-timeout 30
            """
        ),
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation; use 0 for compact JSON.")
    parser.add_argument("--command-timeout", type=float, default=8.0, help="Timeout in seconds for command probes.")
    parser.add_argument("--import-timeout", type=float, default=30.0, help="Timeout in seconds for import probes.")
    args = parser.parse_args(argv)

    indent = None if args.indent == 0 else args.indent
    probe = collect_probe(command_timeout=args.command_timeout, import_timeout=args.import_timeout)
    print(json.dumps(probe, indent=indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
