# P5 Environment Issues: Lessons Learned

Date: 2026-05-06

## Summary

Creating new conda environments on SageMaker Code Editor P5 instances for Tau3 SDPO/GRPO training has been repeatedly blocked by dependency conflicts, ABI mismatches, and platform limitations. This document catalogs the issues encountered and the workarounds used.

## Platform Constraints

- **SageMaker Code Editor** runs inside a container, not a full VM. No `systemctl`, no `yum`/`apt`, and no Docker daemon are available in the current SM CE space.
- **No Docker-in-Docker**: The current SM CE space cannot build or run Docker images. Build Docker images on a separate EC2 instance, CodeBuild/GitHub Actions runner with Docker, or a Docker-enabled SageMaker domain.
- **99 GB EBS root volume**: The domain root volume is small for training artifacts. It filled during SDPO checkpoint save and caused `basic_ios::clear: iostream error`.
- **NVMe is ephemeral**: `/mnt/sagemaker-nvme/` has much more space but is lost on instance stop. Sync checkpoints and manifests to S3.
- **No `nvcc` on us-east-1 P5 SM CE**: FlashInfer GDN kernel JIT can fail with `cuda_runtime.h: No such file or directory`. Use a proven stack/cache or a container/base image with CUDA headers.

## Dependency Conflicts

### tokenizers vs transformers vs huggingface_hub

- `transformers==5.6.2` requires `huggingface-hub>=1.5.0`.
- `tokenizers==0.22.0` requires `huggingface-hub<1.0`.
- **Resolution**: Install with `--no-deps`, then manually install the runtime hub version. This violates tokenizers' resolver constraint but has worked at runtime.

### vLLM ABI mismatch with torch

- `vllm==0.19.1` must be matched with the Torch wheel it was compiled against.
- Loading `vllm==0.19.1` with `torch==2.11.0` caused:
  ```
  ImportError: vllm/_C.abi3.so: undefined symbol: _ZN3c1013MessageLoggerC1EPKciib
  ```
- **Resolution**: Treat Torch and vLLM as one ABI pair. Verify with `python -c "import vllm, importlib; importlib.import_module('vllm._C')"`.

### vLLM 0.20.x hybrid KV cache page-size error

- Qwen3.5 has mixed attention/linear-attention layers with different KV cache page sizes.
- vLLM 0.20.x V1 engine failed with:
  ```
  NotImplementedError: The page size of the layer is not divisible by the maximum page size. Cannot unify by adjusting block_size.
  ```
- **Resolution**: Either use the proven vLLM 0.19.1/V0 stack, or smoke-test vLLM 0.20.x/V1 with explicit `--no-enable-prefix-caching` and the hybrid-KV fallback knobs exposed by the Tau3 launchers. The vLLM 0.20.x path is not proven until P5 engine initialization passes.

### FlashInfer version mismatch

- `flashinfer-python==0.6.8.post1` from the vLLM 0.20.x attempt tried to JIT-compile GDN kernels and failed without CUDA headers.
- `flashinfer-python==0.6.6` worked in the known-good vLLM 0.19.1 environment with prebuilt kernel cache.
- **Resolution**: Pin FlashInfer with the vLLM/Torch stack and record whether CUDA headers and kernel caches are present.

### conda `set -u` / unbound variable

- Conda deactivation hooks can reference unset backup variables such as `CONDA_BACKUP_CXX`.
- **Resolution**: Use `set -eo pipefail` in conda-aware setup/export scripts, or wrap conda activation/deactivation with `set +u`.

## Missing Packages

Fresh environments repeatedly discovered missing transitive packages:

| Package | Why needed | Install method |
|---------|------------|----------------|
| `tensordict` | `verl.protocol` imports it | `pip install tensordict==0.10.0` |
| `multiprocess` | `datasets.arrow_dataset` imports it | `pip install multiprocess` |
| `xxhash` | `datasets.fingerprint` imports it | `pip install xxhash` |
| `fastuuid` | `litellm._uuid` imports it | `pip install fastuuid` |
| `latex2sympy2_extended` | `math_verify` imports it | `pip install latex2sympy2_extended` |
| `codetiming` | VERL runtime dependency | `pip install codetiming` |
| `torchdata` | VERL runtime dependency | `pip install torchdata` |
| `qwen-vl-utils` | Qwen3.5 processor utility | `pip install qwen-vl-utils` |
| `toml`, `addict`, `deepdiff`, `tenacity` | VERL/Tau3 runtime deps | individual pip installs |

## Known-Good Stack

Proven on west P5 for GRPO and vanilla-peer SDPO:

```text
torch==2.10.0+cu128
transformers==5.6.2
vllm==0.19.1
flash-attn==2.8.3
flashinfer-python==0.6.6
ray==2.53.0
tokenizers==0.22.0
huggingface-hub>=1.5.0
datasets==4.4.2
tensordict==0.10.0
```

This stack ran on 8xH100 with the Qwen3.5-4B VLM checkpoint.

## Workarounds Used

1. **Reuse existing env**: Instead of creating fresh envs, use `pip install -e <repo> --no-deps` inside the proven `sdpo-qwen35` env.
2. **Copy FlashInfer cache**: Sync prebuilt kernel cache between P5 instances via S3 when needed.
3. **Symlink libcudart**: Use `ln -sf /usr/local/cuda/lib64/libcudart.so /opt/conda/lib64/` only when a linker path issue is confirmed.
4. **Pin tokenizers separately**: Install `tokenizers==0.22.0 --no-deps` after the rest of the environment.
5. **NVMe workspace**: Symlink `~/verl_tau3_sdpo -> /mnt/sagemaker-nvme/verl_workspace/verl_tau3_sdpo` for disk-heavy work.
6. **S3 checkpoint sync**: Sync checkpoints/manifests to S3 frequently enough to survive instance stops.

## Local Corporate Laptop Probe

The Windows laptop with RTX 3080 is useful for lightweight validation only:

- `nvidia-smi` sees the RTX 3080 and driver CUDA capability.
- No Docker, no WSL, no conda, and no `nvcc` are currently available.
- Native Windows is not a faithful vLLM/PyTorch/FlashInfer target; official vLLM GPU support is Linux-first, with Windows users expected to use WSL for real GPU execution.
- A native Windows dry run for `vllm==0.20.0` found no matching binary wheel, so local vLLM20 validation requires WSL/Docker/Linux rather than this bare Windows Python.

Use the laptop for:

- Python syntax and JSON probe scripts.
- Git/S3 handoff sanity checks.
- Static launcher/config review.
- Read-only runtime probes via `scripts/probe_runtime_surface.py`.

Do not treat the laptop as proof for:

- H100/Hopper kernels.
- NCCL/Ray tensor-parallel behavior.
- FlashInfer GDN JIT/cache behavior.
- 8-GPU memory and 32K-context Tau3 training.

## New Reproducibility Helpers

- `scripts/probe_runtime_surface.py`: read-only JSON probe for local/P5 runtime surface, including platform, Python, disk, `nvidia-smi`, `nvcc`, Docker, WSL, package imports, `vllm._C`, Torch CUDA, and `cuda_runtime.h` candidates.
- `scripts/p5_export_frozen_env.sh`: no-Docker/no-Git frozen environment exporter for the active P5 conda env. It records `pip freeze`, `conda list` when available, runtime import versions, `vllm._C`, CUDA headers, `nvidia-smi`, and `SOURCE_REVISION`/Git metadata when available.

Recommended P5 capture after a successful smoke:

```bash
cd ~/verl_tau3_sdpo
SOURCE_REVISION=<github_commit_short_sha> \
  bash scripts/p5_export_frozen_env.sh outputs/frozen_env/<run_name>

python scripts/probe_runtime_surface.py \
  --indent 2 \
  > outputs/frozen_env/<run_name>/runtime_surface.json
```

## Recommendation for Future Runs

1. **Do not create new conda envs on SM CE** unless absolutely necessary. Reuse the proven `sdpo-qwen35` env.
2. **Build Docker images outside SM CE** for reproducibility. The current SM CE space cannot run Docker.
3. **For paper publication**: Provide `docker/Dockerfile.tau3.vllm20.v1` plus frozen environment manifests from a successful run. Do not rely on reviewers reproducing the ad hoc conda setup.
4. **If a new env is unavoidable**: Install Torch/vLLM as an ABI-matched pair first, then install local VERL/Tau3 with `--no-deps`. Never let pip freely resolve the full dependency tree.
5. **Always verify**: `python -c "import vllm, importlib; importlib.import_module('vllm._C')"` before launching training.
6. **For paper-style SDPO today**: prefer latest-VERL `SDPO_ARM=original` over old-repo `SDPO-paper-original` unless the goal is specifically a historical-code ablation. The latest-VERL path has parser fixes, pass^k validation aliases, and current Tau3 runtime guards.
