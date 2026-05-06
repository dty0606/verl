# P5 Environment Issues: Lessons Learned

Date: 2026-05-06

## Summary

Creating new conda environments on SageMaker Code Editor P5 instances for Tau3 SDPO/GRPO training has been repeatedly blocked by dependency conflicts, ABI mismatches, and platform limitations. This document catalogs all issues encountered and the workarounds used.

## Platform Constraints

- **SageMaker Code Editor** runs inside a container (not a full VM). No `systemctl`, no `yum`/`apt`, no Docker daemon.
- **No Docker-in-Docker**: Cannot build or run Docker images on SM CE. Must use a separate EC2 instance or GitHub Actions for image builds.
- **99 GB EBS root volume** (domain max at creation time). Cannot resize after creation. Filled 100% during SDPO checkpoint save, causing `basic_ios::clear: iostream error`.
- **NVMe is ephemeral** (28 TB `/mnt/sagemaker-nvme/`). Lost on instance stop. Must sync checkpoints to S3 periodically.
- **No `nvcc` on us-east-1 P5**: FlashInfer GDN kernel JIT compilation fails with `cuda_runtime.h: No such file or directory`. Must copy prebuilt kernel cache from a working machine.

## Dependency Conflicts

### tokenizers vs transformers vs huggingface_hub

- `transformers==5.6.2` requires `huggingface-hub>=1.5.0`
- `tokenizers==0.22.0` requires `huggingface-hub<1.0`
- **Resolution**: Install both with `--no-deps`, then manually install `huggingface_hub>=1.5.0 --no-deps`. Accept that tokenizers' hub constraint is violated (works at runtime).

### vLLM ABI mismatch with torch

- `vllm==0.19.1` compiled against torch 2.10.0. Loading with torch 2.11.0 causes:
  ```
  ImportError: vllm/_C.abi3.so: undefined symbol: _ZN3c1013MessageLoggerC1EPKciib
  ```
- **Resolution**: Must use torch 2.10.0 with vllm 0.19.1. Cannot mix versions.

### vLLM 0.20.x hybrid KV cache page-size error

- Qwen3.5 has mixed attention/linear-attention layers with different KV cache page sizes.
- vLLM 0.20.x V1 engine fails with:
  ```
  NotImplementedError: The page size of the layer is not divisible by the maximum page size. Cannot unify by adjusting block_size.
  ```
- **Resolution**: Either use vLLM 0.19.1 (V0 engine), or disable prefix caching (`VLLM_ENABLE_PREFIX_CACHING=false`) in 0.20.x. The latter is unverified — may still fail.

### FlashInfer version mismatch

- `flashinfer-python==0.6.8.post1` (installed by vLLM 0.20.1) tries to JIT-compile GDN kernels but fails because `cuda_runtime.h` is missing.
- `flashinfer-python==0.6.6` (used by vLLM 0.19.1) works with prebuilt kernel cache copied from west P5.
- **Resolution**: Pin flashinfer to match vLLM version. Copy kernel cache between machines via S3.

### conda `set -u` / unbound variable

- conda deactivation hooks reference `CONDA_BACKUP_CXX` which is unset, breaking scripts with `set -euo pipefail`.
- **Resolution**: Use `set -eo pipefail` (drop `-u`) in setup scripts, or wrap conda commands in `set +u; ...; set -u`.

## Missing Packages (discovered iteratively)

When creating a fresh env, these packages were missing and had to be installed one-by-one:

| Package | Why needed | Install method |
|---------|-----------|---------------|
| tensordict | verl.protocol imports it | `pip install tensordict==0.10.0` |
| multiprocess | datasets.arrow_dataset imports it | `pip install multiprocess` |
| xxhash | datasets.fingerprint imports it | `pip install xxhash` |
| fastuuid | litellm._uuid imports it | `pip install fastuuid` |
| latex2sympy2_extended | math_verify imports it | `pip install latex2sympy2_extended` |
| codetiming | verl runtime dep | `pip install codetiming` |
| torchdata | verl runtime dep | `pip install torchdata` |
| qwen-vl-utils | Qwen3.5 processor | `pip install qwen-vl-utils` |
| toml, addict, deepdiff, tenacity | verl/tau3 runtime deps | individual pip installs |

## Known-Good Stack (proven on west P5 for GRPO)

```
torch==2.10.0+cu128
transformers==5.6.2
vllm==0.19.1
flash-attn==2.8.3 (prebuilt wheel for torch 2.10+cu128)
flashinfer-python==0.6.6 (with prebuilt kernel cache)
ray==2.53.0
tokenizers==0.22.0
huggingface-hub>=1.5.0
datasets==4.4.2
tensordict==0.10.0
```

This stack runs GRPO and vanilla-peer SDPO successfully on 8×H100 with Qwen3.5-4B VLM checkpoint.

## Workarounds Used

1. **Reuse existing env**: Instead of creating fresh envs, `pip install -e <repo> --no-deps` into the working `sdpo-qwen35` env.
2. **Copy FlashInfer cache**: `aws s3 cp` the prebuilt kernel cache between P5 instances.
3. **Symlink libcudart**: `ln -sf /usr/local/cuda/lib64/libcudart.so /opt/conda/lib64/` for FlashInfer linker.
4. **Pin tokenizers separately**: `pip install tokenizers==0.22.0 --no-deps` after everything else.
5. **NVMe workspace**: Symlink `~/verl_tau3_sdpo → /mnt/sagemaker-nvme/verl_workspace/verl_tau3_sdpo` for disk space.
6. **S3 checkpoint sync**: Hourly cron or manual sync to survive instance stops.

## Recommendation for Future Runs

1. **Do not create new conda envs on SM CE** unless absolutely necessary. Reuse the proven `sdpo-qwen35` env.
2. **Build Docker images on EC2** (not SM CE) for reproducibility. SM CE cannot run Docker.
3. **For paper publication**: Provide the Dockerfile (`docker/Dockerfile.tau3.vllm20.v1`) and a `requirements-frozen.txt` from the working env. Do not expect reviewers to reproduce the conda setup.
4. **If a new env is unavoidable**: Start from `conda create -n <name> python=3.12`, install torch first from the PyTorch index, then install everything else with `--no-deps` to avoid resolver conflicts. Never let pip resolve the full dependency tree.
5. **Always verify**: `python -c "import vllm; vllm._C"` before launching training. This catches ABI mismatches immediately.
