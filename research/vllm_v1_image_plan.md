# Tau3 latest-VERL vLLM V1 Image Plan

Date: 2026-05-06

## Goal

Build one reproducible container image for Tau3 latest-VERL runs on vLLM 0.20.x with `VLLM_USE_V1=1`, then push it to ECR so future P5 work does not depend on hand-built conda environments.

## Image Strategy

- Base image: `vllm/vllm-openai:v0.20.0-cu129`.
- Reason: the official vLLM image pins the hard part of the stack, namely vLLM, PyTorch, CUDA runtime, and compiled extensions.
- Layer on top:
  - latest-VERL repo code with `pip install -e . --no-deps`
  - VERL/Tau3 Python dependencies
  - `tau2-bench` at commit `220b47844fb74d4351037e81055cf1e2948e4734`
- Keep checkpoints, datasets, outputs, and W&B state outside the image and mount them from the P5 host.

## Files Added

- `.dockerignore`
- `docker/Dockerfile.tau3.vllm20.v1`
- `scripts/p5_build_push_ecr_vllm_v1.sh`
- `scripts/p5_run_image_smoke_vllm_v1.sh`

## P5 Build And Push

```bash
cd ~/verl_tau3_sdpo

AWS_REGION=us-east-1 \
ECR_REPOSITORY=tau3-verl-vllm20-v1 \
IMAGE_TAG=20260506-d880bc31 \
bash scripts/p5_build_push_ecr_vllm_v1.sh
```

The script prints the final `image_uri`. Use that exact URI for the smoke script.

## P5 Smoke

```bash
cd ~/verl_tau3_sdpo

export IMAGE_URI=<printed ECR image uri>
export MODEL_PATH=~/verl_tau3_sdpo/checkpoints/SDPO/tau3_verl_sft/TAU3-VERL-SFT-FULL-Qwen-Qwen3.5-4B-qwen35_4b_vlm_full_traj_sft_real_9k/global_step_800/huggingface
export TASK_PATH=datasets/tau3_live_airline_canonical_json

SMOKE_MODE=image_preflight bash scripts/p5_run_image_smoke_vllm_v1.sh
SMOKE_MODE=preflight bash scripts/p5_run_image_smoke_vllm_v1.sh
SMOKE_MODE=grpo bash scripts/p5_run_image_smoke_vllm_v1.sh
```

Use `SMOKE_MODE=sdpo` only after GRPO smoke proves the engine path.

## Success Criteria

- Image builds and pushes to ECR.
- ECR push records both tag and digest; use the digest for rollback.
- `SMOKE_MODE=image_preflight` passes without mounting the host repo, proving the baked image contains the expected code/deps.
- Container preflight passes:
  - `VLLM_USE_V1=1`
  - vLLM version is at least `0.20.0`
  - `vllm._C` imports
  - `cuda_runtime.h` exists
  - Tau3 parser preserves XML numeric parameters as ints
- GRPO smoke starts vLLM engine and completes 2-3 training steps without:
  - `undefined symbol` ABI errors
  - `cuda_runtime.h` FlashInfer/GDN JIT errors
  - hybrid KV page-size errors
  - immediate Tau3 parser/tool-call regression

## Stop Conditions

- If `vllm._C` fails to import, stop. The base image or mounted Python path is not coherent.
- If `cuda_runtime.h` is missing, stop. The image is not carrying a full enough CUDA toolkit for FlashInfer/GDN JIT.
- If vLLM V1 still fails on hybrid KV page-size initialization, do not start overnight training. Try a tiny follow-up smoke with `VLLM_BLOCK_SIZE` and `VLLM_MAMBA_BLOCK_SIZE`, then report.
- If Bedrock credentials or model access fail, stop after preflight/engine proof; do not debug model-policy behavior.
