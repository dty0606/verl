# Tau3 latest-VERL vLLM V1 Image Plan

Date: 2026-05-06

## Goal

Build one reproducible container image for Tau3 latest-VERL runs on vLLM 0.20.x with `VLLM_USE_V1=1`, then push it to ECR so future P5 work does not depend on hand-built conda environments.

## Image Strategy

- Base image: `vllm/vllm-openai:v0.20.1-cu129`.
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

P5 instances do not need Git metadata for this flow. Use GitHub only between Codex/Kiro, then sync the repo snapshot to P5 through S3. Pass the Git commit as `SOURCE_REVISION` so the image tag and ECR metadata still record the exact source version.

Do not build this image on the corporate laptop for validation. The laptop can run light syntax/docs checks, but the Docker build and smoke must happen on P5 because the image needs to prove the P5 GPU/CUDA/vLLM runtime.

```bash
# Kiro/local side: after pulling the GitHub commit, publish the repo snapshot for P5.
cd ~/verl_tau3_sdpo
SOURCE_REVISION="$(git rev-parse --short HEAD)"

aws s3 sync ~/verl_tau3_sdpo/ s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/repo/ \
  --exclude ".git/*" \
  --exclude "datasets/*" \
  --exclude "checkpoints/*" \
  --exclude "outputs/*" \
  --exclude "output/*" \
  --exclude "wandb/*" \
  --exclude "logs/*" \
  --region us-west-2
```

```bash
# P5 side: pull the repo snapshot from S3, then build and push from P5.
mkdir -p ~/verl_tau3_sdpo
aws s3 sync s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/repo/ ~/verl_tau3_sdpo/ \
  --exclude ".git/*" \
  --exclude "datasets/*" \
  --exclude "checkpoints/*" \
  --exclude "outputs/*" \
  --exclude "output/*" \
  --exclude "wandb/*" \
  --exclude "logs/*" \
  --region us-west-2

cd ~/verl_tau3_sdpo

AWS_REGION=us-east-1 \
ECR_REPOSITORY=tau3-verl-vllm20-v1 \
SOURCE_REVISION=<github_commit_short_sha> \
IMAGE_TAG=20260506-<github_commit_short_sha> \
bash scripts/p5_build_push_ecr_vllm_v1.sh
```

The script does not require `git` on P5. It prints the final `image_uri` and digest. Use that exact URI for the smoke script.

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

## Conda Capacity Matrix For SM Code Editor

SageMaker Code Editor may be container-based and unable to run Docker. In that
case, use the isolated conda env path first, then run the same ordered vLLM V1
capacity ladder without Docker.

See `research/p5_vllm_v1_conda_runbook.md` for the complete no-Docker P5
sequence, including S3 snapshot sync, `tau2-bench`, preflight gates, capacity
smoke criteria, and frozen-env export.

```bash
cd ~/verl_tau3_sdpo

ENV_NAME=sdpo-vllm20-v1 \
VLLM_VERSION=0.20.1 \
bash scripts/p5_setup_vllm_v1_env.sh

conda activate sdpo-vllm20-v1
export VLLM_USE_V1=1
export TAU3_LIVE_USER_MODEL=us.anthropic.claude-sonnet-4-6
export TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION=0
export MODEL_PATH=~/verl_tau3_sdpo/checkpoints/SDPO/tau3_verl_sft/TAU3-VERL-SFT-FULL-Qwen-Qwen3.5-4B-qwen35_4b_vlm_full_traj_sft_real_9k/global_step_800/huggingface
export TASK_PATH=datasets/tau3_live_airline_canonical_json

bash scripts/p5_run_vllm_v1_capacity_matrix.sh
```

The current Dockerfile is pinned to `vllm/vllm-openai:v0.20.1-cu129`, the
latest vLLM 0.20.x tag available during the 2026-05-06 QC pass. If Qwen3.5
hybrid KV support regresses on `0.20.1`, fall back explicitly to the previous
known tag rather than relying on `latest`:

```bash
BASE_IMAGE=vllm/vllm-openai:v0.20.0-cu129 \
ECR_REPOSITORY=tau3-verl-vllm-v1 \
bash scripts/p5_build_push_ecr_vllm_v1.sh
```

Do not rely on `latest` for the paper artifact unless the resulting image
digest is recorded and promoted to a pinned ECR tag.

## vLLM V1 Capacity Ladder

Run the capacity matrix in this order and stop escalating once a profile is both
stable and fast enough for overnight experiments:

1. `auto_no_prefix_24k_48k`: compatibility proof. This keeps prefix caching
   disabled and uses normal KV cache.
2. `auto_prefix_24k_48k`: speed proof for shared Tau3 prefixes.
3. `fp8_no_prefix_24k_48k`: capacity proof for FP8 KV cache without prefix
   caching confounds.
4. `fp8_prefix_24k_48k`: combined speed/capacity proof at the recommended
   near-term training budget.
5. `fp8_prefix_32k_64k`: stretch proof for doubling from the old 16K/32K
   budget.

All launcher paths now default `TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION=0`, which
turns off duplicated full-transcript observations while preserving historical
actor `<think>` messages in the actual chat history.

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
- Capacity matrix identifies the largest stable profile among:
  - BF16/auto KV cache at 24K response / 48K model length
  - prefix caching at 24K response / 48K model length
  - FP8 KV cache at 24K response / 48K model length
  - FP8 KV cache + prefix caching at 32K response / 64K model length

## vLLM V1 Hybrid-KV Smoke Matrix

The launcher now emits explicit `--no-enable-prefix-caching` when `VLLM_ENABLE_PREFIX_CACHING=false`, and the image smoke script forwards the optional vLLM fallback knobs into the container. Smoke GRPO first; run SDPO only after the vLLM engine starts cleanly.

1. Baseline:
   ```bash
   TAU3_VLLM_PROFILE=qwen35_v1 \
   SMOKE_MODE=grpo \
   bash scripts/p5_run_image_smoke_vllm_v1.sh
   ```

2. If the hybrid KV page-size error persists:
   ```bash
   VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER=true \
   SMOKE_MODE=grpo \
   bash scripts/p5_run_image_smoke_vllm_v1.sh
   ```

3. If needed, try aligned blocks with prefix caching enabled:
   ```bash
   VLLM_ENABLE_PREFIX_CACHING=true \
   VLLM_BLOCK_SIZE=16 \
   VLLM_MAMBA_BLOCK_SIZE=16 \
   VLLM_MAMBA_CACHE_MODE=align \
   SMOKE_MODE=grpo \
   bash scripts/p5_run_image_smoke_vllm_v1.sh
   ```

4. Last alignment fallback:
   ```bash
   VLLM_ENABLE_PREFIX_CACHING=true \
   VLLM_BLOCK_SIZE=8 \
   VLLM_MAMBA_BLOCK_SIZE=8 \
   VLLM_MAMBA_CACHE_MODE=align \
   SMOKE_MODE=grpo \
   bash scripts/p5_run_image_smoke_vllm_v1.sh
   ```

Use `VLLM_KV_CACHE_MEMORY_BYTES` only when logs indicate cache-capacity/profiling issues, not as the first response to page-size unification errors.

## Stop Conditions

- If `vllm._C` fails to import, stop. The base image or mounted Python path is not coherent.
- If `cuda_runtime.h` is missing, stop. The image is not carrying a full enough CUDA toolkit for FlashInfer/GDN JIT.
- If vLLM V1 still fails on hybrid KV page-size initialization, do not start overnight training. Try a tiny follow-up smoke with `VLLM_BLOCK_SIZE` and `VLLM_MAMBA_BLOCK_SIZE`, then report.
- If Bedrock credentials or model access fail, stop after preflight/engine proof; do not debug model-policy behavior.
