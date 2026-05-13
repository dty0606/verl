# Qwen3.5 VLM-Format SFT to GRPO Implementation Plan

Date: 2026-05-02

## Decision

Continue the Tau3 airline SFT -> GRPO -> SDPO path in latest VERL, but do not rely on Qwen3.5 text-only checkpoints for vLLM rollout.

The main path is:

1. Train full-parameter thinking SFT from the original `Qwen/Qwen3.5-4B` checkpoint.
2. Save/export the SFT checkpoint as HuggingFace `hf_model` while preserving the original VLM-style Qwen3.5 config.
3. Serve that exported checkpoint with vLLM using `--language-model-only`.
4. Use the same exported checkpoint as `MODEL_PATH` for the latest-VERL GRPO smoke.
5. Only after SFT export and GRPO smoke are clean, run larger SFT/GRPO jobs.

## Why

Public and local evidence are aligned:

- vLLM documents Qwen3.5 VLM checkpoints with `--language-model-only` for text-only serving.
- vLLM text-only `Qwen3_5ForCausalLM` support is still not a safe dependency; open issues/PRs show missing registry/config handling for text-only Qwen3.5 checkpoints.
- Latest VERL has merged Qwen3.5 FSDP/GRPO support, but our exact 4B Tau3 SFT -> vLLM rollout path is still unverified.
- The old VERL fork failed around Qwen3.5 config compatibility and vLLM rollout loading, so runtime proof on P5 is required before long jobs.

Relevant public references:

- `https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md`
- `https://github.com/vllm-project/vllm/issues/39231`
- `https://github.com/vllm-project/vllm/pull/40471`
- `https://github.com/verl-project/verl/pull/5682`

## Model-Path Contract

The first P5 task must answer these exactly.

Rollout generator:

- vLLM inside latest VERL rollout.
- It should load the exported SFT HuggingFace checkpoint, not the base model and not a text-only hacked checkpoint.
- vLLM should receive `language_model_only=true`.

Training model:

- latest VERL FSDP full-parameter actor initialized from `Qwen/Qwen3.5-4B` for SFT.
- For GRPO, actor initialized from the exported SFT checkpoint after the vLLM serve smoke passes.

Evaluation model:

- The same exported SFT checkpoint served by vLLM for smoke inference.
- For GRPO validation, the same checkpoint is used by actor/rollout/ref according to latest VERL config.

Weight movement:

- No LoRA adapter sync in the main path.
- SFT produces a full checkpoint with `hf_model`.
- GRPO starts from the exported HF checkpoint.
- During GRPO, latest VERL's hybrid engine/checkpoint engine updates rollout weights from the actor.

Proof artifacts:

- SFT checkpoint `huggingface/config.json` showing `model_type=qwen3_5` and `architectures` containing `Qwen3_5ForConditionalGeneration`.
- One standalone vLLM output from that exact checkpoint path.
- One latest-VERL GRPO rollout sample with decoded non-gibberish response and non-null `tau3_live_result`.
- One GRPO update step with nonzero actor loss/grad metrics.

## Stage 0: Environment Truth

Local Windows/Codex is not the source of truth for CUDA, vLLM, VERL package compatibility, or Qwen3.5 runtime behavior.

The P5/Kiro environment is the source of truth for:

- installed vLLM/Transformers/VERL versions,
- Qwen3.5 load behavior,
- vLLM `--language-model-only`,
- FSDP checkpoint export,
- live Tau3 official-gym rollout behavior.

Before running anything long, record:

```bash
python - <<'PY'
import importlib.metadata as md
for pkg in ["torch", "transformers", "vllm", "verl"]:
    try:
        print(pkg, md.version(pkg))
    except Exception as exc:
        print(pkg, "UNKNOWN", exc)
PY
echo "S3 snapshot working tree: $(pwd)"
find . -maxdepth 1 -type f | sort | head -20
```

## Stage 1: Tiny VLM-Format SFT Export Smoke

Goal: prove latest VERL can SFT and export a VLM-format Qwen3.5 HF checkpoint.

Use a tiny accepted thinking dataset first. Do not run the full 5K/20K SFT yet.

Suggested launch:

```bash
MODEL_PATH=Qwen/Qwen3.5-4B \
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' \
NUM_GPUS=8 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=1 \
TEST_FREQ=1 \
MAX_LENGTH=4096 \
MAX_TOKEN_LEN_PER_GPU=4096 \
TRAIN_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=1 \
bash scripts/tau3/run_tau3_verl_sft_full_thinking.sh \
  datasets/tau3_sft_thinking_train_only \
  qwen35_4b_vlm_export_smoke
```

Expected artifact:

```text
checkpoints/SDPO/tau3_verl_sft/<experiment>/global_step_*/huggingface
```

Required config check:

```bash
HF_CKPT=/path/to/global_step_*/huggingface
python - <<'PY'
import json, os
p = os.environ["HF_CKPT"] + "/config.json"
c = json.load(open(p))
print("model_type:", c.get("model_type"))
print("architectures:", c.get("architectures"))
assert c.get("model_type") == "qwen3_5"
assert "Qwen3_5ForConditionalGeneration" in (c.get("architectures") or [])
PY
```

If this assertion fails and the checkpoint is `qwen3_5_text` / `Qwen3_5ForCausalLM`, stop. Do not patch only `config.json` and proceed as if the checkpoint is validated.

## Stage 2: Standalone vLLM Serve Smoke

Goal: prove the exact SFT `huggingface` export is loadable by vLLM before involving GRPO.

Suggested serve:

```bash
export HF_CKPT=/path/to/global_step_*/huggingface
python -m vllm.entrypoints.openai.api_server \
  --model "$HF_CKPT" \
  --served-model-name tau3-sft-smoke \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --language-model-only \
  --trust-remote-code \
  --port 9100
```

Suggested request:

```bash
curl http://127.0.0.1:9100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tau3-sft-smoke",
    "messages": [{"role": "user", "content": "Say one short sentence about airline customer service."}],
    "temperature": 0.2,
    "max_tokens": 80
  }'
```

Pass criteria:

- server starts without `Qwen3_5TextConfig` mismatch,
- output is coherent English,
- no multilingual/gibberish corruption,
- no vision-encoder crash.

Stop if this fails. Debug SFT export/config before GRPO.

## Stage 3: One-Step Latest-VERL GRPO Smoke

Goal: prove latest VERL can load the exported SFT checkpoint, run Tau3 live interaction, and update once.

Suggested launch:

```bash
export TAU3_LIVE_USER_MODEL=us.anthropic.claude-sonnet-4-6
export TAU3_LIVE_RUNTIME=official_gym
export MODEL_PATH=/path/to/global_step_*/huggingface
export ENABLE_THINKING=true
export VLLM_LANGUAGE_MODEL_ONLY=true
export N_GPUS_PER_NODE=8
export ROLLOUT_TP_SIZE=1
export TRAIN_BATCH_SIZE=4
export ROLLOUT_BATCH_SIZE=2
export PPO_MINI_BATCH_SIZE=4
export VAL_N=1
export TOTAL_TRAINING_STEPS=1
export TOTAL_EPOCHS=1
export TEST_FREQ=1
export SAVE_FREQ=1
export LR=1e-6
export MAX_PROMPT_LENGTH=4096
export MAX_RESPONSE_LENGTH=512
export MAX_MODEL_LEN=8192
export ROLLOUT_TEMPERATURE=0.2
export ROLLOUT_TOP_P=0.95
export VAL_TEMPERATURE=0.2
export VAL_TOP_P=0.95
export PYTORCH_ALLOC_CONF=expandable_segments:True

ROLLOUT_DATA_DIR=outputs/grpo_vlm_sft_smoke/rollout_data \
bash run_local_tau3_grpo_live_p5.sh \
  datasets/tau3_live_airline_canonical_json \
  grpo_vlm_sft_smoke \
  structured \
  > logs/grpo_vlm_sft_smoke.log 2>&1
```

Pass criteria:

- vLLM args include `--language-model-only`.
- served/rollout model path is the SFT `huggingface` checkpoint path.
- decoded assistant output is not gibberish.
- Tau3 tool calls parse through `tau3_qwen`.
- `tau3_live_result` appears in rollout/reward extra fields.
- at least one actor update logs nonzero `actor/pg_loss` or nonzero `actor/grad_norm`.

Stop if decoded output is corrupted. Do not run GRPO/SDPO long jobs until that is fixed.

## Stage 4: Full SFT Only After Smoke

If Stages 1 to 3 pass, run the real SFT with 5K accepted train-split trajectories plus deterministic ID augmentation.

Target data:

- 5K to 10K accepted thinking-on airline trajectories before augmentation.
- Use only the official 30 train tasks from `datasets/tau3_live_airline_canonical_split.json`.
- Do not use the 20 canonical test tasks for SFT generation or SFT validation.
- The builder rejects canonical test-task validation by default; use `--train-only-holdout`.

Preferred full SFT:

```bash
MODEL_PATH=Qwen/Qwen3.5-4B \
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' \
NUM_GPUS=8 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=50 \
TEST_FREQ=25 \
MAX_LENGTH=32768 \
MAX_TOKEN_LEN_PER_GPU=32768 \
TRAIN_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=1 \
USE_LIGER=true \
bash scripts/tau3/run_tau3_verl_sft_full_thinking.sh \
  datasets/tau3_sft_thinking_train_only \
  qwen35_4b_vlm_export_5k_augmented
```

If full-length SFT OOMs at cross entropy:

- first confirm Liger/FLCE is actually active,
- then reduce `MAX_LENGTH` only as a fallback,
- record skipped/overlength fraction before accepting the run as comparable.

## Stage 5: Full GRPO Baseline

Run only after:

- SFT export is VLM-format,
- standalone vLLM serve is coherent,
- one-step GRPO smoke passes.

Use the existing recipe in `research/P5_RECIPES.md`, but set:

```bash
MODEL_PATH=/path/to/final_sft/global_step_*/huggingface
VLLM_LANGUAGE_MODEL_ONLY=true
SFT_LORA_ADAPTER_PATH=
```

Do not set any LoRA adapter variables.

## Stop Conditions

Stop and report instead of branching if:

- SFT `hf_model` export is text-only config.
- vLLM standalone serve fails on the exported SFT checkpoint.
- vLLM output is gibberish or multilingual garbage.
- VERL GRPO rollout uses the base model path instead of the exported SFT checkpoint.
- `tau3_live_result` is missing from reward extra fields.
- actor update metrics stay zero after a real update step.
- memory failure occurs before proving the checkpoint/rollout path.

## Write-Back Requirements

After the P5 smoke, update:

- `research/session_sync.md` with exact package versions, commit SHA, and pass/fail.
- `research/migration/qwen35_vlm_sft_rl_implementation_plan.md` if the plan changes.
- `logs/` and `outputs/` should remain untracked unless a concise summary is promoted into `research/`.

Do not copy bulky rollout data, checkpoints, wandb dirs, or evidence zips into this repo.
