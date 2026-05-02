# Full-Parameter Thinking SFT Plan: 5K Airline Trajectories

Date: 2026-05-02

This is a compact migration copy of the old-repo SFT plan. The old detailed notes and samples remain in:

- `D:/AI/chatgpt/codex/tmp/dty0606_SDPO/research/sft_5k_plan/FULL_PARAM_SFT_5K_PLAN.md`

This latest-VERL fork should own the next executable SFT run.

## Goal

Train Qwen/Qwen3.5-4B with full-parameter thinking SFT on real accepted tau3 airline train-split trajectories, then use the resulting VERL checkpoint directly for GRPO and SDPO in this same fork.

## Data Target

- Target real accepted trajectories: about 5,000.
- Source: Bedrock Opus 4.6 agent thinking-on tau3 airline trajectories.
- User simulator: Bedrock Sonnet 4.6.
- Split rule: train split only from `datasets/tau3_live_airline_canonical_split.json`.
- Do not use canonical test tasks for SFT generation.
- ID augmentation may be used for training scale, but the 5K gate refers to real accepted source trajectories, not augmented rows.

Historical old-repo status:

- 1K full-parameter pilot completed and overfit.
- 5K generation was in progress in the old workflow.
- Tasks 7 and 39 were excluded from success expectations after appearing genuinely unsolvable by Opus in that setup.

## Recommended Dataset Build

Use `scripts/tau3/build_protocol_sft_data.py` to convert accepted trajectory JSON/JSONL artifacts into latest-VERL SFT parquet rows.

Required options/invariants:

- Include thinking traces.
- Use `--train-only-holdout` or equivalent.
- Preserve original train/test split manifest.
- Prefer `--id-transform randomize --id-variants 3 --include-original-id-variant` after the real accepted-trajectory gate is met.
- Record a manifest with accepted rows, rejected rows, thinking-trace coverage, token-length coverage, and task coverage.

Example:

```bash
python scripts/tau3/build_protocol_sft_data.py \
  --input /path/to/accepted_tau3_train_trajectories \
  --output-dir datasets/tau3_sft_thinking_train_only \
  --train-only-holdout \
  --include-thinking-traces \
  --id-transform randomize \
  --id-variants 3 \
  --include-original-id-variant \
  --overwrite
```

The builder defaults to `datasets/tau3_live_airline_canonical_split.json` and refuses canonical test-task rows by default. Use train-split-only trajectory roots with `--train-only-holdout`; `--allow-canonical-test-validation` is for smoke/debug runs only.

## Training Defaults

Preferred initial run:

```bash
MODEL_PATH=Qwen/Qwen3.5-4B
FULL_FINETUNE=true
ENABLE_THINKING=true
MAX_LENGTH=32768
LR=1e-5
TOTAL_EPOCHS=1
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NUM_GPUS=8
GRADIENT_CHECKPOINTING=true
ATTN_IMPLEMENTATION=flash_attention_2
USE_LIGER=true
FSDP="full_shard auto_wrap"
FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP=Qwen3_5DecoderLayer
```

Rationale:

- One epoch is preferred because the 1K pilot overfit with more training.
- `MAX_LENGTH=32768` covered about 90% of the old 269-sample thinking trajectory set.
- Liger fused linear cross entropy was required to avoid materializing huge Qwen3.5 vocabulary logits at long sequence lengths.

Fallbacks:

- If 32K OOMs, retry at `MAX_LENGTH=24576`.
- If 32K has substantial headroom, consider a second 50K run for long-tail thinking traces.
- Do not change multiple knobs at once before preserving the failed run manifest.

## Must-Pass Checks Before RL

- Checkpoint was produced by latest VERL or by code in this latest-VERL fork.
- Checkpoint loads for inference through the same rollout stack planned for GRPO.
- No manual `qwen3_5` / `qwen3_5_text` config surgery is needed.
- Sample generations are readable and contain no LoRA/vLLM gibberish pattern.
- Chat template handling supports tau3 conversations and thinking traces.

## Hand-Off To RL

After SFT passes:

1. Run a small GRPO smoke from the full-parameter SFT checkpoint.
2. Verify rollout decoding, tau3 tool parsing, reward logging, and nonzero actor update metrics.
3. Launch the comparable GRPO baseline.
4. Run vanilla SDPO from the same checkpoint family.
5. Only then consider memory/feedback SDPO.
