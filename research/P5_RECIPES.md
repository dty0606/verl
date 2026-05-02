# P5 Recipes: Latest VERL tau3 Fork

Date: 2026-05-02

This file is intentionally minimal. The old detailed recipe book remains in:

- `D:/AI/chatgpt/codex/tmp/dty0606_SDPO/research/P5_RECIPES.md`

Use the old file as historical reference only. Commands copied from it may reference old paths, old S3 prefixes, old vLLM behavior, or old VERL patch assumptions.

## Current Rule

For new SFT, GRPO, and SDPO runs, execute from this latest-VERL fork or from a remote clone synced from it.

Do not resume the old pattern:

- TRL SFT in old repo
- manual Qwen3.5 config patching
- old VERL GRPO/SDPO launch

That path is deprecated because it repeatedly failed at Qwen3.5 checkpoint/config/vLLM boundaries.

## Remote Layout Placeholder

Use an explicit latest-VERL remote directory name so logs and checkpoints are not confused with the old archive:

```bash
LATEST_VERL_ROOT=~/verl_tau3_sdpo
OLD_SDPO_ARCHIVE=~/SDPO-qwen35
```

The exact P5 sync path and S3 prefix are TODO until the first remote setup in this fork.

## First Remote Smoke Order

1. Verify latest VERL environment, Transformers, vLLM, flash-attn, Liger, and Bedrock credentials.
2. Build or inspect tau3 airline canonical split data in this fork.
3. Run a tiny full-parameter SFT smoke on Qwen/Qwen3.5-4B.
4. Load that checkpoint in the rollout path and generate a few tau3 rollouts.
5. Run a GRPO smoke and verify nonzero update metrics.
6. Port vanilla SDPO through latest VERL's distillation stack.
7. Run a vanilla SDPO smoke and verify nonempty target metrics.

## Current GRPO Smoke Skeleton

```bash
TAU3_LIVE_USER_MODEL=us.anthropic.claude-sonnet-4-6 \
TAU3_LIVE_RUNTIME=official_gym \
MODEL_PATH=/path/to/latest_verl_sft_checkpoint_or_Qwen/Qwen3.5-4B \
TOTAL_TRAINING_STEPS=1 TOTAL_EPOCHS=1 \
TRAIN_BATCH_SIZE=1 ROLLOUT_BATCH_SIZE=1 PPO_MINI_BATCH_SIZE=1 \
N_GPUS_PER_NODE=1 VAL_N=1 \
./run_local_tau3_grpo_live_p5.sh \
  datasets/tau3_live_airline_smoke_json \
  smoke \
  json
```

Before using that command, build the smoke parquet with
`examples/data_preprocess/tau3_live_multiturn.py`.

## Historical P5 Notes To Preserve

- P5 target was SageMaker CE `p5.48xlarge` with 8x H100 80GB.
- Bedrock Opus 4.6 thinking generation needed throttling-aware shard limits; 28 parallel shards was safer than 100+.
- Thinking generation used `MAX_RESPONSE_TOKENS=8192`.
- Airline rewards are DB-state based; retail has NL assertion rewards and needs an LLM judge.
- Always separate train and test split usage when generating SFT data.
