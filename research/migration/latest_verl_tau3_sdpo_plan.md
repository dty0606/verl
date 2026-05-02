# Latest VERL tau3 SDPO Migration Plan

Date: 2026-05-02

## Purpose

Migrate the tau3 airline SFT/GRPO/SDPO workflow from the archived old SDPO fork into this latest-VERL fork with the smallest safe carry-over surface.

This fork should become the execution repo for current experiments. The old repo remains archive/source evidence:

- `D:/AI/chatgpt/codex/tmp/dty0606_SDPO`

## Why Migrate

The old fork reached a hard integration boundary:

- TRL full-parameter Qwen3.5 SFT could save a usable-looking checkpoint.
- Old VERL GRPO then failed around Qwen3.5 config handling, max position handling, and vLLM model loading.
- Manual conversion attempts produced mutually incompatible requirements: FSDP wanted text-config dimensions, while vLLM wanted `qwen3_5` composite config and then attempted multimodal setup.
- Latest VERL is the intended base because upstream fixes should align SFT checkpointing, FSDP loading, and vLLM rollout behavior.

## Migration Scope

Port only what is needed to run:

- tau3 airline dataset preprocessing with canonical train/test split.
- tau3 live official-gym rollout runtime.
- Qwen/tau3 action parsing needed for JSON and native Qwen XML-style tool calls.
- Bedrock user-simulator support for tau3 live evaluation/training.
- Full-parameter thinking SFT path.
- GRPO baseline from the VERL SFT checkpoint.
- Vanilla SDPO baseline from the same checkpoint family.

Do not migrate by bulk-copying:

- `research/evidence/`
- `outputs/`
- trajectory trees
- old checkpoints
- broad historical notebooks/logs

## Minimum Port Checklist

1. Identify old-repo files that are code, not evidence.
2. For each candidate file, decide whether latest VERL already has an equivalent.
3. Port tau3 support behind narrow names/configs so upstream VERL behavior stays intact.
4. Add smoke tests or CPU checks where possible before any P5 launch.
5. Run one tiny local/static verification pass.
6. Run one P5 smoke for each stage before full runs: SFT build, SFT train, GRPO rollout/update, SDPO target construction/update.

## Stage Gates

### Gate 1: tau3 data

Success means this fork can produce train/test parquet or JSON data from the canonical airline split without pulling old generated data.

Required invariant:

- Train IDs and test IDs must match `datasets/tau3_live_airline_canonical_split.json`.
- SFT data uses train split only.

### Gate 2: full-parameter SFT

Success means latest VERL trains or at least loads/saves a full Qwen/Qwen3.5-4B thinking SFT checkpoint without LoRA.

Preferred settings are summarized in `research/sft_5k_plan/FULL_PARAM_SFT_5K_PLAN.md`.

### Gate 3: GRPO

Success means the VERL SFT checkpoint loads into VERL rollout/training without manual Qwen3.5 config surgery and produces nonzero policy updates.

Current implementation status:

- `tau3_grpo_live.yaml` wires latest VERL PPO/GRPO config to the Tau3 reward function.
- `run_local_tau3_grpo_live_p5.sh` is the P5 entrypoint.
- Latest VERL's `tool_agent` has optional Tau3 interaction support via `interaction_config_path`.
- The `tau3_qwen` parser handles Qwen XML and legacy JSON tool-call outputs.

Abort and debug if:

- decoded rollout text is gibberish,
- vLLM attempts unwanted multimodal processor setup,
- prompt/response token IDs are not integer-normalized,
- `actor/pg_loss` or `actor/grad_norm` remains zero after a real update step.

### Gate 4: vanilla SDPO

Success means vanilla peer-solution SDPO constructs nonempty targets and updates the actor.

Current implementation status:

- Not runnable yet.
- The old fork's `actor.policy_loss.loss_mode=sdpo` hook was deliberately not ported because latest VERL does not support it.
- `run_local_tau3_sdpo_live_p5.sh` fails fast and points to `research/migration/sdpo_latest_verl_port_notes.md`.

Abort and debug if these stay unhealthy for several steps:

- `self_distillation/empty_target_batch=1.0`
- `self_distillation/success_group_fraction=0.0`
- `actor/pg_loss=0.0`
- `actor/grad_norm=0.0`

### Gate 5: memory/feedback SDPO

Do not start until GRPO and vanilla SDPO baselines are understood in latest VERL.

## Continuity Decisions

- Treat old SDPO as archive/evidence, not the active execution base.
- Prefer a single-framework VERL pipeline over TRL SFT -> old VERL RL.
- Prefer full-parameter SFT over LoRA for the main path because old LoRA rollout integration was corrupted.
- Use GRPO and vanilla SDPO as the first comparable RL baselines.
- Keep feedback/memory SDPO as later variants with retrieval controls, not part of the migration minimum.
