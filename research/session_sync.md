# Project: latest-VERL tau3 GRPO/SDPO fork

Last Updated: 2026-05-02

## Stable State

This repository is the execution fork for rebuilding the tau3 SFT -> GRPO -> SDPO stack on latest VERL.

The old SDPO fork remains the archive and source-evidence repo:

- `D:/AI/chatgpt/codex/tmp/dty0606_SDPO`

Use the old repo to inspect historical code, launch recipes, evidence summaries, and trajectory provenance. Do not copy bulky `research/evidence/`, `outputs/`, or trajectory directories into this fork unless a later task explicitly asks for a narrow artifact.

## Current Decision

Run the next end-to-end work in latest VERL, not the old SDPO fork.

Reason:

- The old fork can generate useful tau3 data and run TRL SFT experiments, but full-parameter Qwen3.5 SFT checkpoints hit repeated old-VERL/vLLM config incompatibilities before GRPO.
- Qwen3.5 is represented as a VLM-style `qwen3_5` model in current Transformers, while causal-LM training and vLLM rollout paths need coordinated text-config handling.
- Latest VERL is expected to contain the upstream Qwen3.5, SFT checkpoint, rollout, and max-model-len fixes needed for one-framework SFT -> RL.

## Current Execution Shape

The intended path is:

1. Port only the tau3 and SDPO customizations needed by this project into latest VERL.
2. Build/verify tau3 airline train/test datasets inside this fork.
3. Run full-parameter thinking SFT in VERL on Qwen/Qwen3.5-4B.
4. Use the VERL-produced checkpoint directly for GRPO.
5. Use the same VERL-produced checkpoint/run stack for vanilla SDPO.
6. Only after GRPO and vanilla SDPO are understood, revisit memory/feedback SDPO variants.

## Current Migration Status

- Latest-VERL fork and branch are active at `D:/AI/chatgpt/codex/tmp/verl_tau3_sdpo`.
- Private remote for Kiro/P5 fresh clones: `https://github.com/dty0606/verl_tau3_sdpo.git`.
- Fresh Kiro/P5 onboarding starts with `START_HERE_FOR_KIRO.md`; clone both the old archive repo and this new private execution repo.
- Minimal Tau3 runtime files, tool schema, action parser, feedback/reward scorer, and migration docs have been copied into this fork.
- Latest VERL's built-in `tool_agent` now has an optional Tau3 interaction path through `multi_turn.interaction_config_path`.
- `tau3_qwen` tool parsing is registered for Qwen XML and legacy JSON tool-call outputs.
- `run_local_tau3_grpo_live_p5.sh` is the current runnable RL baseline entrypoint.
- `run_local_tau3_sdpo_live_p5.sh` intentionally fails fast until vanilla SDPO is ported through latest VERL's distillation stack.
- Full-parameter thinking SFT should use latest VERL's native SFT trainer, not the old TRL script path.
- 2026-05-02 QC patch: interaction config is optional-safe for non-Tau3 `tool_agent` configs, Tau3 tool/session runtime routing now follows the active interaction manager, launch scripts are executable, and protocol-SFT building now rejects missing thinking traces and canonical test-task validation by default.
- 2026-05-02 checkpoint-format decision: use VLM-format `Qwen/Qwen3.5-4B` SFT export plus vLLM `--language-model-only`; do not depend on text-only `Qwen3_5ForCausalLM` checkpoints for RL rollout until upstream vLLM support is clearly merged and verified. See `research/migration/qwen35_vlm_sft_rl_implementation_plan.md`.
- 2026-05-02 current external state: user is generating 10K thinking-on Tau3 SFT trajectories on P5. Treat generated trajectories/checkpoints/logs as S3 artifacts, not Git artifacts; promote only concise manifests/summaries into this repo.

## Evidence Carried Forward

From the old repo:

- Base-model tau3 eval suggested Qwen3.5-4B is the best primary seed by GPU-hour; larger Qwen variants mostly changed reliability rather than the no-thinking best@4 ceiling on the tested slice.
- LoRA SFT/RL integration was not reliable enough for the next main path because rollout/vLLM weight-sync corruption produced gibberish decoded outputs.
- A 1K full-parameter thinking SFT pilot completed, but overfit; the next SFT should use the 5K+ real accepted thinking trajectory plan and one epoch.
- Thinking trajectory generation should use train-split-only airline tasks from `datasets/tau3_live_airline_canonical_split.json`; canonical test tasks must not be used for SFT generation.
- Tasks 7 and 39 were treated as genuinely unsolvable by the Bedrock Opus trajectory generator in the old run and should be excluded from accepted-data expectations unless this is revalidated.

## Open Questions

- Does latest VERL SFT accept Qwen3.5 thinking traces and tau3 assistant-first conversations without local template hacks on P5?
- Can latest VERL export a VLM-format `hf_model` for Qwen3.5-4B SFT, with `model_type=qwen3_5` and `Qwen3_5ForConditionalGeneration` preserved?
- Can vLLM serve that exact SFT export with `--language-model-only`, and can the current Tau3 interaction-enabled `tool_agent` pass a one-step official-gym GRPO rollout smoke on P5?
- What exact SDPO implementation path should be used in latest VERL after GRPO works: adapt upstream on-policy distillation or implement a minimal vanilla SDPO target builder first?

## Guardrails

- Keep this fork focused on execution and migration docs; do not turn it into an evidence mirror.
- Mark old-repo numbers as historical unless re-run in this latest-VERL fork.
- SFT, GRPO, and SDPO results are comparable only if they run end-to-end in this fork or in a single pinned latest-VERL remote clone.
- Keep generated outputs out of `research/` unless they are concise summaries or plans.
