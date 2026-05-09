# Project: latest-VERL tau3 GRPO/SDPO fork

Last Updated: 2026-05-09

### 2026-05-09 SDPO NaN fail-fast diagnostics

- W&B run `0xcb5ugy` completed 100/100 steps but became numerically invalid from step 3 onward: `actor/loss`, `actor/grad_norm`, and `actor/self_distillation/teacher_topk_mass` were `NaN`, while `actor/self_distillation/student_topk_mass` stayed finite.
- Codex added fail-fast diagnostics to stop at the first non-finite SDPO value instead of letting W&B finish quietly with corrupted metrics.
- New runtime checks cover: actor/teacher finite parameters before EMA update, FSDP full-logit/top-k logprob finite checks, trainer return-boundary finite checks, and actor-loss teacher top-k logprob finite checks.
- Enable with `SDPO_EMA_FINITE_CHECK=1` and `SDPO_FAIL_FAST_NONFINITE=1`; keep `SDPO_LOGPROB_DIAGNOSTICS=1`, `SDPO_CUDA_MEMORY_DIAGNOSTICS=1`, and `HYDRA_FULL_ERROR=1` for the short P5 reproduction run.
- Follow-up commit `293a986b` strengthens the NaN probe from passive observation into first-touch classification. It adds fail-fast checks for raw/scaled FSDP logits, temperature, actor-side student logits/logprobs/gathered top-k, top-k mass validity, tail/JSD terms, selected SDPO per-token loss, gradients before optimizer step, actor params after optimizer step, and teacher params after EMA update.
- The same commit fixes a propagation hazard where masked-out NaNs could still contaminate sequence-level aggregation via `0 * NaN`, and documents the bounded probe recipe in `research/sdpo_nan_probe_protocol.md`.
- Prior diagnosis before the 30-step probe: the old NaN looked teacher-side (`teacher_topk_log_probs` / teacher logsumexp / teacher gathered top-k / tail-JSD), with masked loss math as a propagation amplifier. The first-touch rerun below supersedes that guess for the reproduced failure.
- Kiro/P5 reran the 30-step fail-fast recipe and caught a deterministic first-touch error before step 1 completed: `Invalid SDPO top-k log-prob mass for student_topk_log_probs shape=(1, ~5.8k, 100) first_bad=[0, 3767] max_log_mass=4.526`. The same bad position appeared on all 8 ranks.
- Updated diagnosis: the immediate bug is not EMA-teacher parameter corruption. Full-sequence SDPO top-k tensors are zero-filled outside response-prediction positions; actor update was validating/gathering those prompt/non-response placeholders as if they were real top-k probability slices. A zero-filled top-k row means roughly 100 tokens with logprob 0, so top-k mass is far above 1 and can poison JSD/mass metrics.
- Local patch now derives the no-padding response-prediction-position mask inside `_sdpo_topk_logits_processor`, validates top-k IDs only on those positions, and replaces non-response placeholder rows with a finite degenerate distribution before JSD validation. Regression test: `test_sdpo_topk_logits_processor_ignores_non_response_prediction_rows`.

### 2026-05-09 True-8K SDPO OOM and 6K rollout-collection recipe

- Pulled W&B run `oe1zswum` (`SDPO-vllm-v1-original-sdpo-safe`): this was the direct true-8K recipe, with `data.max_response_length=8192`, `max_model_len=16384`, and `tau3.sdpo.max_reprompt_len=4096`.
- The run logged through `training/global_step=21` and then OOMed at step 22 during actor update/backward (`loss.backward()`), not checkpoint save and not teacher/student logprob. Step-22 pre-logprob diagnostics showed `actor_student response_mask_mean=3176.45`, `response_mask_max=7737`, and `attention_mask_mean=8190.55`.
- Kiro commit `ef76032e` added Recipe 11 with `ROLLOUT_DATA_DIR`, fixing the missing rollout-data issue from the prior run. Codex then revised Recipe 11 to a safer 6K response-cap rollout collection run: `MAX_RESPONSE_LENGTH=6144`, `MAX_MODEL_LEN=14336`, explicit logprob/actor token caps `14336`, diagnostics enabled, and fallback to `4096/12288` if it OOMs before step 30.
- Codex also added actor-update CUDA diagnostics in `ActorRolloutRefWorker.update_actor`, so future actor-backward OOMs should emit `[sdpo_cuda_memory] phase=actor_update event=exception` in the nohup log and successful steps should log `actor/cuda_memory/*`.

### 2026-05-09 Tau3 SDPO OOM diagnosis and safe audit recipe

- A 3-step original SDPO smoke on P5 passed after the EMA teacher device/checkpoint fixes: actor checkpoint, optimizer shards, and `sdpo_ema_teacher` shards were saved.
- A 100-step original SDPO audit later OOMed at step 7 during `actor_rollout_ref_compute_log_prob`, with PyTorch trying to allocate ~26.67 GiB while ~26.60 GiB was free on GPU 0. The last healthy steps had ordinary response lengths (~2K mean, ~3.1K max), so the risk is from full-vocab logit materialization in SDPO logprob/top-k paths, not simply visible response length.
- Local Codex added opt-in diagnostics and launcher knobs, pending/now intended for Git sync: `SDPO_LOGPROB_DIAGNOSTICS`, `SDPO_CUDA_MEMORY_DIAGNOSTICS`, logprob dynamic-batch env passthrough, and pre-logprob batch length summaries. These are for debugging only and should not change training semantics when disabled.
- Conservative next P5 run while debugging: keep `SDPO_ARM=original`, `SDPO_MEMORY_ENABLED=false`, `SDPO_TEACHER_BACKEND=ema_ref`, but cap `MAX_RESPONSE_LENGTH=8192`, `SDPO_MAX_REPROMPT_LEN=4096`, `MAX_MODEL_LEN=16384`, set `ROLLOUT_GPU_MEMORY_UTILIZATION=0.50`, and enable the two diagnostics env vars for the first safe audit.
- Interpretation note: current successful-peer selection is deterministic but not semantic ranking. It collects eligible successes by current row order and uses `solution_idxs[0]`; sequence-length balancing may affect that order, so future ablations should compare `first_success` against `random_success` and `shortest/cleanest_success` for multi-turn agents.

### 2026-05-07 Multi-turn SDPO/GRPO masking failure-mode note

- Added `research/multiturn_sdpo_mask_failure_modes.md` as a paper-writing note on transferring GRPO/SDPO to thinking-on multi-turn tool agents.
- Key framing: original SDPO handles wrong-but-scorable rollouts, while Tau3 exposes wrong-and-corrupted rollouts with runaway thinking, repeated tool loops, or response-budget exhaustion.
- The note separates paper-aligned SDPO behavior from Tau3-specific hardening: `self_distillation_mask` selects rows with teacher-side corrective context, while Tau3 additionally needs env-error exclusion and target-side overlong/repetition/open-think hygiene.
- Important claim boundary: official SDPO strips `<think>` from successful demonstrations, but target rollout thinking is still scored unless extra masks are added; the authors' motivation for demo-thinking stripping is not explicitly stated in the paper/docs.

### 2026-05-07 Note-SDPO memory bank builder

- Added `scripts/tau3/build_note_sdpo_bank.py` as the production pipeline for teacher-written decision-note memory banks.
- The pipeline emits sanitized note-writer prompts from train rollouts, optionally calls an OpenAI-compatible teacher endpoint, then compiles/validates accepted notes into the existing `tau3.sdpo.memory.path` JSONL card format.
- Deterministic code handles source selection, redaction, schema validation, leakage rejection, and memory-momentum merging; the teacher/self-distilled model writes the semantic note.
- Added `research/note_sdpo_bank_pipeline.md` with P5 commands for prompt emission, teacher note writing, bank compilation, and SDPO integration flags.
- Added `tests/utils/test_tau3_note_sdpo_bank_builder.py` covering redaction, leakage rejection, memory-bank compatibility, and support-count merging.

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
- `run_local_tau3_sdpo_live_p5.sh` is now the vanilla SDPO smoke/baseline launcher; memory/DENSE SDPO remains disabled.
- Full-parameter thinking SFT should use latest VERL's native SFT trainer, not the old TRL script path.
- 2026-05-02 QC patch: interaction config is optional-safe for non-Tau3 `tool_agent` configs, Tau3 tool/session runtime routing now follows the active interaction manager, launch scripts are executable, and protocol-SFT building now rejects missing thinking traces and canonical test-task validation by default.
- 2026-05-02 checkpoint-format decision: use VLM-format `Qwen/Qwen3.5-4B` SFT export plus vLLM `--language-model-only`; do not depend on text-only `Qwen3_5ForCausalLM` checkpoints for RL rollout until upstream vLLM support is clearly merged and verified. See `research/migration/qwen35_vlm_sft_rl_implementation_plan.md`.
- 2026-05-02 current external state: user is generating 10K thinking-on Tau3 SFT trajectories on P5. Treat generated trajectories/checkpoints/logs as S3 artifacts, not Git artifacts; promote only concise manifests/summaries into this repo.

### 2026-05-02 Kiro workspace setup

- Kiro workspace: `/Volumes/workplace/sdpo/`
- Cloned `dty0606_SDPO` on branch `codex/sdpo-tau3-transformers5-qwen35`
- Cloned `verl_tau3_sdpo` on branch `main` at `3e2ea740`
- AWS identity: `arn:aws:iam::343006728691:user/tianyd-sm`
- Old checkpoints on P5 are irrelevant — retraining everything from scratch in latest VERL
- Old S3 prefix `s3://tianyd-rlvr-research/SDPO/` contains old trajectory generations (v1: 450 rollouts, v2: 480 rollouts) and scripts; no checkpoints
- 10K thinking-on generation is P5-local only, not yet synced to S3
  - v1 (seeds 100-103, 30 shards): 269 successes
  - batch1 (seeds 600-849, 28 shards): 1,345 successes
  - batch2 (seeds 951-1040, 28 shards): 2,607 successes
  - batch3 (seeds 1041-1250, 28 shards): running, ~3,400+ successes so far
  - Combined: ~7,600+ successes and climbing toward 10K target
  - All under `~/SDPO-qwen35/outputs/tau3_protocol_candidates_thinking_*` on P5
  - Generated by old repo scripts (`eval_tau3_bedrock_agent.py`); output format verified compatible with new repo's `build_protocol_sft_data.py`
  - P5 env: `sdpo-qwen35` conda env, old repo at `~/SDPO-qwen35/`

### S3 Prefix Convention

```bash
# New latest-VERL artifacts go here (not under SDPO/)
export S3_PREFIX=s3://tianyd-rlvr-research/tau3-sdpo/latest-verl

# Subdirectories:
# $S3_PREFIX/generated_sft/     — accepted trajectory JSONs
# $S3_PREFIX/checkpoints/       — SFT/GRPO/SDPO checkpoints (only when cross-machine needed)
# $S3_PREFIX/rollout_data/      — GRPO/SDPO rollout dumps
# $S3_PREFIX/logs/              — promoted smoke/run logs
```

### Code verification (Kiro-side, 2026-05-02)

Verified in the new repo:
- `scripts/tau3/run_tau3_verl_sft_full_thinking.sh` — wired for `hf_model` export via `CHECKPOINT_SAVE_CONTENTS`
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py` — `save_checkpoint` detects `ForConditionalGeneration` in architectures and uses vision2seq auto model class, preserving VLM config
- No `Qwen3_5TextConfig` conversion anywhere in the new repo (old repo had this in patched `fsdp_workers.py`)
- `run_local_tau3_grpo_live_p5.sh` — passes `VLLM_LANGUAGE_MODEL_ONLY` through to vLLM engine kwargs
- `verl/trainer/config/tau3_grpo_live.yaml` — rollout backend is `vllm`, multi-turn format is `tau3_qwen`, reward function points to `tau3_live.py`
- `run_local_tau3_sdpo_live_p5.sh` — correctly fails fast with port instructions
- `examples/data_preprocess/tau3_live_multiturn.py` — reads canonical split manifest, builds train/test parquet
- `research/P5_RECIPES.md` — updated with concrete smoke sequence commands

### 2026-05-02 Codex follow-up patch after Kiro commit `08c34148`

- Fixed thinking-trace alignment after stripping the leading Tau3 assistant greeting: kept assistant turns now map directly to `turns[*].thinking_text`.
- Fixed Qwen2-VL lazy flash-attn import by invoking the lazy loader inside `_custom_flash_attention_forward` before flash-attn capability checks or varlen calls.
- Updated `scripts/tau3/test_thinking_stitching.py` so the diagnostic checks user-first ordering and catches shifted thinking traces.
- Local Windows verification: `python -m py_compile` passed for the touched files; synthetic validator check confirmed the first real assistant/tool call receives the first thinking trace.
- Next P5 action remains: pull latest main, run the thinking-stitching diagnostic on one real generated trajectory, then retry the tiny SFT export smoke.

### 2026-05-02 Codex QC hardening after commit `d3a1837e`

- Commit `7711222a` handles serialized empty `tool_calls=[]` on the stripped leading greeting, not only `tool_calls=None`.
- `scripts/tau3/test_thinking_stitching.py` now asserts assistant-turn count equals `turns` count before checking per-turn thinking text, so shifted/missing thinking supervision fails loudly.
- Local verification repeated: `python -m py_compile` passed; synthetic Tau3 trajectory check passed with leading greeting, `tool_calls=[]`, one tool-call assistant turn, and one final assistant turn.
- Kiro/P5 should pull private `main` at or after `7711222a` before retrying the SFT smoke.

### 2026-05-02 Codex SimpleSFTDataset tool-call normalization patch

- Patched `verl/utils/dataset/simple_sft_dataset.py` to normalize messages before Qwen chat-template rendering: strip `None` keys, strip leading assistant greeting, remove empty `tool_calls`, and coerce assistant `tool_calls[].function.arguments` from JSON string to dict for training-template rendering.
- Added nested OpenAI-style tool-call rendering first, with flat Qwen-style fallback if the nested shape fails.
- Added `SimpleSFTDataset.audit_item()` plus `data.audit_samples` support to decode masked assistant spans and fail if expected `<think>` or tool names are missing.
- Added `scripts/qwen35/diagnose_tau3_sft_template.py` as a pre-SFT parquet/template audit for 10-20 real rows.
- Local Windows verification: `python -m py_compile` passed for the patched dataset and diagnostic script; synthetic helper check confirmed assistant greeting strip and JSON-string tool arguments -> dict normalization. P5 remains the source of truth for Qwen3.5 tokenizer/runtime behavior.
- Next P5 action: run the diagnostic script on `datasets/tau3_sft_thinking_train_only`; only if it passes, retry tiny SFT smoke with `+data.custom_cls.path=verl/utils/dataset/simple_sft_dataset.py`, `+data.custom_cls.name=SimpleSFTDataset`, and `+data.audit_samples=2`.

### 2026-05-02 Codex QC follow-up after `aad2124c`

- Fixed `SimpleSFTDataset` to choose nested-vs-flat tool-call message shape once per row and reuse that same shape for full rendering and all assistant mask prefix/suffix renders.
- Local verification repeated: `python -m py_compile`, `git diff --check`, and synthetic helper normalization check passed.

### 2026-05-02 Turn-per-row SFT decision

- P5 audit with `max_length=32768` proved the full-trajectory `SimpleSFTDataset` mask is structurally wrong for Qwen3.5 thinking-on SFT: Qwen3.5 strips historical assistant reasoning in full multi-turn renders, causing fewer masked spans than assistant turns and missing `<think>` content.
- Public precedent check: `inclusionAI/AReaL-tau2-data` uses `messages` history plus one current `answer` target with `thinking` and `tool_calls`. VERL docs also document Qwen/QwQ/Qwen3 historical reasoning stripping.
- Added `--sft-format turn` to `scripts/tau3/build_protocol_sft_data.py`, which expands each accepted trajectory into one row per assistant target:
  - `messages`: prior history only, with historical assistant reasoning stripped.
  - `answer`: current assistant target only, with `thinking`, `content`, and optional `tool_calls`.
  - `assistant_turn_index` and `source_message_index`: provenance for audit/balancing.
- Added `verl/utils/dataset/turn_sft_dataset.py`, a custom VERL SFT dataset that renders `messages` with `add_generation_prompt=True`, renders `messages + [answer]`, and masks only the answer suffix.
- Added `research/literature/tau2_areal_sft_notes.md` to record the AReaL Tau2 precedent.
- `research/P5_RECIPES.md` now uses `TurnSFTDataset` and `--sft-format turn` for SFT smoke/full SFT.

### 2026-05-02 Pre-tokenized turn SFT path

- Kiro commit `d5bc3a13` added offline pre-tokenization for turn-per-row SFT after P5 showed on-the-fly `apply_chat_template` training at ~88 seconds/step.
- `scripts/tau3/pretokenize_turn_sft.py` now writes `input_ids` and `loss_mask` parquet files for `PretokenizedSFTDataset`, so training does no chat-template rendering in the hot loop.
- Codex QC hardening after `d5bc3a13`:
  - Pre-tokenization fails fast on row errors unless `--allow-errors` is explicitly set.
  - `left` truncation is implemented and is the default for offline pre-tokenization, preserving the current assistant answer at the sequence end.
  - Empty loss masks after truncation are rejected.
  - If truncation leaves `loss_mask[0] == 1`, the first label is dropped because latest VERL's no-padding SFT loss rolls the flattened jagged mask by one token.
  - `PretokenizedSFTDataset` validates 1-D integer `input_ids`, binary `loss_mask`, equal lengths, non-empty labels, and unknown truncation modes.
- Recommended next P5 sequence:
  - Pre-tokenize full train-only turn dataset with `--max-length 32768 --truncation left --workers 8`.
  - Verify `manifest.json` has `errors == 0`, nonzero `output_rows`, and nonzero `avg_labeled_tokens` for train/test.
  - Run a 2-step pre-tokenized SFT smoke with `TRAIN_BATCH_SIZE=32`, `USE_LIGER=true`, `TRUNCATION=error`, and `trainer.total_training_steps=2`.
  - If smoke step time is healthy and checkpoint export is VLM-format, train all rows for one epoch with `SAVE_FREQ=1000`, `TEST_FREQ=1000`, and no row cap.
- Sampling decision: use all ~75,949 train turn rows by default. Only rebuild with `--max-rows-per-task` if the manifest shows severe task skew, e.g. one task contributes more than ~10-12% of train rows or max/median task count exceeds ~2.5x.

### 2026-05-02 Balanced 5K SFT pilot decision

- P5 timing corrected the old TRL comparison: the 1K full-trajectory TRL run was ~52s/step at `MAX_LENGTH=16384`, not ~1.8s/step. Latest-VERL VLM SFT at ~80s/step is slower per step, but the real blocker is the all-row dataset size (~75K train rows).
- Fused Triton kernels did not improve the latest-VERL Qwen3.5 VLM SFT run (~116s/step), so do not continue engine/kernel tuning for this pass.
- Current decision: train a balanced VLM-format SFT pilot first, not the full 75K turn-row set.
- Added `scripts/tau3/curate_turn_sft_subset.py`:
  - samples from turn-per-row SFT parquet with deterministic task-balanced selection,
  - filters to canonical train tasks by default,
  - supports explicit denylist (`--deny-task-ids 7`; add `39` only if re-confirmed),

  - requires reward=1 and target thinking by default,
  - caps ID variants per source assistant turn,
  - preserves tool/text/turn-position strata by round-robin sampling,
  - writes manifest counts by task, target kind, tool name, turn bucket, and shortfall.
- `pretokenize_turn_sft.py` now preserves lightweight metadata columns such as `task_id`, `assistant_turn_index`, `target_kind`, and `curation_stratum` in the pre-tokenized parquet for audit/debugging. `PretokenizedSFTDataset` ignores extra columns.
- Recommended P5 path:
  - curate `datasets/tau3_sft_thinking_balanced_5k` from `datasets/tau3_sft_thinking_train_only` with `--deny-task-ids 7 --train-rows-per-task 170 --val-rows-per-task 20`,
  - pretokenize to `datasets/tau3_sft_balanced_5k_pretokenized` at `MAX_LENGTH=24576`,
  - train latest-VERL VLM SFT with `engine.use_torch_compile=False`, no fused kernels, batch 32, one epoch.
- Expected scale if only task 7 is denied: 29 usable train tasks, ~4,930 train rows, ~580 validation rows, ~155 optimizer steps at global batch 32, roughly 3-4 hours at the observed latest-VERL baseline.

### 2026-05-03 Vanilla SDPO latest-VERL port

- `run_local_tau3_sdpo_live_p5.sh` now launches a vanilla Tau3 SDPO baseline instead of failing fast.
- Added `verl/trainer/config/tau3_sdpo_live.yaml`, based on the GRPO Tau3 live config but with `actor.policy_loss.loss_mode=sdpo`.
- Latest trainer path now builds SDPO teacher reprompts after Tau3 rewards are extracted:
  - selects failed samples with usable environment feedback,
  - uses the original prompt plus feedback, no memory/retrieval,
  - scores the original response under that reprompt with the actor log-prob path,
  - attaches `teacher_logprobs`, `self_distillation_mask`, and `self_distillation_loss_mask` to the training batch.
- Latest worker loss now supports response-token reverse-KL SDPO through `policy_loss.loss_mode=sdpo`.
- Local Windows QC: `python -m py_compile` passed for `ray_trainer.py`, `losses.py`, and `actor.py`; full Hydra compose could not run locally because this Windows environment lacks `omegaconf`.
- Next P5 action: run Recipe 9 one-step SDPO smoke and verify nonzero `self_distillation/reprompt_sample_fraction` for failed rollouts with feedback plus finite `actor/pg_loss`.

### 2026-05-03 Paired Tau3 evaluation script

- Added `scripts/tau3/eval_tau3_paired.sh` for fair model comparison on the same `task x trial` grid.
- Default grid is canonical airline test20, `EVAL_SEEDS="42 123 456"`, and `EVAL_N=4`; because `eval_tau3_base_models.py` uses `seed + sample_idx`, this produces 12 rollouts per task across the three seed groups.
- The wrapper now derives GPU task groups from `TEST_TASK_IDS`, so recorded config and launched tasks cannot silently diverge.
- It records `eval_config.json`, clears each seed output directory before rerun, checks every seed produces `len(TEST_TASK_IDS) * EVAL_N` trajectories, and writes `results.json`.
- Aggregation now reports tau-style finite-sample `pass^k` using `comb(successes, k) / comb(trials, k)` rather than `p1 ** k`.
- Runtime note: the wrapper still defaults `EVAL_SCRIPT=$HOME/SDPO-qwen35/scripts/eval_tau3_base_models.py`, so P5/Kiro must either keep the old repo at that path or override `EVAL_SCRIPT` explicitly.

### 2026-05-03 Full-trajectory thinking SFT smoke lane

- Paired diagnostics on task 30/37 suggest the balanced 5K turn-row SFT is mechanically valid but loses policy/endpoint behavior relative to base/old LoRA; likely cause is context/template mismatch rather than generic convergence.
- Current decision: restore the old LoRA-style SFT contract first: one successful full trajectory per row, thinking-on, assistant-only loss across all assistant turns.
- Added `verl/utils/dataset/qwen35_preserve_thinking_template.py`, a local ChatML/Qwen XML tool-call template that preserves historical assistant `<think>` blocks.
- Added `scripts/tau3/pretokenize_full_traj_sft.py`, which reads `--sft-format trajectory` parquet, installs the preserve-thinking template in-memory, audits decoded assistant spans, and writes `input_ids`/`loss_mask` for `PretokenizedSFTDataset`.
- Added `scripts/qwen35/patch_chat_template_preserve_thinking.py` to patch exported HF checkpoints so vLLM/VERL rollout loads the same chat template used by full-traj SFT pre-tokenization.
- Updated `research/P5_RECIPES.md` with `Recipe 1F/2F/4F`: build capped full-traj smoke parquet, pretokenize with patched template, run 10-step SFT smoke, patch checkpoint tokenizer, then run VLM/vLLM/GRPO smoke.
- Local Windows QC can only cover static/script checks; P5 remains the source of truth for Qwen3.5 tokenizer rendering, CUDA SFT, vLLM serving, and VERL rollout prompt consistency.

### 2026-05-03 Full-traj mask fix after P5 audit

- P5 full-traj pretokenization smoke showed 499/500 train rows and 60/60 test rows were valid with no truncation; max train sequence length was 21,500 under `MAX_LENGTH=32768`.
- The single failed audit row was not bad data. It exposed a mask-boundary bug: using `add_generation_prompt=True` for the prefix was not token-prefix-compatible with rendering the completed assistant message.
- Patched `scripts/tau3/pretokenize_full_traj_sft.py` to build message spans from completed-prefix diffs only: `render(messages[:i], add_generation_prompt=False)` -> `render(messages[:i+1], add_generation_prompt=False)`.
- The pretokenizer now emits `segments` metadata per message: role, segment type, assistant turn index, token start/end, loss flag, and tool names. Training ignores this today, but it creates a clean component index for future memory/retrieval experiments.

### 2026-05-03 RL token-budget audit before length tuning

- Current GRPO smoke risk moved from SFT mechanics to rollout context length. `MAX_PROMPT_LENGTH=4096` filtered all tau3 airline prompts, producing `filter dataset len: 0`; use at least the normal 16K prompt budget for tau3 smoke.
- `run_local_tau3_grpo_live_p5.sh` now adds `+data.apply_chat_template_kwargs.enable_thinking=...`; without the `+`, Hydra rejects the custom Qwen thinking key in struct mode.
- Added `scripts/tau3/analyze_full_traj_token_budget.py` to read pre-tokenized full-trajectory parquet and summarize exact role/segment/assistant-turn token budgets from `segments`.
- The analyzer also optionally decodes assistant spans to estimate how assistant tokens split into `<think>`, `<tool_call>`, visible answer text, and template overhead. Those subcomponent counts are approximate; role/turn/token-boundary counts are exact.
- Use this audit before choosing RL `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH`, `MAX_MODEL_LEN`, and rollout batch size. It should report prompt-length threshold risks for 4K/8K/16K/24K/32K and P50/P90/P95/max context growth by assistant turn.

### 2026-05-03 SDPO smoke passed, paired bundles archived for deep QC

- Kiro ran one-step vanilla SDPO smoke on P5 with commit `4dac3be5` (Codex port) against the same full-traj 10-step SFT checkpoint used by the GRPO smoke.
- Both runs share the same HF checkpoint, same canonical airline JSON dataset, same prompt/response lengths, same wall clock step time (~46.6s), so they are fair ground for engineering QC comparison of GRPO vs SDPO.
- Paired artifacts archived at:
  - `research/diagnostics/grpo_fulltraj_smoke_full.zip` (full log + rollouts, 8 trajectories)
  - `research/diagnostics/sdpo_vlm_sft_smoke.zip` (full log + rollouts, 8 trajectories)
- GRPO step 1 metrics (expected zero actor update, no learning signal from undertrained SFT):
  - `actor/pg_loss=0.0`, `actor/grad_norm=0.0`, `actor/entropy=0.05688`, `actor/ppo_kl=0`
  - `critic/rewards/mean=0.0` — every rollout scored 0.0, so GRPO advantage collapses to 0 (no group variance)
- SDPO step 1 metrics (finite actor update driven by feedback-augmented teacher reprompt):
  - `actor/pg_loss=-0.0315`, `actor/grad_norm=33.81`, `actor/entropy=0.06257`, `actor/ppo_kl=0` (SDPO path fixes ppo_kl=0 for now)
  - `self_distillation/reprompt_sample_fraction=1.0`
  - `self_distillation/feedback_available_fraction=1.0`, `feedback_used_fraction=1.0`
  - `self_distillation/failure_fraction=1.0`, `success_sample_fraction=0.0`
  - `self_distillation/teacher_prompt_token_mean=2111.375`, `teacher_prompt_saturation_fraction=0.0` (no truncation; well under `max_reprompt_len=8192`)
  - `self_distillation/empty_target_batch=0.0`, `teacher_selected_fraction=1.0`, `token_fraction=1.0`
  - `timing_s/sdpo_teacher=0.90s` (extra teacher forward cost on top of the GRPO pipeline)
- Both runs show coherent Qwen3.5 tool-call output (`get_user_details`, `get_reservation_details` with valid JSON args); no gibberish.
- All five stop-if triggers from the Codex handoff were avoided (dataset nonempty, Hydra compose succeeded, decoded output coherent, reprompt fraction nonzero, SDPO loss finite and shape-stable).
- Known open concerns to QC before claim runs (see Codex handoff note; first item resolved by Guardian QC local patch below):
  - Codex SDPO path calls `agg_loss` without `**config.global_batch_info`, so `dp_size` defaults to 1 for SDPO and scales absolute loss magnitude per rank differently from the vanilla path.
  - SDPO path reports `actor/ppo_kl=0` instead of a real KL diagnostic; the stored value comes from a separate `masked_mean(log_prob - old_log_prob)` not surfaced to the main metric key.
  - Reprompt builder assumes `raw_prompt[-1]["content"]` is the initial user task (holds for current tau3 airline data, but should be asserted).
  - `self_distillation_mask` becomes `self_distillation_loss_mask * mask.unsqueeze(1)`, and the empty-target fallback still runs `log_prob.sum() * 0.0`; verify this does not produce spurious autograd warnings in FSDP.

### 2026-05-03 Guardian QC before full SFT/RL

- Added `research/migration/grpo_sdpo_algorithm_fidelity_qc.md` as the written Guardian Gate report for full-traj SFT, GRPO, and SDPO fidelity.
- Full-traj SFT verdict: go. Builder, preserve-thinking template, pre-tokenized `input_ids`/`loss_mask`, and smoke checkpoint path are coherent. Operational requirement: patch exported checkpoint tokenizers with `scripts/qwen35/patch_chat_template_preserve_thinking.py` before vLLM/VERL rollout.
- SDPO verdict: one real multi-GPU scaling bug was found and patched locally in `verl/workers/utils/losses.py`. The SDPO selected-token loss now all-reduces selected-token and selected-sequence counts across `dp_group` before the empty-target branch, then passes global counts plus `dp_size` into `agg_loss`.
- GRPO verdict: current Tau3 config should be named the SDPO-paper companion GRPO baseline, not literal DeepSeekMath-reference GRPO. This follows the SDPO paper's GRPO comparison setup (`rollout.n=8`, rollout IS clip `2`, KL coefficient `0.0`) while adapting the length budget to Tau3 multi-turn airline rollouts.
- Metrics patch: Tau3 reward scoring now logs terminal/nonterminal fraction, budget-exhausted fraction, turn count, tool count, and reward-source fractions during training. PPO data metrics now log success count plus successful-trajectory response tokens, turns, and tool calls.
- Shared full-run defaults: GRPO and SDPO launchers now default to `MAX_PROMPT_LENGTH=16384`, `MAX_RESPONSE_LENGTH=12288`, `MAX_MODEL_LEN=32768`, `ROLLOUT_BATCH_SIZE=8`, and `PPO_MINI_BATCH_SIZE=8` for comparable Tau3 multi-turn baselines. In latest VERL this config value is in prompt units and is multiplied by `rollout.n` internally before actor updates.
- Smoke verdict: GRPO and SDPO wiring both passed. GRPO had zero actor update because all rewards were zero; SDPO produced a finite nonzero feedback-driven update. These smokes prove engineering shape and feedback wiring, not model quality.
- Remaining pre-claim checks: verify terminal official reward extraction in a short real-SFT smoke, assert or harden the `raw_prompt[-1]` user-task assumption, document that current SDPO is sampled-token reverse-KL rather than full top-k logit distillation, and pin identical GRPO/SDPO env snapshots for fair baseline runs.

### 2026-05-03 Real full-trajectory SFT plan (post-Guardian)

After pulling Codex commit `479b41ed`, the remaining work is recipe + launch, not more code. Kiro added `Recipe 5F` to `research/P5_RECIPES.md`:

- 5F-A: build uncapped full-trajectory dataset (drop `--max-rows-per-task`). Expect ~8-9K train rows from the ~9.2K accepted thinking-on trajectories.
- 5F-B: pretokenize at `MAX_LENGTH=32768`, `--truncation error`, `--audit-samples 10`.
- 5F-C: run Recipe 3F token-budget audit on the real dataset.
- 5F-D: 20-step timing probe before committing to a full epoch:
  - < ~120s/step -> proceed with 9K x 1 epoch (~37 hours single pass).
  - 120-180s -> proceed but prepare LIMA-style 5K x 2 epoch fallback.
  - > 180s -> stop; diagnose Liger/bucketing/microbatch=2 or fall back.
- 5F-E: full 1-epoch SFT, export `hf_model`, patch tokenizer with `patch_chat_template_preserve_thinking.py`, record `REAL_SFT_CKPT` path in this file.
- Post-SFT gate: VLM-format config check -> vLLM serve smoke -> one-step GRPO smoke -> one-step SDPO smoke with explicit record of whether `reprompt_sample_fraction` is in `(0.0, 1.0)` (mixed success) or pinned to 1.0 (all-fail branch).
- Then launch Recipe 8 GRPO full baseline and Recipe 10 SDPO full baseline in parallel or back-to-back.
- Shared full-run length defaults (`MAX_PROMPT_LENGTH=16384`, `MAX_RESPONSE_LENGTH=12288`, `MAX_MODEL_LEN=32768`) are upper bounds for RL rollout, not targets. They exist so `response_length/clip_ratio` can come down when trajectories need more room. On OOM, first step down to `MAX_MODEL_LEN=24576`, `MAX_RESPONSE_LENGTH=8192`; only then investigate microbatch or TP.
- Codex is parked until after the first real full-traj SFT checkpoint exists. Deeper fidelity follow-ups deferred until vanilla baselines are in: rename SDPO `actor/ppo_kl` metric, assert `raw_prompt[-1]` user-task invariant, top-k logit SDPO variant.

### 2026-05-04 Real full-traj SFT completed + checkpoint selection

- SFT run `qwen35_4b_vlm_full_traj_sft_real_9k` completed on P5 through 2 epochs, 1,600 steps, early-stopped before epoch 2 finished after val_loss confirmed overfitting.
- 8 checkpoints saved at steps 200, 400, 600, 800, 1000, 1200, 1400, 1600. All tokenizers patched with `scripts/qwen35/patch_chat_template_preserve_thinking.py`.
- val/loss bathtub curve (step / val_loss):
  - 200 / 0.5511
  - 400 / 0.5359
  - 600 / 0.5323 <- valley
  - 800 / 0.5325 <- valley (tied)
  - 1000 / 0.5466 <- epoch-2 rebound, no real recovery
  - 1200 / 0.5470
  - 1400 / 0.5445
  - 1600 / 0.5499 <- highest since step 200, confirms epoch-2 overfit
- Checkpoint selection (Phase A) evaluated steps 400/600/800/1000 on:
  - Easy train-heldout tasks (0, 47, 49) with val_n=4 -> **all 4 ckpts score 100%**. Saturated, cannot discriminate.
  - Mid-hard train tasks (3, 11, 23, 27, 33, 38, 41, 46) with val_n=2 -> pass@1: step 400=0.562, step 600=0.438, step 800=0.438, step 1000=0.438.
- Hard-task ranking is statistically insignificant: SE on pass@1 at n=16 is ~0.12, 95% CI ~±0.24 -> all four ckpts overlap massively.
- Decision (Kiro + Codex unanimous, 2026-05-04): **use step 800** as `REAL_SFT_CKPT` for RL baselines.
  - val_loss tied valley with step 600; step 800 not worse on hard rollouts.
  - Passes vLLM serve smoke; emits clean Qwen XML tool calls when given proper Tau3-style system prompt (the earlier Python-style test 2 was prompt-format induced, not a training bug).
  - Tasks 0/47/49 (easy train-heldout): all SFT ckpts solve 100%, with runtime=official_gym and real multi-turn tool flows (e.g., cancel-policy-refuse + transfer_to_human_agents on task 0). No reward plumbing issue.
- `REAL_SFT_CKPT = checkpoints/SDPO/tau3_verl_sft/TAU3-VERL-SFT-FULL-Qwen-Qwen3.5-4B-qwen35_4b_vlm_full_traj_sft_real_9k/global_step_800/huggingface`.
- Selection artifacts (per-checkpoint results.json for hard + easy task sets) retained under `outputs/eval_paired/ckpt_compare_step{400,600,800,1000}{,_hard}/`.
- Lessons carried forward for later paper write-up:
  - Constant LR=1e-5 likely over-hot past step 600-800; next SFT run should use cosine decay with 3% warmup, min-LR ratio 0.1.
  - SFT val split with only 3 tasks (0, 47, 49) gives a weak-per-task signal. If we rebuild SFT data, consider enlarging val to 5-7 tasks.
  - Bathtub val_loss pattern is canonical for full-param SFT at constant LR; pattern is benign, epoch-2 is optional.
- Next phase (Phase B): vanilla GRPO and vanilla SDPO full baselines from `REAL_SFT_CKPT`, evaluated on canonical test20 with paired grid (3 seeds, val_n=4) for the paper's main comparison table.

### 2026-05-04 Tau3 validation metric aliases

- Added explicit Tau3-facing validation aliases on top of VERL's native `val-core/val-aux` metrics. Raw VERL metrics remain logged for debugging.
- New aliases include `val/pass^1`, `val/pass^2`, ..., up to the configured validation repeat count, plus per-source versions such as `val/tau3_live/pass^4`.
- Important correction: VERL `best@K/mean` is a best-of-K/bootstrap metric, not Tau3 reliability `pass^K`. The new aliases compute `pass^K` exactly per prompt as `C(successes, K) / C(trials, K)` and then average across prompts, matching the paired eval script's convention.
- Added compact health aliases for dashboard use: `val/incorrect_format`, `val/nonterminal_fraction`, `val/budget_exhausted_fraction`, `val/turn_count`, and `val/tool_count` when those fields are present in reward extras.
- Dashboard recommendation: use the new `val/pass^K` aliases for Tau3-facing curves and keep the existing VERL expanded metrics hidden but available for postmortem/debugging.

### 2026-05-05 SDPO vanilla-arm correction

- Rechecked the old SDPO fork after the us-east-1 `sdpo_full_baseline_r16k` collapse/crash. The old repo's `SDPO_ARM=vanilla` launcher explicitly meant successful-peer teacher demonstrations with environment feedback disabled:
  - `use_successful_peer_solution=true`
  - `include_environment_feedback=false`
  - `only_failed_with_feedback=false`
  - `environment_feedback_only_without_solution=false`
  - `dont_reprompt_on_self_success=false`
- The latest-VERL r16k run that collapsed was not that old vanilla arm. It used the port's feedback-only defaults (`use_successful_peer_solution=false`, `include_environment_feedback=true`), so it should be described as **Tau3 feedback-only sampled SDPO**, not original/old-package vanilla SDPO.
- Interpretation update: the r16k collapse remains useful evidence about the feedback-only/full-response-mask variant, but it is not evidence that successful-peer SDPO fails on Tau3.
- Patched `tau3_sdpo_live.yaml` and `run_local_tau3_sdpo_live_p5.sh` to make the arm explicit:
  - default `SDPO_ARM=vanilla_peer` restores successful-peer teacher demonstrations and disables environment feedback.
  - `SDPO_ARM=feedback_only` preserves the previous feedback-only variant for ablation/debugging.
  - launcher experiment names now include the arm name.
  - default `SDPO_MAX_REPROMPT_LEN` is now `12288` and default `SDPO_REPROMPT_TRUNCATION=error` so successful demonstrations do not silently truncate during smoke validation.
- Guardrail for next remote run: run a short `vanilla_peer` smoke first and stop if `self_distillation/reprompt_sample_fraction=0`, `success_sample_fraction=0`, or `empty_target_batch=1.0` persists. The old fork's archived "vanilla SDPO 16-step health check" had peer-teacher flags enabled but zero selected targets, so target activation must be verified before any full baseline claim.

### 2026-05-05 Operational learnings from us-east-1 P5 setup + GRPO eval

**Environment setup (us-east-1 fresh P5):**
- `conda env export --no-builds` does NOT reliably reproduce envs with `--no-deps` installs, prebuilt wheels, or editable installs. Use a setup script (`scripts/setup_env.sh`) instead.
- Missing packages discovered iteratively: `codetiming`, `torchdata`, `accelerate`, `peft`, `qwen-vl-utils`, `toml`, `addict`, `deepdiff`, `tenacity`, `boto3`. All are tau2/verl transitive deps not captured by the YAML export.
- flash-attn prebuilt wheel URL (`lesj0610/flash-attention`) is dead. Copy the compiled package from a working env via S3 tar.
- tau2-bench must be installed from git at commit `220b4784` with `--no-deps`, then set `TAU2_DATA_DIR=~/tau2-bench/data` (data is a repo-level asset, not pip-distributed).
- FlashInfer GDN kernel cache must be copied from a working machine if `nvcc` is not available. Also need `libcudart.so` symlinked to `/opt/conda/lib64/` for the linker step.
- tokenizers version: tau2 reinstall can pull tokenizers 0.23.1 which breaks transformers 5.6.2. Pin `tokenizers==0.22.0 --no-deps`.

**Disk/storage:**
- SM CE us-east-1 space was created with 99 GB EBS (domain max was 100 GB at creation time). Changing domain max after creation does NOT resize existing volumes.
- Moved workspace to `/mnt/sagemaker-nvme/` (28 TB instance NVMe, ephemeral) via symlink: `~/verl_tau3_sdpo → /mnt/sagemaker-nvme/verl_workspace/verl_tau3_sdpo`.
- NVMe is ephemeral — lost on instance stop. Must sync checkpoints to S3 periodically.
- Checkpoint save crash (`basic_ios::clear: iostream error`) was caused by 99 GB EBS being 100% full, not SDPO logic.

**GRPO checkpoint format:**
- GRPO/SDPO training saves FSDP shards (`model_world_size_8_rank_*.pt`) by default. HF export (`model.safetensors`) is only saved if `"hf_model"` is in `CHECKPOINT_SAVE_CONTENTS`.
- The `huggingface/` subdirectory in GRPO checkpoints contains only config/tokenizer (no weights) unless `hf_model` was explicitly requested.
- To convert FSDP shards to HF for eval/serving: `python3 scripts/legacy_model_merger.py merge --backend fsdp --local_dir <actor_dir> --target_dir <actor_dir>/hf_merged`.
- Do NOT manually merge with `torch.save` or `safetensors.save_file` — FSDP shards have invalid storage pointers that crash safetensors.
- For future runs: include `"hf_model"` in save contents for steps we expect to evaluate, or merge selected checkpoints post-hoc.

**Experiment naming / filesystem:**
- Historical issue: the GRPO/SDPO launchers built experiment names from `MODEL_PATH` via `tr '/:' '--'`. With long absolute checkpoint paths, this produced 300+ char W&B/checkpoint directory names that can exceed filesystem limits (255 chars).
- Fixed in `run_local_tau3_grpo_live_p5.sh` and `run_local_tau3_sdpo_live_p5.sh`: both launchers now use a compact model alias in `trainer.experiment_name`.
- Override explicitly with `MODEL_ALIAS=real_sft_step800` (or similar) when launching.
- If `MODEL_ALIAS`/`MODEL_NAME` is not provided and `MODEL_PATH` contains `global_step_<N>`, the launcher auto-aliases it to `ckpt-<N>`.
- Keep `SUFFIX` short and descriptive (`grpo_r16k_ext100`, `sdpo_vpeer_full`, etc.) so W&B and checkpoint paths stay readable.

**Bedrock quotas (cross-region):**
- Sonnet 4.6 quotas: 10,000 RPM, 6,000,000 TPM (account-level, cross-region).
- GRPO training + SDPO training + eval running simultaneously: peak ~200 RPM, ~100K TPM. Zero throttling risk.

**Vanilla peer SDPO activation:**
- Confirmed working on forced-easy tasks (0, 47, 49): step 2 had 12/64 successes → `success_sample_fraction=0.375`, `reprompt_sample_fraction=0.375`, `pg_loss=-0.145`, `grad_norm=36.37`.
- Step 1 had 0/64 successes (all-fail batch) → `empty_target_batch=1.0`. This is expected sparsity, not a bug.
- `teacher_prompt_saturation_fraction=0.0` at `SDPO_MAX_REPROMPT_LEN=12288` — no truncation.
- Vanilla peer SDPO is success-gated: if no rollout in a prompt group succeeds, that group contributes zero SDPO signal. On harder random train tasks, many steps may have zero signal.

**GRPO r16k step-300 canonical test20 eval audit:**
- Artifact `research/diagnostics/eval_grpo_r16k_step300_test20.zip` reported `pass^1=0.342`, `pass^2=0.333`, `pass^3=0.325`, `pass^4=0.317` over 240 rollouts (3 seeds x 4 trials x 20 test tasks). Codex recomputed the pass^K math from raw `rollouts.jsonl` and matched `results.json` exactly.
- The numeric result is valid for that evaluator run, but interpretation is not yet paper-clean because some failures are evaluation artifacts:
  - 55/240 rollouts hit `VLLMValidationError` from context length (`prompt contains at least 32769 input tokens` with `max_model_len=32768`), concentrated in hard tasks such as task 35 (12/12 context failures), task 8 (10/12), and task 24 (10/12).
  - 31 trajectory JSONs show Tau2 `unsupported operand type` tool errors caused by Qwen XML numeric parameters being parsed as strings, e.g. `<parameter=total_baggages>0</parameter>` became `"total_baggages": "0"`.
- Fixed current repo parser in `verl/utils/tau3_action_parser.py`: Qwen XML parameter values now preserve JSON scalar types (`0` -> int, quoted `"12345"` -> str). Added `tests/utils/test_tau3_action_parser.py`.
- Added `scripts/tau3/audit_tau3_eval_artifact.py` to separate context-length failures, unsupported operand tool errors, numeric-string tool args, and clean policy failures in future eval artifacts.
- Updated `scripts/tau3/eval_tau3_paired.sh` to put the active checkout first in `PYTHONPATH`, because the historical Python evaluator may live outside this repo. Kiro should still port/copy the working evaluator into `scripts/tau3/eval_tau3_base_models.py` or explicitly verify it imports this repo's parser before rerunning final numbers.
- Current interpretation: GRPO step 300 clearly improves over SFT and learns consistent simple/policy-boundary behavior, but the exact `34.2%` pass^1 may be a conservative lower bound. Re-run canonical test20 after parser fix before claiming GRPO maximum capacity or deciding whether to extend training.

### 2026-05-05 GRPO step-300 canonical test20 eval v2 (later shown still unfixed)

- Reran eval on us-west-2 P5 after commit `d6c15789`, but later v3 diagnostics showed the old external evaluator still imported `~/SDPO-qwen35/verl/utils/tau3_action_parser.py`. Treat v2 as another **unfixed-parser replicate**, not a true parser-fixed result.
- Same grid: 3 seeds (42/123/456), val_n=4, 20 test tasks, MAX_STEPS=50, temp=0.4, top_p=0.95.
- v2 observed results: pass^1=**0.354**, pass^2=**0.342**, pass^3=**0.338**, pass^4=**0.333**.
- Comparison with v1: pass^1 0.342→0.354 (+0.012), pass^4 0.317→0.333 (+0.016). This is now interpreted as stochastic variation/noise, not confirmed parser-fix gain.
- Notable task changes: task 16 (0→0.08, 1 new success), task 30 (0→0.08), task 48 (0.92→1.00).
- Audit results (v2):
  - successes: 85/240 (was 82/240 in v1)
  - context_length errors: 61/240 (was 55 in v1 — stochastic, same tasks dominate: 18, 35, 8)
  - unsupported_operand_trajectory_rows: 26 (was 31 — reduced but not zero)
  - numeric_string_tool_arg_trajectory_rows: 34 (was 55 — significantly reduced)
  - tool_execution_error_rows: 59
  - json_parse_error_rows: 0
- Residual numeric-string issue: 26 trajectories still show `unsupported operand type` errors. Later v3 diagnostics found the actual cause: the eval subprocess imported the old repo parser, so v2 did **not** prove any tau2-bench internal re-stringification.
- Context-length remains the dominant artifact (61/240 = 25.4%). Tasks 18, 35 hit it on all 12 rollouts. These are genuinely hard multi-turn tasks where the model exhausts 32K context.
- Updated interpretation after v3: GRPO step 300 unfixed-parser replicates are v1=0.342, v2=0.354, v3=0.350. A true corrected baseline requires v4 after copying the fixed parser into the external old repo path.
- Artifact: `research/diagnostics/eval_grpo_r16k_step300_test20_v2.tar.gz` (11MB, 240 trajectories + results.json + eval log).

### 2026-05-05 v3 diagnostic: parser import path root cause

- v3 diagnostic patch (`patch_eval_v3_diagnostics.py`) added a startup sanity check that prints `PARSER_FILE`, asserts int types for numeric XML params, and logs `PRE_ENV_ACTION_FOR_ENV` before `env.step()`.
- **Root cause found**: the eval subprocess imports `verl.utils.tau3_action_parser` from `~/SDPO-qwen35/verl/utils/tau3_action_parser.py` (the OLD repo), NOT from `~/verl_tau3_sdpo/verl/utils/tau3_action_parser.py` (this repo).
- Diagnostic proof: `PARSER_FILE=/home/sagemaker-user/SDPO-qwen35/verl/utils/tau3_action_parser.py`, `PARSER_TYPES={'nonfree_baggages': 'str', 'total_baggages': 'str'}`, assertion failed.
- Why `PYTHONPATH` didn't help: the eval script lives at `~/SDPO-qwen35/scripts/eval_tau3_base_models.py`. When Python resolves `from verl.utils.tau3_action_parser import ...`, it finds the `verl/` package in the old repo's directory tree before checking `PYTHONPATH` entries, because the old repo's directory is implicitly on `sys.path` (script's parent or `.pth` file).
- **Fix**: `eval_tau3_paired.sh` now copies the active parser into `~/SDPO-qwen35/verl/utils/tau3_action_parser.py` before launching eval subprocesses. This is belt-and-suspenders: even if Python resolves `verl` from the old repo, it gets the fixed code.
- **Implication**: v1, v2, and v3 full-run results (pass^1=0.342/0.354/0.350) all used the UNFIXED parser. The true corrected numbers require a v4 run after copying the parser to the old repo.
- v3 full-run numbers (pass^1=0.350) are still valid as a stochastic replicate of the unfixed parser, confirming v1/v2 are in the same noise band.
- Next: run v4 with the parser actually fixed in both repos, then compare to v1-v3 to measure the real parser-fix delta.

### 2026-05-05 GRPO step-300 canonical test20 eval v4 (parser fix CONFIRMED active)

- First truly parser-fixed eval. Diagnostic proof in every GPU subprocess:
  - `PARSER_FILE=/home/sagemaker-user/SDPO-qwen35/verl/utils/tau3_action_parser.py`
  - `PARSER_TYPES={'nonfree_baggages': 'int', 'total_baggages': 'int'}`
  - `PARSER_SANITY_CHECK=PASS`
  - `PRE_ENV_ARG_TYPES={'nonfree_baggages': 'int', 'total_baggages': 'int'}` — ints reach env boundary.
- Same grid: 3 seeds (42/123/456), val_n=4, 20 test tasks, MAX_STEPS=50, temp=0.4, top_p=0.95.
- **v4 results**: pass^1=**0.362**, pass^2=**0.329**, pass^3=**0.315**, pass^4=**0.308**.
- Audit:
  - successes: 87/240 (vs 82-85 unfixed)
  - `unsupported_operand_trajectory_rows`: **0** (was 26-31 in v1-v3)
  - `numeric_string_tool_arg_trajectory_rows`: **0** (was 34-55 in v1-v3)
  - `tool_execution_error_rows`: 45 (was 54-59 — genuine policy failures, not parser artifacts)
  - `context_length` errors: 62/240 (stochastic, same hard tasks)
  - `json_parse_error_rows`: 0
- **Task 22** (baggage update task): 0.00 in all unfixed runs → **0.42** in v4. This is the definitive proof that the parser fix enables real policy successes on numeric-arg tasks.
- Task 8 also gained 1 success (0.00 → 0.08), task 2 dropped slightly (0.92 → 0.67) — stochastic.
- **Comparison summary**:
  | Metric | v1-v3 unfixed (mean±std) | v4 fixed |
  |--------|--------------------------|----------|
  | pass^1 | 0.349 ± 0.006 | **0.362** |
  | unsupported_operand rows | 26-31 | **0** |
  | numeric_string_arg rows | 34-55 | **0** |
  | Task 22 pass^1 | 0.00 | **0.42** |
- **Interpretation**: The parser fix adds ~+1.3pp pass^1 overall, but the real impact is concentrated on tasks requiring numeric tool arguments (task 22: +42pp). The fix eliminates all parser-caused tool execution errors. Remaining 45 tool errors are genuine policy failures.
- **GRPO step 300 corrected baseline**: pass^1=**0.362** on canonical test20 with fixed parser. This is the number to use for SDPO comparison.
- Context-length (62/240 = 25.8%) remains the only evaluation artifact. Tasks 8, 18, 24, 25, 35 are dominated by context exhaustion.
- Artifact: `research/diagnostics/eval_grpo_r16k_step300_test20_v4.tar.gz`.

### 2026-05-05 SDPO vanilla_peer full baseline status (us-east-1)

- Running on us-east-1 P5, at step 50/300 as of this update.
- Step 46 was the last step with nonzero SDPO signal: `success_sample_fraction=0.375`, `pg_loss=0.0011`, `grad_norm=12.03`.
- Steps 47-50: all-fail batches (`success_sample_fraction=0.0`, `empty_target_batch=1.0`, zero gradient). This is expected sparsity for vanilla_peer on random hard tasks — the algorithm is success-gated.
- `response_length/clip_ratio` is 0.89-0.98 (most rollouts hitting 16384 max response). The model is generating very long responses on hard tasks.
- `tau3_live/terminal_fraction` is only 0.01-0.11 — most rollouts are nonterminal (budget exhausted or still running when max response hit).
- Run is healthy (GPUs at 86-100% utilization, ~550s/step). ETA ~70 more hours for remaining 250 steps.
- Key question for analysis: what fraction of the 300 steps will have nonzero SDPO signal? If <10%, the vanilla_peer arm may need easier task sampling or curriculum.

### 2026-05-05 GRPO fixed-parser continuation readiness

- Decision: continue GRPO from the existing step-300 FSDP checkpoint for 200 more steps rather than restarting from SFT. The parser bug affected training rewards only on numeric-argument tool calls, so continuation is the fastest way to test whether the corrected reward interface unlocks additional progress.
- Updated `run_local_tau3_grpo_live_p5.sh` to expose the required continuation knobs as environment variables:
  - `RESUME_MODE` and `RESUME_FROM_PATH` for explicit checkpoint resume.
  - `CHECKPOINT_SAVE_CONTENTS` and `CHECKPOINT_LOAD_CONTENTS` for checkpoint contents.
  - `MAX_ACTOR_CKPT_TO_KEEP` and `MAX_CRITIC_CKPT_TO_KEEP` for optional retention.
- Use `RESUME_MODE=resume_path` plus the exact old `global_step_300` checkpoint path. Do not rely on `resume_mode=auto`, because compact experiment names changed after the original long-name GRPO run and auto-resume may otherwise start a fresh compact-name run from SFT.
- Recommended continuation config:
  - New suffix: `grpo_r16k_fixedparser_cont200`.
  - `TOTAL_TRAINING_STEPS=500`, `TOTAL_EPOCHS=500`.
  - `SAVE_FREQ=50`, `TEST_FREQ=50`, `VAL_N=4`.
  - `CHECKPOINT_SAVE_CONTENTS=["model","optimizer","extra","hf_model"]` so checkpoints at 350/400/450/500 include eval-ready HF weights under `actor/huggingface/`.
- Remote P5 must verify before launch:
  - `bash -n run_local_tau3_grpo_live_p5.sh`.
  - `git rev-parse HEAD` is at or after this readiness update.
  - The resume path contains `actor/model_world_size_8_rank_0.pt` through rank 7 and `data.pt`.
  - A Python parser sanity check returns int types for Qwen XML numeric parameters.
- Suggested stop conditions: resume load fails; first step prints "Training from scratch"; checkpoint save does not create `actor/huggingface/model*.safetensors` or equivalent HF weights after the first save; reward/terminal metrics collapse like the failed SDPO run.

### 2026-05-05 SDPO original-style hybrid arm restored

- Rechecked the upstream SDPO paper/code defaults before the next Tau3 run:
  - Paper/repo framing: SDPO uses feedback-conditioned self-teacher predictions and can also use successful high-reward rollouts as implicit feedback when rich feedback is unavailable.
  - Upstream `actor.self_distillation` defaults include `include_environment_feedback=True`, `environment_feedback_only_without_solution=True`, `dont_reprompt_on_self_success=True`, `remove_thinking_from_demonstration=True`, `distillation_topk=100`, `alpha=0.5`, `is_clip=2`, `max_reprompt_len=10240`, and `reprompt_truncation=right`.
  - Upstream `run_local_sdpo.sh` sets `rollout.n=8`, `policy_loss.loss_mode=sdpo`, `dont_reprompt_on_self_success=True`, and `alpha=0.5`; rich-feedback experiments use the feedback-enabled default path.
- Corrected this latest-VERL Tau3 launcher/config so `SDPO_ARM=original` is now the default. `SDPO_ARM=vanilla` is accepted as an alias for this original-style hybrid arm. It means:
  - `include_environment_feedback=true`
  - `use_successful_peer_solution=true`
  - `only_failed_with_feedback=true`
  - `dont_reprompt_on_self_success=true`
  - `environment_feedback_only_without_solution=true`
  - `serialize_nonstring_feedback=true`
  - `reprompt_truncation=right`
- Interpretation update:
  - `SDPO_ARM=vanilla_peer` is now explicitly a diagnostic peer-only arm, not the main roadmap baseline and not the memory-SDPO proposal. It is success-gated and can produce empty-target batches on all-fail rollout groups.
  - `SDPO_ARM=feedback_only` remains a diagnostic ablation.
  - The overnight baseline should use `SDPO_ARM=original` and a short suffix such as `sdpo_original_full`.
- Fidelity caveat:
  - This is original-style SDPO teacher-context routing in the latest-VERL Tau3 port, not a byte-for-byte upstream SDPO reproduction.
  - Current local loss remains sampled-token reverse-KL using stored `teacher_logprobs` (`SDPO_ALPHA=1.0` required by the port). It does not yet implement upstream full-logit/top-k/JSD/EMA teacher regularization. If the paper needs literal upstream SDPO, port those pieces as a separate fidelity task.
- Overnight health gates:
  - Expect `self_distillation/feedback_available_fraction > 0` and `feedback_used_fraction > 0` on failed samples.
  - `empty_target_batch` should not persist once failures have feedback, even if `success_sample_fraction=0`.
  - Stop or inspect if `response_length/clip_ratio` stays near 1.0 with `tau3_live/terminal_fraction` collapsing, or if `reprompt_sample_fraction=0` for several consecutive steps.

### 2026-05-06 latest-VERL vLLM V1 / vLLM20 migration path

- Decision: move the Tau3 execution path forward in the latest-VERL fork instead of spending more time repairing the old paper-original SDPO conda environment.
- Added a reversible Tau3 vLLM profile in both GRPO and SDPO launchers:
  - Default `TAU3_VLLM_PROFILE=qwen35_v1`.
  - Exports `VLLM_USE_V1=1` and `VLLM_ALLREDUCE_USE_SYMM_MEM=0`.
  - Sets Qwen3.5-safe rollout defaults from the repo's own Qwen3.5 FSDP examples: `enable_prefix_caching=false`, `enable_chunked_prefill=true`, `max_num_batched_tokens=8192`, `enforce_eager=false`, `language_model_only=true`.
  - `TAU3_VLLM_PROFILE=legacy` now fails closed because the current async vLLM server path imports vLLM V1 directly. Roll back with the previous known-good env/image, not this profile.
  - Explicitly propagates `VLLM_USE_V1`, `VLLM_ALLREDUCE_USE_SYMM_MEM`, and `VLLM_LANGUAGE_MODEL_ONLY` into vLLM Ray server actors.
- Updated Tau3 launch defaults back to the r16k comparison setup:
  - `data.max_response_length=16384` by default for GRPO and SDPO.
  - `tau3.sdpo.max_reprompt_len=16384` by default for the SDPO launcher.
- Added vLLM V1-specific runtime overrides for Kiro/P5 debugging without code edits:
  - `VLLM_BLOCK_SIZE`
  - `VLLM_MAMBA_BLOCK_SIZE`
  - `VLLM_MAMBA_CACHE_MODE`
  - `VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER`
  - `VLLM_KV_CACHE_MEMORY_BYTES`
  - `VLLM_DISABLE_CASCADE_ATTN`
  - `VLLM_COMPILATION_CONFIG_JSON`
- Added `scripts/p5_preflight_vllm_v1.py` to fail before training if:
  - `VLLM_USE_V1` is not active.
  - vLLM is older than `0.20.0` unless explicitly allowed.
  - `vllm._C` cannot import, which catches the torch/vLLM ABI mismatch seen on east P5.
  - `cuda_runtime.h` is missing, which catches the FlashInfer/GDN JIT failure seen on east P5.
  - Tau3 parser no longer preserves numeric XML parameters as ints.
- Added `scripts/p5_setup_vllm_v1_env.sh` as a best-effort conda setup helper using `uv pip install vllm==$VLLM_VERSION --torch-backend $TORCH_BACKEND`. Prefer the official image when available; this script is for temporary P5 bring-up.
- Remaining risk:
  - vLLM V1 hybrid KV support is exactly the moving part for Qwen3.5/GDN. The first P5 smoke must be 2-3 steps only, and if the page-size error persists, try `VLLM_BLOCK_SIZE`/`VLLM_MAMBA_BLOCK_SIZE` and `VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER` before full overnight training.

### 2026-05-06 ECR image package for vLLM V1

- Added a containerized path so future P5 runs do not depend on hand-upgraded conda environments:
  - `.dockerignore`
  - `docker/Dockerfile.tau3.vllm20.v1`
  - `scripts/p5_build_push_ecr_vllm_v1.sh`
  - `scripts/p5_run_image_smoke_vllm_v1.sh`
  - `research/vllm_v1_image_plan.md`
- Image strategy:
  - Base image: `vllm/vllm-openai:v0.20.1-cu129` after the 2026-05-06 latest-vLLM QC update.
  - The "openai" label means OpenAI-compatible HTTP API shape, not OpenAI API usage. We use it for the pinned vLLM/PyTorch/CUDA stack.
  - Layer latest-VERL Tau3 code and `tau2-bench` commit `220b47844fb74d4351037e81055cf1e2948e4734`.
  - Keep datasets, checkpoints, W&B, and outputs mounted from the P5 host rather than baked into the image.
- ECR flow:
  - GitHub is only for Codex/Kiro synchronization. P5 has no Git requirement; Kiro should sync the repo snapshot to S3, P5 should pull from S3, and the P5 build should pass `SOURCE_REVISION=<github_commit_short_sha>`.
  - Build/push on P5 with `AWS_REGION`, `ECR_REPOSITORY`, `SOURCE_REVISION`, and `IMAGE_TAG`.
  - Script creates the ECR repo if missing, logs in, builds, runs baked-image preflight, tags, pushes, and prints final `image_uri` plus digest.
  - Do not use the corporate laptop to validate the Docker image. The laptop can do lightweight checks, but the authoritative build/smoke must be on P5 because that is the target GPU/CUDA/vLLM surface.
- Smoke flow:
  - `SMOKE_MODE=image_preflight` runs the vLLM V1 preflight against the baked image without mounting the host repo.
  - `SMOKE_MODE=preflight` runs the vLLM V1 preflight against the mounted P5 repo/checkpoint/dataset.
  - `SMOKE_MODE=grpo` runs a tiny 2-3 step GRPO smoke using the mounted SFT checkpoint and Tau3 dataset.
  - `SMOKE_MODE=sdpo` is available but should wait until GRPO proves engine startup.
- Stop before overnight training if any of these appear:
  - `vllm._C` ABI/import failure.
  - missing `cuda_runtime.h`.
  - vLLM V1 hybrid KV page-size initialization failure.
  - Tau3 parser numeric type regression.
  - Bedrock credentials/model access failure during live user simulation.

### 2026-05-06 environment majority-vote update

- Six-way QC aligned on the environment path:
  - The corporate Windows RTX 3080 laptop can validate repo scripts, probes, and basic `nvidia-smi`, but it cannot prove P5/H100 vLLM, NCCL, FlashInfer, Ray, or 32K-context behavior without WSL/Docker/Linux GPU runtime.
  - The current SageMaker Code Editor P5 space cannot run Docker-in-Docker. Docker/ECR remains reproducibility infrastructure, but image build must happen outside this SM CE space or on a Docker-enabled SageMaker domain.
  - For runnable paper-style SDPO today, prefer latest-VERL `SDPO_ARM=original` over old-repo `SDPO-paper-original`, unless the goal is specifically a historical-code ablation. Use `SDPO_ARM=original` or `SDPO_ARM=paper`; the launcher does not accept literal `paper_original`.
- Added reproducibility helpers:
  - `scripts/probe_runtime_surface.py` emits a read-only JSON runtime surface probe for local/P5.
  - `scripts/p5_export_frozen_env.sh` exports a no-Docker frozen env manifest from the active P5 conda env.
- Hardened vLLM V1 smoke mechanics:
  - `build_cli_args_from_config` now normalizes snake_case config keys to kebab-case CLI flags and emits explicit `--no-*` flags for selected vLLM BooleanOptionalAction options such as `enable_prefix_caching=false`.
  - `scripts/p5_run_image_smoke_vllm_v1.sh` now forwards optional hybrid-KV/mamba/cache and training-size env knobs into the Docker smoke container.
- P5 smoke order for vLLM 0.20.x/V1:
  1. Current baseline with `TAU3_VLLM_PROFILE=qwen35_v1`.
  2. If page-size error persists, try `VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER=true`.
  3. If needed, try aligned cache blocks with prefix caching enabled: `VLLM_ENABLE_PREFIX_CACHING=true VLLM_BLOCK_SIZE=16 VLLM_MAMBA_BLOCK_SIZE=16 VLLM_MAMBA_CACHE_MODE=align`, then a block-8 variant.
  4. Only use `VLLM_KV_CACHE_MEMORY_BYTES` after logs show capacity/profiling issues rather than page-size unification.

### 2026-05-06 latest vLLM V1 environment QC

- Stage-1 QC result:
  - Current repo is suitable for a Kiro/P5 environment bring-up once the vLLM V1 capacity-matrix changes are pushed.
  - vLLM latest PyPI/Docker tag checked on 2026-05-06 is `0.20.1`; defaults now use `vllm/vllm-openai:v0.20.1-cu129` and `VLLM_VERSION=0.20.1`.
  - `0.20.0` remains the explicit fallback tag if Qwen3.5 hybrid KV support regresses on `0.20.1`.
- No-Docker SM Code Editor path:
  - Use `scripts/p5_setup_vllm_v1_env.sh` to create `sdpo-vllm20-v1`.
  - Use `scripts/p5_run_vllm_v1_capacity_matrix.sh` to run the ordered capacity ladder without Docker.
- Capacity ladder:
  1. `auto_no_prefix_24k_48k`
  2. `auto_prefix_24k_48k`
  3. `fp8_no_prefix_24k_48k`
  4. `fp8_prefix_24k_48k`
  5. `fp8_prefix_32k_64k`
- All current launchers and the official-gym runtime fallback default `TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION=0`, preserving historical actor `<think>` in the actual chat history while avoiding duplicated full-transcript observations from Tau3 Gym.
- Stage-2 mitigation work should start with diagnostics-only changes unless explicitly requested otherwise:
  - component token/repetition diagnostics,
  - rollout dump enrichment.
  - Defer length-clipped sample masking until diagnostics confirm it is needed, because it changes the optimization objective.

### 2026-05-06 vLLM V1 no-Docker conda runbook branch

- Created branch `codex/vllm-v1-env-bringup` for the P5 no-Docker environment handoff.
- Added `research/p5_vllm_v1_conda_runbook.md` as the one-page execution path for SageMaker Code Editor P5:
  - S3 snapshot flow for P5 hosts without Git.
  - `sdpo-vllm20-v1` conda env build from `scripts/p5_setup_vllm_v1_env.sh`.
  - Required `tau2-bench` snapshot/install at commit `220b47844fb74d4351037e81055cf1e2948e4734`.
  - Preflight, GRPO capacity smoke, original-SDPO smoke, hybrid-KV bounded fallbacks, frozen-env export, and stop conditions.
- Hardened `scripts/p5_setup_vllm_v1_env.sh` so the conda path now installs Tau3 live runtime deps and installs `tau2-bench` from an existing S3-synced directory or Git when available.
- Important gating rule for Kiro/P5: `scripts/p5_preflight_vllm_v1.py` is necessary but not sufficient. A 2-3 step GRPO capacity smoke is the first true proof that the vLLM V1 engine works for Qwen3.5 on P5.

### 2026-05-07 vLLM V1 setup QA after west-P5 success

- Kiro updated `research/p5_environment_issues.md` with the working west-P5 stack: `sdpo-vllm20-v1`, Python 3.12.13, torch 2.11.0+cu129, vLLM 0.20.1 cu129, flash-attn 2.8.3 community wheel, FlashInfer 0.6.8.post1, CUDA/NVVM tools 12.9.86.
- Codex QA aligned the runnable helpers with that manual flow:
  - `scripts/p5_setup_vllm_v1_env.sh` now defaults to the vLLM cu129 wheel index fallback, installs CUDA `cuda-nvcc-tools` + `cuda-nvvm-tools` + `cuda-cudart-dev`, installs the flash-attn torch2.11/cp312 community wheel, and prints the required CUDA/GDN env exports.
  - `scripts/p5_run_vllm_v1_capacity_matrix.sh` now defaults `TRAIN_BATCH_SIZE=8`, `ROLLOUT_BATCH_SIZE=8`, `PPO_MINI_BATCH_SIZE=8` for 8-GPU P5 divisibility.
  - `research/p5_vllm_v1_conda_runbook.md` now requires flash-attn import plus `nvcc`/`cicc` visibility before full runs.
- QA stance: a GDN JIT failure can be tolerated only for a tiny smoke if the profile completes. Do not start overnight GRPO/SDPO until GDN is fixed or the passing profile is explicitly documented as stable enough despite the warning.

### 2026-05-07 FP8 KV cache follow-up

- West-P5 capacity matrix showed:
  - `01_auto_no_prefix_24k_48k`: PASS.
  - `02_auto_prefix_24k_48k`: PASS.
  - FP8 with `VLLM_CALCULATE_KV_SCALES=true`: FAIL with `AttributeError: 'list' object has no attribute 'zero_'` inside vLLM `init_fp8_kv_scales`.
- Interpretation: the failure is likely dynamic FP8 KV scale initialization on Qwen3.5's hybrid attention/linear-attention cache, not necessarily FP8 KV storage itself.
- Updated `scripts/p5_run_vllm_v1_capacity_matrix.sh` so default FP8 profiles use `VLLM_CALCULATE_KV_SCALES=false`. Dynamic-scale FP8 profiles are opt-in with `RUN_EXPERIMENTAL_FP8_SCALES=1`.
- Main overnight baseline recommendation remains auto KV + prefix caching unless no-scale FP8 passes and a short fixed-task comparison shows no quality regression.

### 2026-05-07 ten-step GRPO readiness smoke

- Added `CAPACITY_PROFILES` to `scripts/p5_run_vllm_v1_capacity_matrix.sh` so Kiro/P5 can run only the known-good profile instead of the full capacity matrix.
- Final readiness profile for clean GRPO rerun: `CAPACITY_PROFILES="02_auto_prefix_24k_48k"`, `VLLM_KV_CACHE_DTYPE=auto`, `VLLM_ENABLE_PREFIX_CACHING=true`, `TAU3_LIVE_ALL_MESSAGES_AS_OBSERVATION=0`, `MAX_RESPONSE_LENGTH=24576`, `MAX_MODEL_LEN=49152`, and 8/8/8 batch sizes.
- `research/P5_RECIPES.md` now contains `Recipe 8A: Ten-Step vLLM V1 GRPO Readiness Smoke`; use it as the last gate before increasing `TOTAL_TRAINING_STEPS` for the full rerun.
- Added `scripts/tau3/bundle_grpo_readiness_from_p5.sh` and `scripts/tau3/analyze_grpo_smoke_bundle.py` so the passing 10-step smoke can be packaged from P5 with logs, rollout JSONLs, W&B metadata, runtime snapshot, and transcript/repetition heuristics. The bundler now prefers `BUNDLE_NAME`; `SMOKE_NAME` remains only as a legacy alias.
- `scripts/p5_run_vllm_v1_capacity_matrix.sh` now supports `RUN_NAME_PREFIX` and `ROLLOUT_OUTPUT_ROOT`; use these for 300-step runs so full-run W&B names and rollout JSONLs do not mix with the 10-step capacity-smoke artifacts.

### 2026-05-07 east-P5 vLLM V1 bootstrap

- Added an `East P5 Fresh Bootstrap` section to `research/p5_vllm_v1_conda_runbook.md`.
- Recommended layout for the new us-east-1 P5:
  - Code snapshot: `~/verl_tau3_sdpo_vllm20`.
  - Large artifacts/checkpoints: `~/verl_tau3_sdpo/checkpoints`.
  - Tau2/Tau3 runtime dependency: `~/tau2-bench`.
- Keep S3 sync region as `us-west-2` for the existing `s3://tianyd-rlvr-research/tau3-sdpo/latest-verl` bucket, but set `AWS_REGION=us-east-1` / `AWS_DEFAULT_REGION=us-east-1` for Bedrock/Tau3 user-simulator calls on east P5.
- East readiness gates: vLLM V1 preflight, `vllm._C` import, `flash_attn` import, `nvcc`/`cicc` visibility, 3-step GRPO profile `02_auto_prefix_24k_48k`, then 1-step `MODE=sdpo SDPO_ARM=original` smoke before any overnight vanilla-SDPO baseline.
- East preflight found `cuda_runtime.h` only under Python `site-packages/nvidia/cuda_runtime/include`, not under `$CONDA_PREFIX/targets/x86_64-linux/include`. Root cause is missing CUDA runtime header package; `cuda-cudart` alone did not install headers. Install `cuda-cudart-dev=12.9.79` from `nvidia/label/cuda-12.9.1`, keep `CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"`, and rerun preflight before GRPO smoke.
- East GDN JIT then progressed one include deeper and failed on `crt/host_config.h`. This is the same FlashInfer/GDN JIT setup family, but a different missing CUDA component: install `cuda-crt=12.9.86`, verify `$CUDA_HOME/include/crt/host_config.h`, clear the failed `~/.cache/flashinfer/0.6.8.post1/90a/cached_ops/gdn_prefill_sm90` cache, then rerun the 3-step GRPO smoke.
- East GDN JIT then progressed again and failed on `fatbinary_section.h`. This is still the same CUDA compiler-dev surface issue: install `cuda-nvcc-dev_linux-64=12.9.86`, `cuda-nvvm-dev_linux-64=12.9.86`, and `cuda-crt-dev_linux-64=12.9.86`; verify `$CUDA_HOME/include/fatbinary_section.h`; clear the FlashInfer GDN cache; rerun the 3-step GRPO smoke. This fix was confirmed on east P5: after the dev headers were installed and the cache was cleared, the run started without the GDN missing-header error. If another CUDA-internal header is missing on a future host, stop piecemeal fixes and install the broader `cuda-compiler=12.9.1` meta-package from `nvidia/label/cuda-12.9.1`.
- East P5 has only a 99 GB EBS root but large ephemeral NVMe. For east smoke/full SDPO, set `NVME_ROOT=/mnt/sagemaker-nvme/tau3_sdpo` and export `TMPDIR`, `RAY_TMPDIR`, `SDPO_OUTPUT_ROOT`, `SDPO_CHECKPOINT_ROOT`, `LOG_ROOT`, `ROLLOUT_OUTPUT_ROOT`, `WANDB_DIR`, and cache dirs under NVMe. `ROLLOUT_OUTPUT_ROOT` alone is not enough because checkpoints still follow `SDPO_CHECKPOINT_ROOT`.
- Added `scripts/p5_run_east_original_sdpo_full.sh` for the overnight east-P5 original-SDPO baseline. It assumes `sdpo-vllm20-v1` is active, sets `MODE=sdpo` and `SDPO_ARM=original`, uses profile `02_auto_prefix_24k_48k`, defaults to 300 steps with save/test every 30, keeps at most three actor checkpoints, and routes Ray temp/checkpoints/rollouts/W&B/cache/logs to `/mnt/sagemaker-nvme/tau3_sdpo`.
- West clean-GRPO v2 run `ghyw4x9s` crashed at step 78 with clean rollout metrics (`prompt_clip=0`, `response_clip=0`, `budget_exhausted=0`, `env_error=0`, `bedrock_error=0`). Local disk check showed `/` container overlay at 100% despite large `/home` and `/mnt/sagemaker-nvme`. Added `scripts/p5_resume_west_grpo_vllm_v1_clean_from_step60.sh` to resume from the saved `global_step_60` checkpoint while forcing Ray temp, Python temp, W&B, HF/Torch/Triton/XDG caches, logs, rollouts, and new checkpoints to NVMe. Also hardened `scripts/p5_run_vllm_v1_capacity_matrix.sh` to default P5 heavy paths to `NVME_ROOT`.

### 2026-05-07 Tau3 Bedrock env-error guardrails

- GRPO vLLM V1 clean rerun showed Bedrock/LiteLLM `ServiceUnavailableError` bursts around steps 106-135. These are user-simulator infrastructure failures, not actor-policy failures, and should not be treated as ordinary reward-0 Tau3 workflow evidence.
- Added first-class Tau3 env-error metadata: `env_error`, `bedrock_error`, `env_error_type`, `env_error_message`, `bedrock_retry_count`, and `bedrock_fallback_model` in the official-gym live result.
- `compute_score` now emits `tau3_live/env_error_fraction`, `tau3_live/bedrock_error_fraction`, `tau3_live/bedrock_retry_count`, `tau3_live/bedrock_fallback_fraction`, and `reward_source=official_gym_env_error`; env-error feedback is suppressed so SDPO does not build teacher prompts from infrastructure failures.
- Trainer guardrail: env-error samples have `response_mask` and reward zeroed and are moved into singleton GRPO UID groups so they cannot affect actor loss or sibling group baselines. If `TAU3_ENV_ERROR_SKIP_UPDATE_THRESHOLD` is reached, actor update is skipped for that batch; critic update is also skipped if enabled.
- SDPO guardrail: env-error rows are excluded from failed-sample teacher construction and logged as `self_distillation/env_error_excluded_fraction`.
- Launch defaults: `TAU3_LIVE_USER_ARGS_JSON='{"num_retries":8,"timeout":120}'`, `TAU3_MASK_ENV_ERROR_ROLLOUTS=1`, `TAU3_ENV_ERROR_SKIP_UPDATE_THRESHOLD=0.25`, and `TAU3_RETRY_STEP_ON_TRANSIENT=0`.

## 2026-05-07 Memory-SDPO First Slice

- Added a teacher-side Memory-SDPO hook for Tau3 SDPO. Actor/student rollout context is unchanged; memory is inserted only into the SDPO teacher reprompt before teacher logprob scoring.
- New utility: `verl/utils/tau3_sdpo_memory.py` loads compact JSONL/JSON cards, strips raw `<think>`, and retrieves either `relevant` or `random` cards.
- Seed card bank: `research/memory_cards/tau3_airline_seed_cards.jsonl`.
- Offline prompt builder: `scripts/tau3/build_sdpo_memory_teacher_probe.py`.
- Retrieval audit scaffold: `scripts/tau3/audit_sdpo_memory_retrieval.py` compares `random`, `lexical`, optional `dense_hf`, and optional `bedrock_titan` retrieval over seed cards and/or successful rollout memories. It supports `full_trajectory`, `event_chunk`, and `char_chunk` memory units, emits top-k retrieval JSONL plus timing summary JSON, and is offline-only for now.
- Live toggles:
  - `SDPO_MEMORY_ENABLED=true`
  - `SDPO_MEMORY_PATH=research/memory_cards/tau3_airline_seed_cards.jsonl`
  - `SDPO_MEMORY_MODE=relevant|random`
  - `SDPO_MEMORY_INJECT_WHEN=no_solution`
  - `SDPO_MEMORY_ALLOW_WITHOUT_FEEDBACK=false`
- New W&B metrics include `self_distillation/memory_used_fraction`, `memory_random_used_fraction`, `memory_no_solution_used_fraction`, and `memory_section_char_mean`.
- Recommended next step before a full Memory-SDPO run: build T0/T1/T2 teacher probe prompts from failed rollout JSONLs, run a separate teacher-output scoring/manual rubric pass, and verify relevant memory beats random memory on teacher next-action quality.
- Recommended retrieval order: first audit seed cards plus successful raw trajectories/chunks with `random` and `lexical`; then add `dense_hf` CPU or `bedrock_titan` with an embedding cache. Do not wire dense retrieval into live SDPO until latency and retrieval relevance are measured.
- Retrieval audit has a dedicated CPU test file: `tests/utils/test_tau3_sdpo_memory_retrieval_audit.py`.
- Rationale for `TAU3_RETRY_STEP_ON_TRANSIENT=0`: replaying outer `env.step(action)` can duplicate transactional writes if the tool state changed before the user-simulator call failed. Prefer retry inside Tau2/LiteLLM user-call handling; the outer layer should mark/mask env errors rather than replay actions.

## 2026-05-07 Faithful Tau3 SDPO Guarded EMA/Top-K Path

- Added explicit Tau3 SDPO teacher backend routing. Default `SDPO_TEACHER_BACKEND=ema_ref` builds an ActorRolloutRef worker, computes teacher logprobs with the colocated ref/self-teacher, updates the actor, then applies `teacher <- (1 - rate) * teacher + rate * actor` with `SDPO_TEACHER_UPDATE_RATE=0.05`.
- Kept `SDPO_TEACHER_BACKEND=actor_snapshot` as the cheaper latest-VERL approximation. Run names now include the teacher backend so EMA and actor-snapshot SDPO are not conflated.
- Ported the original SDPO full-logit/top-k loss shape into latest VERL for Tau3. Default launcher/config now use `sdpo_full_logit_distillation=true`, `sdpo_alpha=0.5`, `sdpo_distillation_topk=100`, and `sdpo_distillation_add_tail=true`, matching the paper/code defaults rather than the earlier sampled-token `alpha=1.0` approximation.
- Latest-VERL implementation detail: because the current model-engine actor update does not expose the colocated EMA teacher inside the loss function the way the old SDPO `dp_actor.py` did, the trainer precomputes actor/student top-k support immediately before the optimizer step, gathers EMA-teacher probabilities on that same support under the reprompted teacher context, then backprops the SDPO JSD/top-k/tail loss during actor update. With the default one-minibatch/one-epoch Tau3 launch this matches the official support choice before parameters move. The trainer now fails fast if `ppo_epochs>1` or multiple actor minibatches are requested, because that would make the precomputed top-k support stale relative to later optimizer steps.
- Added target-side SDPO guardrails via `tau3.sdpo.target_guard`: nonterminal/budget-exhausted, response saturation, parser-format artifacts (`incorrect_format`), open `<think>`, repetition, and tool-loop rows can be zeroed or downweighted through `corrupted_row_weight`.
- Successful peer demonstrations are also filtered through the same target-validity guard before being eligible for teacher reprompt context, and `<think>` stripping now removes both closed and unterminated/case-variant think spans. This prevents reward-successful but corrupted trajectories from becoming privileged teacher demonstrations.
- The guardrail preserves original SDPO row routing (`self_distillation_mask`) but changes the token-level SDPO target mask. The actor loss now consumes an explicit `self_distillation_target_token_mask`.
- EMA-teacher SDPO checkpoints now save a colocated `actor/sdpo_ema_teacher/` model checkpoint and fail fast on resume if an EMA SDPO checkpoint lacks that teacher state. This avoids silently resetting `teacher <- EMA(student)` to the initial SFT/base model after a crash.
- P5 guarded-SDPO smoke reached the checkpoint path after step 1, then exposed a ref/EMA-teacher save bug: the forward-only ref teacher has no optimizer, but its instantiated checkpoint manager still tried to save `optimizer`. Fixed in commit `3f099671` by forcing the EMA ref checkpoint manager to model-only save/load contents before `actor/sdpo_ema_teacher` save/load. Kiro should retest from a fresh run stem after syncing this commit.
- EMA teacher updates tolerate actor/ref parameters living on different devices after engine offload transitions by copying the actor shard to the teacher parameter device/dtype for the in-place EMA operation.
- EMA worker update metrics are surfaced in trainer logs, including `ema_teacher_device_transfer_tensors_{mean,max}`, so smoke runs can confirm whether offload-induced device transfers occurred.
- Ref/EMA teacher micro-batch config is now defined in `ref.yaml` with `log_prob_micro_batch_size_per_gpu` mirroring `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` for the ref forward path, plus `ppo_micro_batch_size_per_gpu=1` as a config-compat fallback. The worker preserves whichever non-null value reaches the ref config so `ActorConfig` construction never loses the required micro-batch field.
- SDPO loss metrics now emit stable top-k mass keys even when a rank has an empty target batch or when sampled-token SDPO is used without full-logit tensors. This avoids DP metric aggregation failures from per-rank branch differences.
- The no-padding converter now distinguishes full-sequence top-k teacher tensors from response-shaped sampled-token teacher logprobs. Full-sequence top-k tensors are unpadded to nested values; response-shaped sampled-token logprobs remain padded for the sampled-token SDPO loss path.
- New/important metrics: `self_distillation/teacher_backend_ema_ref`, `self_distillation/teacher_backend_actor_snapshot`, `self_distillation/full_logit_distillation`, `self_distillation/distillation_topk`, `self_distillation/student_topk_mass`, `self_distillation/teacher_topk_mass`, `self_distillation/ema_teacher_updated`, `self_distillation/target_guard_selected_fraction`, `self_distillation/target_guard_token_keep_fraction`, and per-reason target-guard fractions.
- Claim boundary: this is the faithful original-SDPO baseline for Tau3 under the default launch shape, plus Tau3-specific target guardrails. For a no-guard ablation, set `tau3.sdpo.target_guard.enabled=false`; for the old approximation, set `SDPO_FULL_LOGIT_DISTILLATION=false SDPO_ALPHA=1.0`.

### 2026-05-09 SDPO full-logit memory bottleneck and true-8K run correction

- Added `research/sdpo_full_logit_memory_bottleneck.md` to document the current systems bottleneck: faithful SDPO full-logit/top-k/JSD uses transient `[tokens, vocab]` logits tensors for student top-k, EMA teacher gather, and actor-update loss, so Tau3 multi-turn response spikes can OOM even with a 4B model on 80GB GPUs.
- Pulled W&B run `rag0ih8i` from project `SDPO-vllm-v1-original-sdpo-safe`. The run name said `safe_8kreprompt4k`, but actual W&B config was `data.max_prompt_length=8192`, `data.max_response_length=24576`, `max_model_len=49152`, and `tau3.sdpo.max_reprompt_len=4096`. The 8K response cap did **not** apply.
- Root cause: `scripts/p5_run_east_original_sdpo_full.sh` invokes `scripts/p5_run_vllm_v1_capacity_matrix.sh`, and selected profile `02_auto_prefix_24k_48k` hardcodes `MAX_RESPONSE_LENGTH=24576` and `MAX_MODEL_LEN=49152`, overriding the intended safe response cap.
- The same W&B run logged steps 1-7 only. `output.log` contains step-8 pre-logprob diagnostics, then no completed `step:8` line: `actor_student attention_mask_max=28749`, `actor_student response_mask_max=23679`, `ema_ref_teacher attention_mask_max=27058`, `ema_ref_teacher response_mask_max=23679`. This strongly supports a step-8 OOM during or soon after SDPO top-k logprob, caused by a near-24K response spike.
- Completed-step CUDA metrics show actor-student top-k was the largest SDPO logprob phase: step 4 reached `student_peak=31.01 GiB`, `teacher_peak=27.23 GiB` with `response_length/max=5296`. Step 8's `response_mask_max=23679` is far larger and was not W&B-committed as a completed training row.
- Correct next safe test: bypass `p5_run_vllm_v1_capacity_matrix.sh` or add an env-driven custom profile. Use `run_local_tau3_sdpo_live_p5.sh` directly with `MAX_PROMPT_LENGTH=8192`, `MAX_RESPONSE_LENGTH=8192`, `SDPO_MAX_REPROMPT_LEN=4096`, `MAX_MODEL_LEN=16384`, logprob microbatch size 1, diagnostics enabled, and NVME paths for logs/checkpoints/Ray/W&B.

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
- Does the sampled-token vanilla SDPO latest-VERL path deliver learning gains versus the SDPO-paper companion GRPO baseline after full-traj SFT?
- If the paper later needs a literal DeepSeekMath-reference GRPO ablation, add it as a separate explicitly named ablation rather than changing the main SDPO companion baseline.

### 2026-05-08 GRPO vLLM-V1 clean-300-v2 step 270 canonical test20 eval

- Eval completed on west P5 with merged FSDP→HF checkpoint.
- Root cause of initial smoke failure: `huggingface/` directory only had config/tokenizer, no weight files. GRPO training saves FSDP shards by default; standalone vLLM `LLM()` requires HF-format weights. Fixed by running `python3 scripts/legacy_model_merger.py merge --backend fsdp --local_dir <actor_dir> --target_dir <actor_dir>/huggingface`.
- Same grid as prior evals: 3 seeds (42/123/456), val_n=4, 20 canonical test tasks, MAX_MODEL_LEN=49152, vLLM V1, 8 GPU parallel.
- Parser: fixed and confirmed (`PARSER_SANITY_CHECK=PASS`, int types for numeric XML params).
- **Results**: pass^1=**0.512**, pass^2=**0.437**, pass^3=**0.405**, pass^4=**0.389**.
- Comparison with GRPO step 300 (old r16k run, v4 fixed-parser eval):
  - pass^1: 0.362 → **0.512** (+15pp)
  - pass^4: 0.308 → **0.389** (+8pp)
- Per-task highlights:
  - Perfect (1.00): tasks 2, 6, 13, 26, 31, 45, 48 (7 tasks)
  - Strong (≥0.50): tasks 8 (0.42), 19 (0.42), 22 (0.50), 35 (0.92), 37 (0.58)
  - Zero: tasks 16, 18, 24, 29, 32, 44 (6 tasks — hard/context-limited)
- Interpretation: step 270 from the clean vLLM-V1 GRPO run is substantially stronger than step 300 from the old r16k run. The improvement is likely due to: (a) cleaner training without parser-induced reward noise, (b) vLLM V1 prefix caching enabling longer coherent rollouts, (c) step 270 being pre-overfit relative to step 300.
- This is now the **GRPO baseline** for SDPO comparison: pass^1=0.512 on canonical test20.
- Artifact: `research/diagnostics/eval_grpo_vllm_v1_clean_300_v2_step270_test20_3seed.tar.gz`.
- Lesson: always merge FSDP shards before standalone eval. For future GRPO runs, include `"hf_model"` in `CHECKPOINT_SAVE_CONTENTS` for steps expected to be evaluated.

## Guardrails

- Keep this fork focused on execution and migration docs; do not turn it into an evidence mirror.
- Mark old-repo numbers as historical unless re-run in this latest-VERL fork.
- SFT, GRPO, and SDPO results are comparable only if they run end-to-end in this fork or in a single pinned latest-VERL remote clone.
- Keep generated outputs out of `research/` unless they are concise summaries or plans.
