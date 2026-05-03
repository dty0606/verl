# GRPO / SDPO Algorithm Fidelity QC

Date: 2026-05-03

Repo state reviewed: `e431696b` plus the Guardian fixset in this commit, including the SDPO loss-normalization patch in `verl/workers/utils/losses.py`.

Evidence bundles:

- `research/diagnostics/grpo_fulltraj_smoke_full.zip`
- `research/diagnostics/sdpo_vlm_sft_smoke.zip`

Primary references:

- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- Reinforcement Learning via Self-Distillation / SDPO: https://arxiv.org/abs/2601.20802

SDPO-paper baseline note: Appendix E.2 Table 13 of the SDPO paper lists the GRPO comparison hyperparameters as number of rollouts `8`, rollout importance-sampling clip `2`, and KL coefficient `0.0`. Our main GRPO baseline should therefore be described as the SDPO-paper companion GRPO baseline, not as a separate DeepSeekMath-reference GRPO ablation.

## Executive Verdict

No fatal engineering blocker was found for the full-trajectory SFT path. The full-traj data contract, preserve-thinking template, pre-tokenized `input_ids` / `loss_mask`, and smoke checkpoint path are coherent.

One real SDPO training bug was found and patched locally: SDPO selected-token loss aggregation used local target-token counts and implicit `dp_size=1`, unlike the vanilla PPO path. This was safe for the one-step smoke but not safe for multi-GPU claim runs. The local patch all-reduces selected-token counts and selected-sequence counts across `dp_group` before the empty-target branch, then passes `dp_size`, `batch_num_tokens`, `global_batch_size`, and `loss_scale_factor` into `agg_loss`.

The larger paper-fidelity issue is naming/config, not a crash bug. Current Tau3 GRPO should be framed as the SDPO-paper companion GRPO baseline: KL coefficient `0.0`, rollout `n=8`, rollout importance-sampling clip `2`, and no critic/value model. This is not literal DeepSeekMath-reference GRPO, which includes group std normalization and reference-model KL, but it is aligned with the comparison baseline we want for sampled-token SDPO.

Majority vote:

- Full-trajectory SFT: GO.
- GRPO/SDPO smoke wiring: GO.
- SDPO claim run before the local normalization fix: NO-GO.
- SDPO claim run after the normalization fix: conditional GO, but run a short mixed-success/terminal-reward smoke first.
- "Reference DeepSeekMath GRPO" claim with current config: NO-GO. For this project, call it the SDPO-paper companion GRPO baseline.

## Smoke Evidence

The paired smokes used the same 10-step full-traj SFT checkpoint, same Tau3 airline dataset, same prompt/response budgets, and same one-step engineering setup.

GRPO smoke:

- 8 rollout rows.
- Rewards all `0.0`.
- Feedback present in 8/8 rows.
- No gibberish detected in decoded outputs.
- `actor/pg_loss=0.0`, `actor/grad_norm=0.0`.
- Interpretation: expected zero update because all rewards are zero, so group-relative advantages have no variance.

SDPO smoke:

- 8 rollout rows.
- Rewards all `0.0`.
- Feedback present in 8/8 rows.
- No gibberish detected in decoded outputs.
- `self_distillation/reprompt_sample_fraction=1.0`.
- `self_distillation/feedback_used_fraction=1.0`.
- `self_distillation/token_fraction=1.0`.
- `self_distillation/teacher_prompt_saturation_fraction=0.0`.
- `actor/pg_loss=-0.03145`, `actor/grad_norm=33.81`.
- Interpretation: the feedback reprompt path is active and produces a finite nonzero update in the all-fail sparse-reward case.

The smoke does not prove quality. It proves shape stability, feedback plumbing, non-gibberish generation, and that SDPO still has a dense update where GRPO has zero group signal.

## Full-Trajectory SFT Path

Files checked:

- `scripts/tau3/build_protocol_sft_data.py`
- `scripts/tau3/pretokenize_full_traj_sft.py`
- `scripts/qwen35/patch_chat_template_preserve_thinking.py`
- `verl/utils/dataset/qwen35_preserve_thinking_template.py`
- `verl/utils/dataset/pretokenized_sft_dataset.py`

The data builder supports `--sft-format trajectory`, which keeps a whole successful trajectory as one SFT row. Thinking traces are preserved when `--include-thinking-traces` is enabled. This restores the old LoRA-style contract: previous assistant tool calls, tool results, and historical `<think>` blocks remain visible in context.

The full-traj pretokenizer installs the custom preserve-thinking Qwen3.5 template, renders completed prefixes, and derives per-message token boundaries. It labels only assistant-message spans in `loss_mask`; system, user, and tool observations remain masked out. The generated `segments` metadata is useful for future component-level diagnostics and memory retrieval work.

`PretokenizedSFTDataset` reads `input_ids` and `loss_mask` directly, validates equal lengths and binary masks, rejects empty masks, and protects the first token from accidental supervision. This avoids slow on-the-fly template calls in VERL SFT.

Required operational check before every SFT-to-RL handoff:

- Verify the exported checkpoint tokenizer was patched with `patch_chat_template_preserve_thinking.py`.
- Verify `tokenizer_config.json` contains `tau3_preserve_thinking_template.enabled=true` and the expected template SHA.
- Keep the pretokenize manifest with zero errors, nonzero `avg_labeled_tokens`, and no unexpected truncation.

Nonblocking caveat: the custom template is text-only. That is correct for Tau3 airline, but the patched checkpoint should not be reused as a general multimodal Qwen3.5 checkpoint.

## GRPO Fidelity

Files checked:

- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/config/tau3_grpo_live.yaml`
- `run_local_tau3_grpo_live_p5.sh`

What matches the GRPO family:

- Rollout grouping is structurally correct: one UID per prompt, repeated by `rollout.n`, then grouped by UID for outcome advantages.
- PPO-style ratio clipping is implemented in `compute_policy_loss_vanilla`.
- The critic/value path is not used for GRPO advantages.
- All-zero rewards produce zero advantages and a finite zero update, which is the correct sparse-reward edge-case behavior.

Current differences from literal DeepSeekMath GRPO:

- `algorithm.norm_adv_by_std_in_grpo=False`, while the original DeepSeekMath GRPO formulation normalizes group rewards by group standard deviation. This is acceptable for the SDPO-paper companion baseline but should be named as such.
- `actor_rollout_ref.actor.use_kl_loss=False` and `algorithm.use_kl_in_reward=False`, matching the SDPO paper's GRPO comparison KL coefficient of `0.0`. DeepSeekMath-reference GRPO would be a separate ablation with a reference-model KL term.
- `loss_agg_mode` is `token-mean`, while VERL's own GRPO notes distinguish this from the original `seq-mean-token-mean` formulation.
- Tau3 config enables rollout correction, which changes the objective from a purely on-policy vanilla GRPO objective.

Recommendation:

- If the claim is "DeepSeekMath-reference GRPO", change config before the claim run: enable std-normalized group advantages, enable exactly one KL path, consider `seq-mean-token-mean`, and set rollout correction according to that separate ablation's target objective.
- If the claim is "SDPO-paper companion GRPO baseline", current config is acceptable: the SDPO paper's reported GRPO hyperparameters use KL coefficient `0.0`, rollout `n=8`, and rollout importance-sampling clip `2`.

## SDPO Fidelity

Files checked:

- `verl/trainer/ppo/ray_trainer.py`
- `verl/workers/utils/losses.py`
- `verl/trainer/config/tau3_sdpo_live.yaml`
- `run_local_tau3_sdpo_live_p5.sh`
- `verl/utils/reward_score/feedback/tau3_live.py`

What matches the intended vanilla sampled SDPO path:

- Failed samples with feedback are selected by `only_failed_with_feedback=True`.
- Tau3 feedback is emitted as `feedback` and carried through reward extra infos into the SDPO reprompt builder.
- Teacher prompts are built from the original task, environment feedback, and the configured feedback template.
- Teacher logprobs are computed over the student's sampled response tokens.
- The loss uses the sampled-token reverse-KL policy-gradient form: `detach(log pi_student - log pi_teacher) * log pi_student`.
- Truncated importance sampling is applied as `exp(log_prob - old_log_prob)` with `sdpo_is_clip=2.0`.
- Non-target samples stay shape-compatible and contribute zero SDPO gradient.

Important paper-fidelity caveat:

The SDPO paper reports top-k logit-level distillation for its main experiments and also discusses token-level and sequence-level variants. This repo currently implements sampled response-token reverse-KL only. That is a valid narrow "vanilla sampled SDPO" variant, but it should not be described as full top-k logit SDPO.

Fixed locally:

- The SDPO branch previously called `agg_loss` with local `sdpo_loss_mask.sum()` and no `dp_size`, unlike the vanilla PPO branch that passes global batch info. This made multi-GPU loss scale depend on per-rank selected-token distribution.
- The local patch all-reduces selected-token and selected-sequence counts across `dp_group` before the empty-target branch and passes the proper global counts to `agg_loss`. This keeps every rank on the same control-flow path and preserves global loss scaling.

Remaining nonblocking concerns:

- `raw_prompt[-1]["content"]` assumes the last raw prompt message is the initial user task. This holds for current Tau3 airline data but should be asserted or made robust before retail/general multi-message prompt paths.
- `actor/ppo_kl` in the SDPO branch is not teacher KL; it is a rollout-policy drift diagnostic over selected tokens. Rename or document before using it in plots.
- `use_successful_peer_solution=False` means this is feedback-only SDPO, not the successful-peer reprompt variant.
- `remove_thinking_from_demonstration=True` is currently irrelevant unless peer-solution demonstrations are enabled.

## Data Plumbing

Tau3 reward/feedback path is coherent for the smoke:

- `tau3_live.py` emits string feedback for reward `< 1.0`.
- Reward extra infos preserve the `feedback` key.
- SDPO collects `feedback`, then `teacher_feedback`, then `reward_feedback`.
- The smoke had feedback in 8/8 rows and `feedback_used_fraction=1.0`.

The smoke only exercised nonterminal/failure feedback. It did not prove terminal official reward extraction or mixed success/failure batches. Before claim runs, run a short smoke where at least one rollout reaches terminal official reward and where `reprompt_sample_fraction` is neither `0.0` nor `1.0`.

## Cross-Baseline Comparability

The paired smoke is fair as an engineering comparison: same checkpoint, same dataset, same prompt budgets, same one-step setup.

It is not enough for statistical comparison:

- Rollouts are stochastic and not guaranteed identical between GRPO and SDPO runs.
- Smoke recipes intentionally use smaller train/mini-batches, while the full GRPO/SDPO baseline launchers now share the same train batch, rollout group size, PPO mini-batch, and length defaults.
- `MODEL_PATH` defaults to base Qwen if not exported in the shell.
- `VLLM_LANGUAGE_MODEL_ONLY=true` is opt-in.

Before full baseline runs, use one shared recipe file or explicit env snapshot that pins:

- `MODEL_PATH`
- dataset path and split
- seed/task grid
- `TRAIN_BATCH_SIZE`
- `PPO_MINI_BATCH_SIZE`
- `ROLLOUT_BATCH_SIZE` / `rollout.n`
- prompt/response/model length budgets
- temperature/top-p
- `VLLM_LANGUAGE_MODEL_ONLY=true`

## Stop / Go Decision

Full SFT run:

- GO.
- Run with full-trajectory pretokenized data and preserve-thinking template.
- Patch the exported checkpoint tokenizer immediately after each save/export before vLLM/VERL rollout smoke.

GRPO baseline:

- GO as the SDPO-paper companion GRPO baseline.
- NO-GO if we intend to call it original/reference DeepSeekMath GRPO without config changes.

SDPO baseline:

- GO after the local global selected-token aggregation fix is reviewed.
- Before a long claim run, run one short real-SFT smoke with mixed success/failure if possible and verify terminal reward extraction, `reprompt_sample_fraction`, `feedback_used_fraction`, `teacher_prompt_saturation_fraction`, finite `pg_loss`, and sane `grad_norm`.

Memory SDPO / DENSE:

- Defer. The vanilla GRPO/SDPO and full-SFT contracts should be locked first.
