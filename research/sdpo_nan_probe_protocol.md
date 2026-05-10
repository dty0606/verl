# SDPO NaN Probe Protocol

This note records the current NaN diagnosis strategy for Tau3 SDPO.

## What We Know

- Prior W&B run `0xcb5ugy` first logged non-finite values at `training/global_step=3`.
- The first non-finite metrics were `actor/loss`, `actor/pg_loss`, `actor/grad_norm`, and `actor/self_distillation/teacher_topk_mass`.
- The step-3 NaN was not a length spike: response max was lower than step 2, and CUDA memory was not near OOM.
- A later 5-step fail-fast rerun with the same core r4k/b4n8 recipe reached `5/5` locally with finite loss/grad/top-k mass.
- Therefore the NaN has not reproduced deterministically from the high-level recipe alone. It is likely sensitive to rollout nondeterminism, model numeric path, or a rare batch/tensor state.
- The 30-step P5 fail-fast rerun archived as `research/diagnostics/sdpo_nan_probe_30step_student_topk_mass.tgz` caught the first concrete bug before step 1 completed: `student_topk_log_probs` had invalid top-k mass at no-padding position `[0, 3767]`, repeated across ranks.
- Root cause for that run: full-sequence top-k tensors use zero-filled placeholders outside response-prediction positions, but actor loss validated/gathered all no-padding positions. Non-response placeholder rows are not probability distributions and must be sanitized or ignored before top-k/JSD validation.

## Probe Goal

Do not just observe `actor/loss=NaN`. The probe must identify the first phase that becomes non-finite:

1. Raw model logits before temperature scaling.
2. Scaled logits after temperature scaling.
3. Student top-k support logprobs.
4. EMA teacher logprobs gathered on student support.
5. Actor-side student `log_softmax` or gathered student top-k logprobs.
6. Top-k tail/JSD terms or selected per-token SDPO loss.
7. Actor parameters before backward.
8. Scalar actor loss before backward.
9. Actor gradients before optimizer step.
10. Actor parameters immediately before optimizer step.
11. AdamW/FSDP optimizer state immediately before optimizer step.
12. Actor parameters immediately after optimizer step.
13. AdamW/FSDP optimizer state immediately after optimizer step.
14. EMA teacher parameters after EMA update.

## Current Code Probes

Enable with:

```bash
export SDPO_FAIL_FAST_NONFINITE=1
export SDPO_EMA_FINITE_CHECK=1
export SDPO_LOGPROB_DIAGNOSTICS=1
export SDPO_CUDA_MEMORY_DIAGNOSTICS=1
export HYDRA_FULL_ERROR=1
```

The current instrumentation checks:

- FSDP raw/scaled logits and temperature tensors when fail-fast is enabled.
- FSDP top-k logprobs before returning from the logprob worker.
- Trainer-side top-k tensors after no-padding to padding conversion.
- Actor-loss teacher top-k logprobs always.
- Actor-loss student logits, student logprobs, gathered student top-k logprobs, JSD terms, and selected per-token SDPO loss when fail-fast is enabled.
- Actor parameters before backward, scalar actor loss before backward, gradients before optimizer step, actor parameters before optimizer step, optimizer state before optimizer step, actor parameters after optimizer step, and optimizer state after optimizer step when fail-fast is enabled.
- Actor and teacher parameters before EMA update, and teacher parameters after EMA update when fail-fast is enabled.

## Important Guard Fix

Masked-out NaNs can still poison sequence aggregations through `0 * NaN = NaN`.

The SDPO loss now zeroes masked-out full-logit distillation loss before aggregation, and sequence aggregation modes use `torch.where(mask, loss, 0)` rather than direct multiplication.

## Response-Prediction Mask Fix

Full-logit SDPO places response-token top-k support at full-sequence prediction
positions: the final prompt token predicts response token 0, response token 0
predicts response token 1, and so on. All other full-sequence positions are
zero-filled placeholders.

Actor update must not validate those placeholder rows as distributions. The loss
path now derives the no-padding response-prediction-position mask, validates
top-k IDs only on those positions, and replaces non-response rows with a finite
degenerate distribution before JSD/tail computation.

## 2026-05-10 Probe v2 Result

Kiro's v2 rerun archived `west_p5_original_sdpo_r4k_b4n8_nan_probe_30step_v2`
confirmed the response-prediction placeholder fix: the previous
`student_topk_log_probs` invalid-mass failure did not reappear.

The new first failure was in the EMA finite checker before step 1 completed:

```text
RuntimeError: Non-finite actor parameter after SDPO EMA device/dtype transfer
name=_fsdp_wrapped_module.model.visual.blocks.*._fsdp_wrapped_module._flat_param
shape=(1574528,) bad_count=0
```

Because `bad_count=0`, the checker did not identify any concrete NaN/Inf entry.
This is treated as an FSDP/flat-parameter diagnostic false positive, not model
corruption. Finite-check helpers now raise only when a concrete bad-entry count
is positive, while still reporting `first_bad`, `finite_min`, and `finite_max`
for real failures.

The extracted P5 log was moved out of Git to:

```text
D:\AI_research\sdop\logs\20260509_205035_sdpo_nan_probe_v2_ema_false_positive\
```

## 2026-05-10 Probe v3 Result

Kiro's v3 rerun archived `west_p5_original_sdpo_r4k_b4n8_nan_probe_30step_v3`
confirmed both earlier fixes:

- the top-k placeholder-row invalid-mass failure did not reappear;
- `bad_count=0` EMA finite-check false positives no longer stopped the run.

The new first failure was a real actor-parameter corruption after the first
actor update:

```text
RuntimeError: Non-finite actor parameter after SDPO EMA device/dtype transfer
name=_fsdp_wrapped_module.model.visual.blocks.12._fsdp_wrapped_module._flat_param
shape=(1574528,) bad_count=2 first_bad=[735396]
```

Tau3 airline runs are text-only but use a Qwen3.5 VLM checkpoint. The visual
tower was still trainable, so unused visual parameters could receive RL
gradients and become non-finite. The Tau3 GRPO/SDPO configs and P5 launchers
now freeze the visual tower by default, and the FSDP engine honors
`actor.freeze_vision_tower` by freezing visual/vision flat parameters after
FSDP wrapping and before optimizer construction.

The follow-up hardening is layered rather than just hiding the check. In
language-only Tau3, Qwen3.5 should not run the dummy no-image `model.visual(...)`
branch in either the actor or ref/EMA teacher, because `0.0 * image_embeds.mean()`
can still propagate NaNs if the frozen visual tower is already non-finite. The
actor optimizer and fail-fast checks also skip frozen/vision params, and SDPO EMA
skips frozen/vision read/write while preserving finite checks for trainable
language params. Optimizer construction fails loudly if filtering leaves no
trainable parameters. W&B/local metrics surface skipped EMA tensors via
`self_distillation/ema_teacher_skipped_vision_param_tensors_*`.

The extracted P5 log was moved out of Git to:

```text
D:\AI_research\sdop\logs\20260509_212600_sdpo_nan_probe_v3_actor_visual_nan\
```

## 2026-05-10 Probe v4 Direction

After vision isolation was pushed, Kiro reported a different first concrete
failure:

```text
name=_fsdp_wrapped_module.model.language_model.layers.8._fsdp_wrapped_module._flat_param
shape=(14115480,) bad_count=1 first_bad=[7079771]
```

This means the remaining non-finite is now in the SDPO actor-update path for a
trainable language parameter, not the frozen visual tower. Do not label this as
"extreme SDPO gradient" until the first bad phase is classified. The next probe
must distinguish:

- parameter already bad before backward;
- scalar actor loss or selected SDPO terms bad before backward;
- local gradient bad after backward;
- optimizer state bad before the step;
- finite params/grads/state before step but bad actor parameter after
  `optimizer.step()`;
- clean actor update followed by EMA corruption.

The current fail-fast code includes optimizer-step forensics for this exact
classification.

Probe v4 then showed the bad trainable language actor parameter was still first
caught inside EMA, but specifically after copying/casting the finite actor shard
to the teacher device/dtype:

```text
Non-finite actor parameter after SDPO EMA device/dtype transfer
name=_fsdp_wrapped_module.model.language_model.layers.9._fsdp_wrapped_module._flat_param
bad_count=1
```

This is more specific than "actor param is NaN." The actor-side tensor passed
the pre-transfer finite check, so the next diagnostic records the source dtype,
target dtype, source finite min/max/abs-max, and source value at the first bad
index. If the source value is huge but finite and the target dtype is narrower,
the immediate mechanism is cast overflow during EMA transfer; the upstream cause
is still the SDPO actor update producing an unstable finite outlier.

Probe v5 ruled out cast overflow. The first bad transferred index had a small
finite CUDA fp32 source value and no large source outliers:

```text
source_value_at_first_bad=0.00010579199442872778
source_dtype=torch.float32
source_device=cuda:0
source_abs_max=0.3827260434627533
target_dtype=torch.float32
target_device=cpu
```

The same float32 value became non-finite only after CUDA-to-CPU transfer. That
points to an asynchronous transfer/offload/resharding race, not SDPO loss scale,
not dtype narrowing, and not a huge actor weight. The EMA actor-to-teacher copy
now uses a blocking transfer (`non_blocking=False`) and synchronizes before the
finite check when fail-fast is enabled.

## P5 Reproduction Recipe

Use the direct SDPO launcher rather than relying on the capacity-matrix wrapper when debugging NaNs.

```bash
unset TORCH_SHOW_CPP_STACKTRACES
export TORCH_DISABLE_ADDR2LINE=1

export RUN_STEM=west_p5_original_sdpo_r4k_b4n8_nan_probe_30step_v1
export PROJECT_NAME=SDPO-vllm-v1-original-sdpo-debug
export TOTAL_TRAINING_STEPS=30
export TOTAL_EPOCHS=30
export TEST_FREQ=0
export SAVE_FREQ=0

export TRAIN_BATCH_SIZE=4
export PPO_MINI_BATCH_SIZE=4
export ROLLOUT_BATCH_SIZE=8
export MAX_RESPONSE_LENGTH=4096
export MAX_MODEL_LEN=12288
export SDPO_MAX_REPROMPT_LEN=4096
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=12288

export SDPO_FAIL_FAST_NONFINITE=1
export SDPO_EMA_FINITE_CHECK=1
export SDPO_LOGPROB_DIAGNOSTICS=1
export SDPO_CUDA_MEMORY_DIAGNOSTICS=1
export HYDRA_FULL_ERROR=1

bash run_local_tau3_sdpo_live_p5.sh datasets/tau3_live_airline_canonical_json "$RUN_STEM" json
```

## Reproduction Criteria

A run confirms the NaN bug only if it produces one of the explicit non-finite RuntimeErrors, or logs the same first non-finite actor/teacher metrics before any OOM, env crash, checkpoint failure, or W&B teardown failure.

If it OOMs first, that is a memory reproduction, not a NaN reproduction.

If W&B says crashed but the local nohup reaches the configured final step, treat W&B as a teardown/sync failure and use the local log as source of truth.
