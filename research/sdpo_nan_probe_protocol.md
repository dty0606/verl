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
7. Actor gradients before optimizer step.
8. Actor parameters after optimizer step.
9. EMA teacher parameters after EMA update.

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
- Actor gradients before optimizer step and actor parameters after optimizer step when fail-fast is enabled.
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
