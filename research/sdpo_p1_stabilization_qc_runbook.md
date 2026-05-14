# Tau3 SDPO P1 Stabilization QC Runbook

This is a harness/checklist only. It does not change training semantics and must
not be used to justify dynamic sampling, live cross-rollout context, NL
assertions, or synthetic feedback.

## Local One-Command Test Set

Run from the repo root:

```bash
python -m pytest tests/utils/test_tau3_sdpo_full_logit_loss.py tests/utils/test_tau3_sdpo_ema_teacher.py tests/utils/test_tau3_faithful_sdpo_config.py tests/utils/test_tau3_length_metrics.py tests/utils/test_rollout_skip_on_cpu.py tests/utils/test_mlflow_key_sanitization.py tests/utils/test_sdpo_p1_stabilization_qc.py
python scripts/tau3/sdpo_p1_stabilization_qc.py --help
```

## P5 Smoke Command

Capture the log so the QC harness can scan it after the run:

```bash
SDPO_LOGPROB_DIAGNOSTICS=1 \
SDPO_CUDA_MEMORY_DIAGNOSTICS=1 \
SDPO_FAIL_FAST_NONFINITE=1 \
SDPO_EMA_FINITE_CHECK=1 \
TOTAL_TRAINING_STEPS=2 \
TEST_FREQ=1 \
SAVE_FREQ=1 \
bash run_local_tau3_sdpo_live_p5.sh "$TASK_PATH" "$RUN_NAME" none 2>&1 | tee "$RUN_LOG"
```

Then scan the captured evidence:

```bash
python scripts/tau3/sdpo_p1_stabilization_qc.py \
  --log "$RUN_LOG" \
  --metrics-json "$METRICS_JSON" \
  --update-counter-changed \
  --output-md "research/${RUN_NAME}_sdpo_p1_qc_report.md"
```

Add `--mask-debug-enabled` only when the smoke explicitly enables mask-debug
logging. Add `--teacher-scheduling-enabled` only when the smoke enables the
teacher scheduling/equivalence path.

## Required Gates

- Strict reward metrics present: evidence includes `pass^1`/`pass_at_1` and a
  reward mean metric such as `reward/mean@1`, `reward/mean_at_1`, or
  `reward/mean`.
- Mask-debug sample present when enabled: if mask-debug logging is enabled, the
  log/report must include at least one selected/target mask sample.
- Checkpoint saved when update counter changed: if the actor update counter
  changed, the log/report must include checkpoint-save evidence for that step.
- Early skip cause logged: empty-target or optimizer-step skip paths must expose
  a concrete metric/log cause, not just an absent update.
- CUDA memory phase metrics present: with `SDPO_CUDA_MEMORY_DIAGNOSTICS=1`, the
  log must include `ref_compute_log_prob`, `actor_compute_log_prob`, and
  `actor_update` phase evidence.
- Teacher scheduling equivalence if enabled: when a teacher scheduling path is
  enabled, the log/metrics must show explicit equivalence or teacher-backend
  evidence.

## Report Template

```markdown
# Tau3 SDPO P1 Stabilization QC Report

## Scope
- Branch: `codex/sdpo-p1-stabilization`
- Harness only: no trainer/reward/worker semantics changed.
- Smoke target: P1 stabilization on P5 with faithful peer-only SDPO.

## Local Tests
- Command:
- Result:

## P5 Smoke
- Command:
- Result:
- Run log:
- Metrics artifact:
- Checkpoint directory:
- W&B run:

## Gates
- [ ] Strict reward metrics present.
- [ ] Mask-debug sample present when enabled.
- [ ] Checkpoint saved when update counter changed.
- [ ] Early skip cause logged.
- [ ] CUDA memory phase metrics present.
- [ ] Teacher scheduling equivalence if enabled.

## Global Safety Sign-Off
- [ ] No dynamic sampling.
- [ ] No live cross-rollout context.
- [ ] No NL assertion.
- [ ] No synthetic feedback.

## Reviewer Notes
- TODO
```
