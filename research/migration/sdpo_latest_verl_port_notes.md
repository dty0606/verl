# Tau3 SDPO Latest-VERL Port Notes

Status: vanilla SDPO port implemented for smoke testing.

Latest VERL no longer has the old local fork's FSDP actor worker path, so the
port does not copy the old worker wholesale. Instead, the latest trainer builds
the SDPO teacher/reprompt batch on the driver after Tau3 rewards are available,
scores the sampled response under that reprompt with the actor log-prob path,
then attaches response-shaped `teacher_logprobs` and `self_distillation_mask`
before actor update.

Implemented vanilla SDPO path:

1. Keep the Tau3 GRPO rollout path as the source of sampled trajectories,
   rewards, and feedback diagnostics.
2. After rewards are available, select failed samples with usable feedback and
   construct SDPO reprompts without memory/retrieval augmentation.
3. Score the original response under the teacher/reprompt context through
   `actor_rollout_wg.compute_log_prob`.
4. Attach response-shaped `teacher_logprobs`, `self_distillation_mask`, and
   `self_distillation_loss_mask` to the rollout `DataProto`.
5. Use `actor.policy_loss.loss_mode=sdpo` in latest `ppo_loss` for response-token
   reverse-KL SDPO.

Current scope:

- Vanilla SDPO only.
- No memory/DENSE retrieval.
- No full-logit/top-k distillation.
- No separate teacher model pool.
- No MT-STePO endpoint-boundary variant.

Next validation:

1. Run a one-step Tau3 SDPO smoke with the 10-step full-trajectory SFT checkpoint.
2. Check for non-empty `self_distillation/reprompt_sample_fraction` on failed
   rollouts with feedback.
3. Check that `actor/pg_loss` and `self_distillation/token_fraction` are finite.
4. Only after this smoke passes, compare against the GRPO baseline.
