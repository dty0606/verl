# Tau3 SDPO Latest-VERL Port Notes

Status: scaffolded, not runnable yet.

Latest VERL no longer has the old local fork's `actor.policy_loss.loss_mode=sdpo`
path. The supported actor policy losses are vanilla, clip-cov, kl-cov, and gpg.
The new integration point is the upstream distillation stack plus
`engine_workers.py`, not the old FSDP worker patches.

Minimum vanilla SDPO port:

1. Keep the Tau3 GRPO rollout path as the source of sampled trajectories,
   rewards, and feedback diagnostics.
2. After rewards are available, select failed samples with usable feedback and
   construct SDPO reprompts without memory/retrieval augmentation.
3. Score the original response under the teacher/reprompt context and attach
   response-shaped teacher ids/logprobs to the rollout `DataProto`.
4. Register an SDPO distillation loss in latest VERL's distillation loss
   registry, instead of adding a new actor `loss_mode`.
5. Validate on a one-step smoke run before enabling any memory SDPO or feedback
   augmentation variants.

Do not port the old `self_distillation` actor config wholesale. It was tied to
the previous fork's trainer and worker layout and will silently mislead the run.
