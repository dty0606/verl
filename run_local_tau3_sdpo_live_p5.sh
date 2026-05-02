#!/usr/bin/env bash
# Tau3 SDPO latest-VERL port entrypoint.
#
# This is intentionally fail-fast until the SDPO trainer hook is ported to
# latest VERL's distillation/engine-workers stack. Do not run the old fork's
# actor.policy_loss.loss_mode=sdpo config here; latest VERL only knows vanilla,
# clip-cov, kl-cov, and gpg policy losses.

set -euo pipefail

cat >&2 <<'EOF'
Tau3 SDPO is not runnable on latest VERL yet.

GRPO is the first migrated baseline. Vanilla SDPO needs a real latest-VERL port:
- build SDPO reprompt/teacher targets after rollout rewards;
- attach response-shaped teacher logprobs/ids through DataProto;
- register the SDPO loss through latest VERL distillation/loss hooks;
- keep feedback/memory SDPO disabled until vanilla SDPO is validated.

Use ./run_local_tau3_grpo_live_p5.sh for the current runnable baseline.
See research/migration/sdpo_latest_verl_port_notes.md for the port plan.
EOF

exit 2
