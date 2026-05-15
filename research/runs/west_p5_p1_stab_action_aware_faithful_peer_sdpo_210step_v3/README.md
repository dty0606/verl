# west_p5_p1_stab_action_aware_faithful_peer_sdpo_210step_v3

Date: 2026-05-15
Branch: codex/sdpo-p1-stabilization (HEAD `55e8b160` at launch, after `04732a2c`)
W&B: https://wandb.ai/dtygame1/SDPO-vllm-v1-p1-stabilization/runs/exdj4ir9
Host: P5 (~/verl_tau3_sdpo_p1_stab)

## Scope

210-step stabilized **action-aware** faithful peer-only SDPO run. First run with
the `04732a2c` action-aware strict reward overlay live.

Launch config (verified from W&B):
- `TOTAL_TRAINING_STEPS=210`, `TOTAL_EPOCHS=40`, `TEST_FREQ=30`, `SAVE_FREQ=30`
- `MAX_ACTOR_CKPT_TO_KEEP=3` (Codex flagged this as smaller than the
  intended `8`; matters for ckpt retention, not training quality)
- `default_local_dir` under `~/verl_tau3_sdpo_p1_stab/checkpoints/...`
  (intended `~/tw/checkpoints`; same caveat — storage routing only)
- `VAL_N=4` ✓
- Faithful peer-only contract: `SDPO_ARM=peer_only`,
  `TAU3_LIVE_FEEDBACK_FORMAT=none`, `TAU3_LIVE_RUNTIME=official_gym`,
  `TAU3_MARK_ENV_EXCEPTIONS=false`, `TAU3_STRICT_ACTION_REWARD=1`,
  `tau3.sdpo.target_guard.enabled=false`, no memory.

## Why this run is research-relevant

The action-aware strict reward overlay is active for every sample
(`tau3_live/strict_action_overlay_enabled_fraction=1.0` for all 93+ logged
steps). Two new metrics emit cleanly:

- `tau3_live/official_success_zero_tool_with_actions_fraction`
- `tau3_live/official_success_zero_tool_without_actions_fraction`

Yet the policy **collapses toward low-tool / action-missing behavior**:

| Window | strict_score | tool_count | empty_target_skip | val/pass^4 |
|---|---|---|---|---|
| steps 1-30 | 0.50 | 5.95 | 0.40 | 0.25 (@30) |
| steps 31-60 | 0.46 | 4.94 | 0.33 | 0.05 (@60) |
| steps 61-93 | 0.17 | 1.93 | 0.61 | 0.05 (@90) |

`same_uid_strict_all_fail_fraction` rises 0.42 → 0.45 → 0.77, so the SDPO
peer signal vanishes precisely when the actor needs it most. This is a
load-bearing observation for the drift-localized hindsight method
direction.

## Diagnostic Tarball

`west_p5_p1_stab_action_aware_faithful_peer_sdpo_210step_v3_diagnostics_20260515_214020.tgz`
(57 MB, S3/local only; `.tgz` files are git-ignored)

Contents:
- `home/sagemaker-user/tw/logs/<RUN_STEM>.nohup.log` — full step trace
- `home/sagemaker-user/tw/logs/<RUN_STEM>.pid`
- `home/sagemaker-user/tw/outputs/<RUN_STEM>/rollout_data/{1..108}.jsonl` —
  108 rollout step files, 32 rows each (TBD verify)

## S3 Mirror (canonical)

```
s3://tianyd-rlvr-research/tau3-sdpo/diagnostics/west_p5_p1_stab_action_aware_faithful_peer_sdpo_210step_v3/
  west_p5_p1_stab_action_aware_faithful_peer_sdpo_210step_v3_diagnostics_20260515_214020.tgz
```

Pull command (from laptop, profile `training-role`, region us-west-2):

```bash
AWS_PROFILE=training-role aws s3 cp \
  s3://tianyd-rlvr-research/tau3-sdpo/diagnostics/west_p5_p1_stab_action_aware_faithful_peer_sdpo_210step_v3/west_p5_p1_stab_action_aware_faithful_peer_sdpo_210step_v3_diagnostics_20260515_214020.tgz \
  research/runs/west_p5_p1_stab_action_aware_faithful_peer_sdpo_210step_v3/ \
  --region us-west-2
```

## Verification

- Schema: rollout JSONL rows include top-level `uid`, `task_id`,
  `strict_score` per stabilization contract.
- Patch active: action-aware overlay enabled fraction = 1.0 throughout.
- Step coverage: 108 rollout files captured (run was at step 108 when
  packaged at TS `20260515_214020`).
