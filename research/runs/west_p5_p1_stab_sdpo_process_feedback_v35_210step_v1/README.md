# west_p5_p1_stab_sdpo_process_feedback_v35_210step_v1

Date: 2026-05-17
Branch: codex/sdpo-p1-stabilization
W&B: https://wandb.ai/dtygame1/SDPO-vllm-v1-p1-stabilization/runs/066jhtdz
Host: P5 (~/verl_tau3_sdpo_p1_stab)

## Scope

210-step **v3.5** SDPO run with controlled teacher context:
- `SDPO_GT_METADATA_ENABLED=true` (redacted GT metadata in teacher prompt)
- `SDPO_FAILED_PEER_ENABLED=true` (one same-UID failed assistant evidence)
- `SDPO_FAILED_PEER_MAX_CHARS=4096`
- `MAX_ACTOR_CKPT_TO_KEEP=8` (all save boundaries preserved)
- `VAL_N=4`
- Faithful peer-only contract otherwise: `SDPO_ARM=peer_only`,
  `TAU3_LIVE_FEEDBACK_FORMAT=none`, `TAU3_LIVE_RUNTIME=official_gym`,
  `TAU3_MARK_ENV_EXCEPTIONS=false`, `TAU3_STRICT_ACTION_REWARD=1`,
  `tau3.sdpo.target_guard.enabled=false`, no memory.

## Observation

Mechanically healthy but behaviorally collapsed after ~step 40. Codex
requested rollout JSONLs for steps 34, 38, 40-45 (transition window),
75/81/84 (collapsed), and 90 (post-collapse) for offline diagnosis.

## Diagnostic Tarball (local-only, S3-canonical)

`west_p5_p1_stab_sdpo_process_feedback_v35_210step_v1_diagnose_20260517_184507.tgz`
(32 MB, git-ignored via `research/runs/**/*.tgz`)

Contents:
- `<RUN_STEM>.nohup.log` — full step trace
- `<RUN_STEM>.pid`
- `rollout_data/{34,38,40,41,42,43,44,45,75,81,84,90}.jsonl` — 12 step
  files spanning pre-collapse, transition, and collapsed regimes
- `checkpoint_inventory/inventory_20260517_184507.txt` — ckpt dirs and
  EMA teacher shard counts (no shards copied)

## S3 Mirror (canonical)

```
s3://tianyd-rlvr-research/tau3-sdpo/diagnostics/west_p5_p1_stab_sdpo_process_feedback_v35_210step_v1/
  west_p5_p1_stab_sdpo_process_feedback_v35_210step_v1_diagnose_20260517_184507.tgz
```

Pull command (laptop, profile `training-role`, region us-west-2):

```bash
AWS_PROFILE=training-role aws s3 cp \
  s3://tianyd-rlvr-research/tau3-sdpo/diagnostics/west_p5_p1_stab_sdpo_process_feedback_v35_210step_v1/west_p5_p1_stab_sdpo_process_feedback_v35_210step_v1_diagnose_20260517_184507.tgz \
  research/runs/west_p5_p1_stab_sdpo_process_feedback_v35_210step_v1/ \
  --region us-west-2
```

## Codex's Diagnostic Questions

1. What shortcut appears after collapse? (text-only? fake "I checked"?
   premature transfer? refusal? short generic answer?)
2. Compare good low-tool (steps 34, 38) vs transition (40-45) vs
   collapsed (75/81/84). What changed behaviorally?
3. For rows with `official_score=1` AND `strict_score=0` AND `tool_count=0`:
   list `task_id`, `uid`, scores, final response excerpt, expected vs
   executed vs missing action names. Classify failure mode.
4. For `active_without_success_peer` rows: does failed-evidence dominate
   the all-fail path? Does selected evidence resemble the later shortcut?
   Self-fallback frequency? Likelihood of teacher pollution?
5. For successful-peer rows: do mixed groups still beat all-fail
   GT/failed-evidence rows? Compare strict/tool by teacher context type.
6. Inspect prompt/target metadata: GT metadata redacted only? Any
   argument-value or communicate-value leakage? Failed snippets
   assistant-only and `<think>`-stripped?
7. Likely cause classification:
   - A. redacted GT too weak
   - B. failed assistant evidence pollutes teacher
   - C. distilling over original failed response cannot correct first
        missing tool decision
   - D. top-k support cannot teach tool tokens
   - E. official reward shortcut still dominates
   - F. batch/task mix artifact
