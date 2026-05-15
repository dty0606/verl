# west_p5_p1_stab_rollout_collection_30step_v1

Date: 2026-05-14
Branch: codex/sdpo-p1-stabilization
Host: P5 (~/verl_tau3_sdpo_p1_stab)

## Scope

30-step stabilized peer-only SDPO rollout collection on Tau3 airline canonical
train split. No dynamic sampling, no live cross-rollout teacher context, no NL
assertion feedback, no historical rollout memory, no synthetic feedback.

## Artifacts

- `west_p5_p1_stab_rollout_collection_30step_v1.tar.gz` — bundled run artifacts
  (force-added past `.gitignore`'s `*.tar.gz` rule):
  - `west_p5_p1_stab_rollout_collection_30step_v1.nohup.log`
  - `west_p5_p1_stab_rollout_collection_30step_v1.pid`
  - `rollout_data/{1..30}.jsonl` (32 rows per step, 960 rows total)

## S3 Mirror

Canonical home for raw artifacts (use `--profile training-role`):

```
s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/runs/west_p5_p1_stab_rollout_collection_30step_v1/
s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/runs/west_p5_p1_stab_rollout_collection_30step_v1.tar.gz
```

## Verification

- Rollout JSONLs: 30 files × 32 rows = 960 rows.
- Schema confirmed to include `uid`, `task_id`, `strict_score` per stabilization
  contract (see `research/session_sync.md` 2026-05-14 entry).
