# west_p5_p1_stab_faithful_peer_sdpo_210step_v1

Date: 2026-05-15
Branch: codex/sdpo-p1-stabilization (HEAD `b2da2b06` at launch)
Host: P5 (~/verl_tau3_sdpo_p1_stab)

## Scope

210-step stabilized faithful peer-only SDPO overnight run on Tau3 airline
canonical train split. No dynamic sampling, no live cross-rollout teacher
context, no NL assertion feedback, no historical rollout memory, no synthetic
feedback.

Launch config:

- `TOTAL_TRAINING_STEPS=210`, `TOTAL_EPOCHS=40`, `TEST_FREQ=30`, `SAVE_FREQ=30`
- `MAX_ACTOR_CKPT_TO_KEEP=3`
- `ROLLOUT_DATA_DIR=~/tw/outputs/${RUN_STEM}/rollout_data`
- Third launcher arg: `none` (faithful peer-only contract; no heuristic feedback)

## Snapshot Artifacts (this commit)

- `README.md` — this file (only thing committed to git per the
  pointer-only-in-git rule established 2026-05-15).
- `${RUN_STEM}_rollouts_logs_20260515_120758.tar.gz` — local-only, git-ignored
  (`.gitignore: **/*.tar.gz`). 78 MB. Contains nohup log + pid + rollout JSONLs
  for steps 1..171 captured at the time of pull.

The 78 MB tarball is intentionally NOT in git. Pull from S3 if you need it.

## Snapshot Step Coverage

- Rollout JSONLs: steps 1..171 captured in this snapshot (run was still active).
- Final-state coverage will be re-captured in a later tarball with a different
  timestamp suffix.

## S3 Mirror

Canonical home for raw artifacts (use `--profile training-role`):

```
s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/runs/west_p5_p1_stab_faithful_peer_sdpo_210step_v1/
  west_p5_p1_stab_faithful_peer_sdpo_210step_v1_rollouts_logs_20260515_120758.tar.gz
  rollout_data/        (raw mirror, easier for spot-grep)
  west_p5_p1_stab_faithful_peer_sdpo_210step_v1.nohup.log
  west_p5_p1_stab_faithful_peer_sdpo_210step_v1.pid
  checkpoint_inventory_<TS>.txt
```

Pull command (from laptop):

```bash
AWS_PROFILE=training-role aws s3 cp \
  s3://tianyd-rlvr-research/tau3-sdpo/latest-verl/runs/west_p5_p1_stab_faithful_peer_sdpo_210step_v1/west_p5_p1_stab_faithful_peer_sdpo_210step_v1_rollouts_logs_20260515_120758.tar.gz \
  research/runs/west_p5_p1_stab_faithful_peer_sdpo_210step_v1/ --region us-west-2
```

## Verification

- Schema: rollout JSONL rows include top-level `uid`, `task_id`, `strict_score`
  per stabilization contract (see `research/session_sync.md` 2026-05-14 entry).
- Row count: 32 rows per step (`train_batch_size=4` × `rollout.n=8`).
- Checkpoint pruning: `MAX_ACTOR_CKPT_TO_KEEP=3` is in effect on P5; the
  in-place checkpoints will be pruned to the last 3 boundaries. P5 preserved
  the current set under `~/tw/outputs/${RUN_STEM}/checkpoints_preserved_<TS>/`
  via hardlink (or full copy fallback) before further pruning.
