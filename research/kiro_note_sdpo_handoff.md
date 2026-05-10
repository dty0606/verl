# Kiro Handoff: Note-SDPO Decision Hindsight

Last updated: 2026-05-10

## Project

- Repo: `dty0606/verl_tau3_sdpo`
- Branch: `codex/note-sdpo-decision-hindsight`
- Expected P5 cwd: `/home/sagemaker-user/verl_tau3_sdpo_vllm20` or the equivalent fresh clone.
- Local drafting happened on Windows; P5/Kiro is the source of truth for CUDA, torch, omegaconf, Ray, vLLM, FSDP, and trainer integration.
- Base method: stable guarded Tau3 SDPO baseline. T3 truncation is intentionally not integrated in this branch.

## Current State

This branch adds the first Note-SDPO implementation scaffold. The goal is not yet a full training claim. The immediate milestone is an offline teacher-quality probe proving that compact train-only decision notes improve the teacher over original SDPO and random/matched-irrelevant notes, without leakage, prompt saturation, or shorter-but-unsolved bias.

Original SDPO uses same-prompt successful peers as teacher-side hindsight. Note-SDPO extends this with cross-rollout decision notes, but still keeps notes teacher-side only. The deployed student/actor must not see note text at inference time.

## Read First

Read these files before changing code:

- `research/session_sync.md`
- `research/note_sdpo_bank_pipeline.md`
- `research/kiro_note_sdpo_handoff.md`
- `scripts/tau3/build_note_sdpo_bank.py`
- `scripts/tau3/build_sdpo_memory_teacher_probe.py`
- `scripts/tau3/audit_sdpo_memory_retrieval.py`
- `verl/utils/tau3_sdpo_memory.py`
- `verl/utils/tau3_sdpo_decision_spans.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/config/tau3_sdpo_live.yaml`
- `run_local_tau3_sdpo_live_p5.sh`
- Tests under `tests/utils/test_tau3_note_sdpo_bank_builder.py`, `tests/utils/test_tau3_sdpo_memory_teacher_probe.py`, `tests/utils/test_tau3_sdpo_decision_spans.py`, and `tests/utils/test_tau3_sdpo_memory_retrieval_audit.py`.

## New Structure To Learn

### 1. Note bank builder

File: `scripts/tau3/build_note_sdpo_bank.py`

Purpose:

- Emit sanitized teacher-note-writing prompts from train rollouts.
- Optionally call an OpenAI-compatible teacher endpoint.
- Compile teacher-written JSON notes into memory cards consumed by SDPO.

Strict note schema:

- `note_type`
- `situation`
- `applicability`
- `known_evidence`
- `missing_state_required_before_action`
- `decision_boundary`
- `correct_action_pattern`
- `avoid`
- `unsafe_if`
- `stop_condition`
- `why_reusable`
- `task_cluster`
- `tool_family`
- `confidence`

Important behavior:

- `note_type` must be `decision_lesson`.
- Unknown fields are rejected.
- Empty required list fields are rejected.
- `decision_boundary` must be one of the allowed values; invalid values are rejected, not coerced.
- `confidence` must be numeric in `[0, 1]`.
- `compile` requires `--prompt-jsonl`, so unprovenanced writer rows cannot silently become train cards.
- Compiled cards include `split`, `status`, `unit_type`, `leakage_audit`, `blocked_task_ids`, `blocked_uids`, and source metadata.

Allowed decision boundaries:

- `sufficient_to_act`
- `need_more_tool_evidence`
- `need_user_clarification`
- `evidence_exhausted_negative`
- `must_refuse`
- `must_transfer`

### 2. Live memory retrieval

File: `verl/utils/tau3_sdpo_memory.py`

Purpose:

- Load compiled memory cards.
- Retrieve cards for teacher-side SDPO reprompting.

Live eligibility is intentionally strict:

- `split == train`
- `status == active`
- `unit_type == teacher_written_decision_note`
- `leakage_audit.passed is True`
- not blocked by current query text
- not blocked by metadata IDs passed from the trainer

Trainer-side blocking:

- `ray_trainer.py` passes current `uid` and, when present, `task_id` / `tau3_task_id` into `memory_bank.retrieve(..., blocked_ids=[...])`.
- This blocks same-UID/same-task retrieval when card provenance matches the current sample.

### 3. Teacher-quality probe prompt builder

File: `scripts/tau3/build_sdpo_memory_teacher_probe.py`

Purpose:

- Generate paired teacher prompt arms for offline scoring.
- It does not call a model yet.

Current prompt arms:

- `T0_original`: original SDPO teacher context with feedback.
- `T1_relevant_memory`: feedback plus relevant compact note.
- `T2_random_memory`: feedback plus random compact note.
- `T3_no_context`: original prompt only, no feedback or memory lower bound.
- `T4_matched_irrelevant_memory`: feedback plus length-matched irrelevant note.
- `T5_peer_success`: same-UID successful peer demonstration when available.

Important behavior:

- Peer-success arm strips `<think>` from the successful demo.
- Peer-success arm suppresses failure feedback, matching live SDPO behavior when a success solution is available.
- T4 is length-matched and avoids same cluster/state when possible, but it is still a heuristic negative control.

Missing next piece:

- This branch exports prompt arms only. Kiro should add or run a scoring pass that evaluates generated teacher outputs and SDPO-faithful logprob preference before any training claim.

### 4. Decision-sufficiency span shadow metrics

Files:

- `verl/utils/tau3_sdpo_decision_spans.py`
- `verl/trainer/config/tau3_sdpo_live.yaml`
- `run_local_tau3_sdpo_live_p5.sh`
- `verl/trainer/ppo/ray_trainer.py`

Purpose:

- Compute decision/action/finalization span masks and metrics.
- Default is disabled.
- Shadow mode is safe: metrics only, no loss change.

Important guardrail:

- `apply_to_loss` only takes effect when `shadow_mode=false`.
- Keep `SDPO_DECISION_WEIGHTING_APPLY_TO_LOSS=false` until offline QC proves the spans do not reward short unsolved answers.

Current limitation:

- Span detection is regex plus approximate char-to-token mapping. It is useful for shadow diagnostics but not yet strong enough for final loss weighting.

### 5. Note-SDPO live hook

File: `verl/trainer/ppo/ray_trainer.py`

Live teacher-side note injection happens in `_maybe_build_sdpo_teacher_batch`.

Important config defaults:

- `tau3.sdpo.memory.enabled=false`
- `tau3.sdpo.memory.inject_when=no_solution`
- `tau3.sdpo.memory.allow_without_feedback=false`
- `tau3.sdpo.decision_weighting.enabled=false`
- `tau3.sdpo.decision_weighting.shadow_mode=true`
- `tau3.sdpo.decision_weighting.apply_to_loss=false`

This preserves the guarded original SDPO baseline unless memory or decision weighting is explicitly enabled.

## Execute Now

### Phase 0: Pull and verify branch

```bash
cd /home/sagemaker-user/verl_tau3_sdpo_vllm20
git fetch private
git checkout codex/note-sdpo-decision-hindsight
git pull --ff-only
git log -1 --oneline
```

Expected current commit from Codex:

```text
f230c7a8 Add Note-SDPO decision hindsight scaffolding
```

If the handoff-doc commit is newer, use the latest branch head.

### Phase 1: Run local/P5 tests

Run CPU/static tests:

```bash
python -m py_compile \
  scripts/tau3/build_note_sdpo_bank.py \
  scripts/tau3/build_sdpo_memory_teacher_probe.py \
  scripts/tau3/audit_sdpo_memory_retrieval.py \
  scripts/tau3/replay_sdpo_note_memory_momentum.py \
  verl/utils/tau3_sdpo_memory.py \
  verl/utils/tau3_sdpo_decision_spans.py

pytest -q \
  tests/utils/test_tau3_note_sdpo_bank_builder.py \
  tests/utils/test_tau3_sdpo_memory_retrieval_audit.py \
  tests/utils/test_tau3_sdpo_memory_teacher_probe.py \
  tests/utils/test_tau3_sdpo_decision_spans.py
```

Run SDPO hardening tests on P5, where `torch`, `ray`, and `omegaconf` exist:

```bash
pytest -q \
  tests/utils/test_tau3_sdpo_full_logit_loss.py \
  tests/utils/test_tau3_sdpo_agg_loss_nan_mask.py \
  tests/utils/test_tau3_sdpo_ema_teacher.py \
  tests/utils/test_tau3_sdpo_target_guard.py
```

Do not treat Windows local skip results as runtime evidence. P5 is the source of truth.

### Phase 2: Build a tiny train-only note bank smoke

Use successful train rollouts only for the first bank. If the exact rollout archive path differs, locate the latest clean train rollout bundle and write the path into the run notes.

```bash
source activate sdpo-vllm20-v1

export NVME_ROOT=/mnt/sagemaker-nvme/tau3_sdpo
export NOTE_RUN=note_sdpo_v1_smoke_$(date +%Y%m%d_%H%M%S)
export NOTE_DIR="$NVME_ROOT/memory_banks/$NOTE_RUN"
mkdir -p "$NOTE_DIR"

python scripts/tau3/build_note_sdpo_bank.py emit-prompts \
  --rollout "$NVME_ROOT/memory_sources/grpo_vllm_v1_clean_300_rollouts.tgz" \
  --output-prompts "$NOTE_DIR/prompts.jsonl" \
  --manifest-json "$NOTE_DIR/emit_manifest.json" \
  --score-threshold 1 \
  --source-step-max 60 \
  --source-run-id grpo_vllm_v1_clean_300 \
  --max-source-rows 64 \
  --prompt-tail-chars 6000 \
  --output-excerpt-chars 2500 \
  --feedback-chars 2000 \
  --source-split train
```

Then write notes either with a local/OpenAI-compatible teacher endpoint or externally. The writer output must be JSONL with `prompt_id` and a strict note payload.

Compile:

```bash
python scripts/tau3/build_note_sdpo_bank.py compile \
  --prompt-jsonl "$NOTE_DIR/prompts.jsonl" \
  --writer-output-jsonl "$NOTE_DIR/writer_outputs.jsonl" \
  --output-bank "$NOTE_DIR/note_sdpo_memory_bank.jsonl" \
  --rejections-jsonl "$NOTE_DIR/rejections.jsonl" \
  --manifest-json "$NOTE_DIR/compile_manifest.json" \
  --writer-model "teacher_note_writer_smoke" \
  --initial-status active \
  --merge-similar \
  --fail-on-rejection
```

Expected:

- `compile_manifest.json` has nonzero `accepted_new_cards`.
- `rejections.jsonl` is empty if `--fail-on-rejection` is used.
- Cards are compact decision lessons, not raw trajectories.

### Phase 3: Retrieval/leakage audit

Run an offline retrieval audit before any live teacher use:

```bash
python scripts/tau3/audit_sdpo_memory_retrieval.py \
  --memory-path "$NOTE_DIR/note_sdpo_memory_bank.jsonl" \
  --rollout "$NVME_ROOT/memory_sources/grpo_vllm_v1_clean_300_rollouts.tgz" \
  --output-jsonl "$NOTE_DIR/retrieval_audit.jsonl" \
  --summary-json "$NOTE_DIR/retrieval_audit_summary.json" \
  --block-same-task-id \
  --block-same-uid
```

If CLI flags differ, inspect `--help` and preserve the same semantics: block same task, block same UID, audit train/test/source leakage, and write a summary JSON.

Pass criteria:

- No eval/test-derived note source.
- No same UID/task retrieval.
- No raw IDs, names, dates, reservation IDs, emails, prices, flight numbers, or tool JSON in retrieved note text.
- Relevant notes retrieve at nonzero rate for failed/hard samples.

### Phase 4: Teacher-quality probe

Build prompt arms:

```bash
python scripts/tau3/build_sdpo_memory_teacher_probe.py \
  --rollout "$NVME_ROOT/memory_sources/grpo_vllm_v1_clean_300_rollouts.tgz" \
  --memory-path "$NOTE_DIR/note_sdpo_memory_bank.jsonl" \
  --output-jsonl "$NOTE_DIR/teacher_probe_prompts.jsonl" \
  --max-samples 100 \
  --require-no-successful-peer
```

Then add or run a scoring pass. Minimum scoring should compare:

- generated teacher output quality
- valid next action/tool family
- whether read/search loops stop
- response length and prompt saturation
- SDPO-faithful logprob preference for correct/action tokens when possible

Required comparisons:

- `T1_relevant_memory` > `T0_original`
- `T1_relevant_memory` > `T2_random_memory`
- `T1_relevant_memory` > `T4_matched_irrelevant_memory`
- `T1_relevant_memory` should not simply be shorter and unsolved
- `T1_relevant_memory` should not leak raw source facts

Green light only if the relevant-note arm improves no-peer/hard failures without leakage or prompt saturation.

### Phase 5: Optional P5 SDPO smoke with memory enabled

Only after Phases 1-4 pass, run a short fail-fast memory-enabled SDPO smoke. Do not start overnight training yet.

Use the current safe SDPO recipe and change only memory flags:

```bash
export SDPO_MEMORY_ENABLED=true
export SDPO_MEMORY_PATH="$NOTE_DIR/note_sdpo_memory_bank.jsonl"
export SDPO_MEMORY_MODE=relevant
export SDPO_MEMORY_INJECT_WHEN=no_solution
export SDPO_MEMORY_ALLOW_WITHOUT_FEEDBACK=false
export SDPO_FAIL_FAST_NONFINITE=1
export SDPO_EMA_FINITE_CHECK=1
export SDPO_LOGPROB_DIAGNOSTICS=1
export SDPO_CUDA_MEMORY_DIAGNOSTICS=1
export SDPO_DECISION_WEIGHTING_ENABLED=false
export SDPO_DECISION_WEIGHTING_APPLY_TO_LOSS=false
export TOTAL_TRAINING_STEPS=30
export TEST_FREQ=0
export SAVE_FREQ=0
```

Success criteria:

- finite actor loss
- finite grad norm
- finite teacher/student top-k mass
- finite EMA checks
- nonzero selected target tokens
- stable CUDA memory
- memory metrics show notes are used only on eligible failed/no-solution rows

## Constraints

- Do not integrate T3 into this branch. T3 remains a separate branch until Note-only and T3-only ablations are independently interpretable.
- Do not enable raw historical trajectories in live teacher path.
- Do not enable `SDPO_DECISION_WEIGHTING_APPLY_TO_LOSS=true`.
- Do not train from notes until the teacher-quality probe passes.
- Do not interpret W&B `Finished` alone as success. Cross-validate local nohup/Ray log, W&B config/history, and rollout JSONL samples.
- Do not claim Note-SDPO improves training until there is a memory-enabled P5 smoke and an ablation against random/matched-irrelevant notes.

## Model-Path Contract

For the current Note-SDPO smoke/probe:

- Rollout generator: current Tau3 SDPO actor/rollout stack from `MODEL_PATH`, normally the SFT Qwen3.5-4B VLM-format checkpoint served language-only.
- Updated model during training: actor model in VERL FSDP.
- Teacher for original SDPO: EMA reference teacher when `SDPO_TEACHER_BACKEND=ema_ref`.
- Teacher for note writing: separate offline writer endpoint or external writer output. This is not automatically the same as the SDPO EMA teacher unless explicitly served that way.
- Evaluation model: validation uses the trainer config and `VAL_N`; pass@k requires `VAL_N >= k`.
- Weight movement: actor weights update through VERL; EMA teacher updates through SDPO EMA update path; vLLM rollout weight sync remains the trainer's responsibility.

Proof artifacts to log:

- exact `MODEL_PATH`
- exact note writer model name
- exact `SDPO_MEMORY_PATH`
- W&B run URL
- nohup log path
- note bank manifest path
- retrieval audit summary path
- teacher probe output path

## QC And Merge Rules

Hard blockers:

- NaN/Inf
- OOM before expected horizon
- missing fail-fast diagnostics
- wrong recipe config
- note leakage
- prompt saturation
- same UID/task retrieval
- short unsolved answers being preferred
- critical tests skipped on P5

Majority vote rule:

- Six-agent vote may decide non-blocking tradeoffs.
- Any hard blocker is automatic STOP.

Minimum evidence bundle for a remote write-back:

- commit SHA
- test commands and pass/skip/fail summary
- note bank path and compile manifest
- retrieval audit summary
- teacher probe summary
- if smoke was run: W&B URL, nohup log path, final global step, and first/last relevant finite metrics

## Write Back

Update these repo-visible files after remote work:

- `research/session_sync.md`: append concise state update and decision.
- `research/kiro_note_sdpo_handoff.md`: update this handoff if commands or gates change.
- Optional: create `research/diagnostics/<timestamp>_note_sdpo_probe_summary.md` with links/paths to artifacts.

Do not commit bulky rollout archives, W&B folders, or full note-writer raw dumps unless explicitly requested. Prefer manifests and summaries.

## Stop If

Stop and report instead of branching if:

- core SDPO NaN/OOM/mask/EMA tests fail on P5
- the note bank compile rejects most notes
- retrieval audit finds leakage or same UID/task retrieval
- teacher probe shows relevant notes do not beat random/matched irrelevant notes
- notes mostly make outputs shorter but not solved
- memory-enabled 30-step smoke hits non-finite, OOM, prompt saturation, or zero selected target tokens
- the model path or rollout source cannot be verified
