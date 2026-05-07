# Teacher-Side Memory-SDPO Probe Plan

Status: first implementation slice added behind config flags.

## Goal

Test whether compact train-only correction memory improves the original SDPO teacher on peer-sparse Tau3 failures before spending a full P5 training run.

The deployed actor/student context is unchanged. Memory is privileged teacher context only:

```text
actor/student:
  same Tau3 prompt/history used by GRPO and original SDPO

teacher:
  same original SDPO reprompt
  + successful peer when available
  + compact train-only memory card when enabled and no successful peer exists

loss:
  same SDPO response-token distillation path
```

## Files

- `verl/utils/tau3_sdpo_memory.py` loads JSONL/JSON memory cards, strips raw thinking, and retrieves either relevant or random cards.
- `research/memory_cards/tau3_airline_seed_cards.jsonl` contains small hand-written train-only workflow cards.
- `scripts/tau3/build_sdpo_memory_teacher_probe.py` builds offline T0/T1/T2 teacher prompts from failed rollout JSONLs.
- `scripts/tau3/audit_sdpo_memory_retrieval.py` compares memory-unit and retrieval backends offline before any live Memory-SDPO run.
- `verl/trainer/config/tau3_sdpo_live.yaml` adds `tau3.sdpo.memory.*` config.
- `run_local_tau3_sdpo_live_p5.sh` forwards `SDPO_MEMORY_*` env vars into Hydra.

## Retrieval Audit

Before running Memory-SDPO on P5, audit whether retrieval itself is plausible.
This is not a teacher-quality score yet; it only answers what memory each backend
would select and how expensive the retrieval path is.

```bash
python scripts/tau3/audit_sdpo_memory_retrieval.py \
  --query-rollout /path/to/failed_or_mixed_rollouts.tgz \
  --memory-card research/memory_cards/tau3_airline_seed_cards.jsonl \
  --memory-rollout /path/to/successful_or_mixed_train_rollouts.tgz \
  --memory-unit cards \
  --memory-unit full_trajectory \
  --memory-unit event_chunk \
  --backend random \
  --backend lexical \
  --output-jsonl /mnt/sagemaker-nvme/tau3_sdpo/memory_probe/retrieval_audit.jsonl \
  --summary-json /mnt/sagemaker-nvme/tau3_sdpo/memory_probe/retrieval_audit.summary.json \
  --max-queries 200 \
  --top-k 3 \
  --block-same-task-id \
  --block-same-uid
```

Optional dense HF CPU probe:

```bash
python scripts/tau3/audit_sdpo_memory_retrieval.py \
  --query-rollout /path/to/failed_or_mixed_rollouts.tgz \
  --memory-rollout /path/to/successful_or_mixed_train_rollouts.tgz \
  --memory-unit full_trajectory \
  --memory-unit event_chunk \
  --backend lexical \
  --backend dense_hf \
  --hf-model BAAI/bge-small-en-v1.5 \
  --hf-device cpu \
  --embedding-cache /mnt/sagemaker-nvme/tau3_sdpo/memory_probe/embedding_cache.jsonl \
  --output-jsonl /mnt/sagemaker-nvme/tau3_sdpo/memory_probe/dense_hf_audit.jsonl
```

Optional Bedrock Titan V2 probe:

```bash
python scripts/tau3/audit_sdpo_memory_retrieval.py \
  --query-rollout /path/to/failed_or_mixed_rollouts.tgz \
  --memory-rollout /path/to/successful_or_mixed_train_rollouts.tgz \
  --memory-unit full_trajectory \
  --memory-unit event_chunk \
  --backend lexical \
  --backend bedrock_titan \
  --bedrock-region us-east-1 \
  --bedrock-model-id amazon.titan-embed-text-v2:0 \
  --bedrock-dimensions 1024 \
  --embedding-cache /mnt/sagemaker-nvme/tau3_sdpo/memory_probe/bedrock_embedding_cache.jsonl \
  --output-jsonl /mnt/sagemaker-nvme/tau3_sdpo/memory_probe/bedrock_audit.jsonl
```

Memory units to compare:

```text
cards: seed/debug decision cards, useful for sanity checks
full_trajectory: bounded successful trajectory precedent
event_chunk: coarse observable event chunks from successful trajectories
char_chunk: broad sliding-window chunks, no semantic hand-snipping
```

By default, rollout-derived retrieval text strips `<think>` to avoid raw
reasoning spam dominating the index. To test the user's raw-trajectory concern
directly, run the same audit with:

```bash
--keep-thinking-in-memory
```

and optionally:

```bash
--keep-thinking-in-query
```

Compare retrieval agreement, latency, and teacher-quality outputs before
deciding whether raw historical thinking should be part of trajectory memory.

Use dense retrieval as a probe first. Do not put dense retrieval in the live
training hot path until the audit shows acceptable query-embedding latency and
relevant memory beats random memory in the teacher-quality probe.

If lexical or dense retrieval collapses onto one generic trajectory/chunk, first
debug query construction rather than swapping in a larger embedding model. The
audit script now emits `query_preview`, `query_part_lengths`, `query_hash`, and
`query_mode`. Re-run the same source set with:

```bash
--query-mode feedback_only
--query-mode prompt_tail_only
--query-mode prompt_tail_no_schema
```

and compare top-1 concentration and score margins. To avoid generic opening
chunks dominating retrieval, use:

```bash
--exclude-generic-opening-memory
```

or an explicit regex filter such as:

```bash
--require-memory-regex "book_reservation|update_reservation|cancel_reservation|transfer_to_human_agents|search_direct_flight|search_onestop_flight"
```

## Offline Prompt Builder

Build prompt variants from pulled rollout JSONLs:

```bash
python scripts/tau3/build_sdpo_memory_teacher_probe.py \
  --rollout /path/to/rollout_data \
  --memory-path research/memory_cards/tau3_airline_seed_cards.jsonl \
  --output-jsonl /tmp/tau3_memory_teacher_probe_prompts.jsonl \
  --max-samples 100
```

Prompt arms:

```text
T0_original: original SDPO feedback context only
T1_relevant_memory: original SDPO + relevant compact memory card
T2_random_memory: original SDPO + random compact memory card
```

This script only builds probe prompts. It does not call the teacher model or
score outputs. Green-light a tiny Memory-SDPO run only after a separate model
scoring/manual-rubric step shows that T1 produces cleaner, more actionable
teacher outputs than T0 and T2 on hard workflow prefixes.

## Training Toggle

Enable the live teacher-side memory path:

```bash
export SDPO_MEMORY_ENABLED=true
export SDPO_MEMORY_PATH=research/memory_cards/tau3_airline_seed_cards.jsonl
export SDPO_MEMORY_MODE=relevant
export SDPO_MEMORY_INJECT_WHEN=no_solution
export SDPO_MEMORY_ALLOW_WITHOUT_FEEDBACK=false
export SDPO_ARM=original
bash run_local_tau3_sdpo_live_p5.sh "$TASK_PATH" memory_sdpo_probe json
```

Useful W&B metrics:

```text
self_distillation/memory_available_fraction
self_distillation/memory_eligible_fraction
self_distillation/memory_used_fraction
self_distillation/memory_random_used_fraction
self_distillation/memory_no_solution_used_fraction
self_distillation/memory_section_char_mean
self_distillation/memory_used_and_prompt_saturated_fraction
self_distillation/teacher_prompt_token_mean
self_distillation/teacher_prompt_saturation_fraction
```

## Controls

Run at least:

```text
A0: original SDPO, memory disabled
A1: original SDPO + relevant memory
A2: original SDPO + random memory
```

Later controls:

```text
A3: raw retrieved trajectory memory
A4: matched-token irrelevant card
A5: same task-cluster blocked retrieval
```

Do not claim memory helps unless relevant memory beats random memory on hard workflow clusters without increasing response clipping or teacher prompt saturation.
