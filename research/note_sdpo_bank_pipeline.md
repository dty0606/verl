# Note-SDPO Memory Bank Pipeline

This is the runnable pipeline for teacher-written decision-note memory. It is
designed to plug into the existing `tau3.sdpo.memory.*` teacher-only injection
path, not into actor inference.

## Design

The deterministic code does only three things:

- It selects train rollout rows, sanitizes them, and emits note-writer prompts.
- It validates teacher-written JSON notes for schema, length, provenance, and leakage.
- It compiles accepted notes into the same JSONL card format consumed by `verl/utils/tau3_sdpo_memory.py`.

The semantic lesson is written by the teacher/self-distilled model. This avoids
hard-coding Tau3-specific failure categories while still giving us a safe,
auditable memory object.

## Build Prompts

Use successful train rollouts only for the first bank.

```bash
cd ~/verl_tau3_sdpo_vllm20
source activate sdpo-vllm20-v1

export NVME_ROOT=/mnt/sagemaker-nvme/tau3_sdpo
mkdir -p "$NVME_ROOT/memory_banks/note_sdpo_v1"

python scripts/tau3/build_note_sdpo_bank.py emit-prompts \
  --rollout "$NVME_ROOT/memory_sources/grpo_vllm_v1_clean_300_rollouts.tgz" \
  --output-prompts "$NVME_ROOT/memory_banks/note_sdpo_v1/prompts.jsonl" \
  --manifest-json "$NVME_ROOT/memory_banks/note_sdpo_v1/emit_manifest.json" \
  --score-threshold 1 \
  --source-step-max 60 \
  --source-run-id grpo_vllm_v1_clean_300 \
  --max-source-rows 512 \
  --prompt-tail-chars 6000 \
  --output-excerpt-chars 2500 \
  --feedback-chars 2000 \
  --source-split train
```

## Write Notes With A Teacher

Option A: use any OpenAI-compatible local teacher endpoint.

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8000/v1

python scripts/tau3/build_note_sdpo_bank.py write-openai \
  --prompt-jsonl "$NVME_ROOT/memory_banks/note_sdpo_v1/prompts.jsonl" \
  --writer-output-jsonl "$NVME_ROOT/memory_banks/note_sdpo_v1/writer_outputs.jsonl" \
  --manifest-json "$NVME_ROOT/memory_banks/note_sdpo_v1/write_manifest.json" \
  --writer-model "$MODEL_PATH" \
  --temperature 0 \
  --retries 2 \
  --timeout 120
```

Option B: write `writer_outputs.jsonl` externally. Each row must contain
`prompt_id` and either `note`, `raw_response`, or direct note fields.

## Compile Bank

```bash
python scripts/tau3/build_note_sdpo_bank.py compile \
  --prompt-jsonl "$NVME_ROOT/memory_banks/note_sdpo_v1/prompts.jsonl" \
  --writer-output-jsonl "$NVME_ROOT/memory_banks/note_sdpo_v1/writer_outputs.jsonl" \
  --output-bank "$NVME_ROOT/memory_banks/note_sdpo_v1/note_sdpo_memory_bank.jsonl" \
  --rejections-jsonl "$NVME_ROOT/memory_banks/note_sdpo_v1/rejections.jsonl" \
  --manifest-json "$NVME_ROOT/memory_banks/note_sdpo_v1/compile_manifest.json" \
  --writer-model "ema_or_actor_teacher_step60" \
  --initial-status active \
  --merge-similar
```

The resulting `note_sdpo_memory_bank.jsonl` is ready for SDPO:

```bash
+tau3.sdpo.memory.enabled=true \
+tau3.sdpo.memory.path="$NVME_ROOT/memory_banks/note_sdpo_v1/note_sdpo_memory_bank.jsonl" \
+tau3.sdpo.memory.mode=relevant \
+tau3.sdpo.memory.inject_when=no_solution \
+tau3.sdpo.memory.max_card_chars=2200 \
+tau3.sdpo.memory.fail_on_error=true \
+tau3.sdpo.memory.allow_without_feedback=false
```

## Leakage Rules

The compiled memory bank must be train-only and must not include:

- Raw `<think>` text.
- Raw tool JSON or tool-call traces.
- Names, IDs, reservation numbers, payment IDs, emails, dates, prices, flight numbers, or hidden task IDs.
- Eval/test task-derived source rows.

The prompt JSONL contains local safety metadata used for validation. Treat it as
train-only internal artifact. The deployable memory bank is the compiled JSONL.

## QC

Local checks:

```bash
python -m py_compile scripts/tau3/build_note_sdpo_bank.py
pytest -q tests/utils/test_tau3_note_sdpo_bank_builder.py
```

Real-data smoke already verified prompt emission on
`research/diagnostics/sdpo_vanilla_peer_full.zip`.
