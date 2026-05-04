# P5 Recipes: Latest VERL tau3 Fork

Date: 2026-05-02

Old recipe book (historical reference only):

- `dty0606_SDPO/research/P5_RECIPES.md`

Do not copy old commands without checking paths, S3 prefixes, model format,
LoRA assumptions, and old VERL config keys. The old path is deprecated.

---

## Environment

- P5: SageMaker CE `p5.48xlarge`, 8× H100 80GB
- Airline rewards: DB-state based, no LLM judge needed
- Retail rewards: NL assertion, needs LLM judge (`TAU3_NL_ASSERTION_MODEL`)
- Bedrock throttling: 28 parallel shards safe, 60+ risky
- Thinking generation: `MAX_RESPONSE_TOKENS=8192`
- Canonical split: 30 train / 20 test from `datasets/tau3_live_airline_canonical_split.json`
- Tasks 7, 39: historically unsolvable by Opus generator, exclude from success expectations
- P5 env: `sdpo-qwen35` conda, torch 2.10.0+cu128, transformers 5.6.2, vllm 0.19.1, verl 0.8.0.dev0, liger-kernel 0.7.0, flash-attn 2.8.3 (prebuilt wheel for torch 2.10+cu128)

## Remote Layout

```bash
LATEST_VERL_ROOT=~/verl_tau3_sdpo    # active execution repo
OLD_SDPO_ARCHIVE=~/SDPO-qwen35       # old archive, read-only reference
```

## S3 Sync

```bash
export S3_PREFIX=s3://tianyd-rlvr-research/tau3-sdpo/latest-verl

# Code: Kiro → S3 → P5
aws s3 sync verl_tau3_sdpo/ "$S3_PREFIX/repo/" --exclude ".git/*" --exclude "*.pyc" --exclude "datasets/*"
# On P5:
aws s3 sync "$S3_PREFIX/repo/" ~/verl_tau3_sdpo/ --exclude ".git/*"

# Checkpoints from P5 (only when needed for cross-machine access)
aws s3 sync ~/verl_tau3_sdpo/checkpoints/ "$S3_PREFIX/checkpoints/" --exclude "*/optim*"
```

Keep generated trajectories, checkpoints, rollout data, wandb artifacts,
and large logs in S3 or P5-local. Do not commit them to GitHub.

---

## SFT Format Decision: Full-Trajectory Smoke Lane

Current priority: restore the strongest known SFT contract from the old LoRA
run: **one full successful trajectory per row, thinking-on, assistant-only loss
across all assistant turns**.

Why this changed from the 5K turn-row pilot:

- The turn-row VLM SFT pipeline worked mechanically, but paired eval regressed
  on policy/endpoint decisions such as task 30 and task 37.
- The old LoRA baseline saw full trajectories with historical assistant
  thinking and tool history, and behaved better on the same kind of decisions.
- The likely issue is context/template mismatch, not generic SFT convergence.

The full-trajectory path must use a patched Qwen3.5 chat template that preserves
historical assistant `<think>` blocks. The stock Qwen3.5 thinking template can
strip old reasoning in multi-turn renders.

Historical turn-row format remains useful for ablations:

```text
messages = prior conversation history (thinking stripped from historical assistants)
answer   = current assistant target with thinking/content/tool_calls
```

Dataset class for training is still `PretokenizedSFTDataset`; the difference is
which offline pre-tokenizer produced `input_ids` and `loss_mask`.

<!-- Historical turn-row note retained below for audit context:
This is required because Qwen3.5 strips historical assistant reasoning during
full multi-turn chat-template rendering. Each row trains one assistant action
given history — matching rollout inference.

Dataset class: `TurnSFTDataset` via `+data.custom_cls` hydra override.
-->

---

## Recipe 0: Environment Truth

Run before any training. Record output in `research/session_sync.md`.

```bash
cd ~/verl_tau3_sdpo
echo "=== Git ===" && git rev-parse HEAD && git status --short
echo "=== Packages ==="
python3 - <<'PY'
import importlib.metadata as md
for pkg in ["torch", "transformers", "vllm", "verl", "liger-kernel",
            "flash-attn", "datasets", "gymnasium"]:
    try:
        print(f"{pkg:20s} {md.version(pkg)}")
    except Exception as exc:
        print(f"{pkg:20s} UNKNOWN ({exc})")
PY
echo "=== GPU ===" && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

---

## Recipe 1: Build Turn-Per-Row SFT Dataset

Build from all successful thinking-on trajectories.

```bash
cd ~/verl_tau3_sdpo

python3 scripts/tau3/build_protocol_sft_data.py \
  --input ~/SDPO-qwen35/outputs/tau3_protocol_candidates_thinking_5k_b2 \
  --input ~/SDPO-qwen35/outputs/tau3_protocol_candidates_thinking_5k \
  --input ~/SDPO-qwen35/outputs/tau3_protocol_candidates_thinking_v1 \
  --output-dir datasets/tau3_sft_thinking_train_only \
  --train-only-holdout \
  --include-thinking-traces \
  --sft-format turn \
  --overwrite
```

Verify:

```bash
python3 -c "
import json
m = json.load(open('datasets/tau3_sft_thinking_train_only/manifest.json'))
print(f'accepted: {m[\"accepted_rows\"]}  rejected: {m[\"rejected_rows\"]}')
print(f'train: {m[\"train_rows\"]}  test: {m[\"test_rows\"]}')
print(f'format: {m[\"sft_format\"]}')
print(f'rejected_reasons: {m[\"rejected_reasons\"]}')
"
```

Expected: ~82K accepted turn-rows from ~9K successful trajectories, ~7K rejected (failed trajectories).

---

## Recipe 1F: Build Full-Trajectory SFT Smoke Dataset

Use this before any new real SFT run. It creates one row per accepted successful
trajectory, preserving all stitched assistant thinking/tool history.

```bash
cd ~/verl_tau3_sdpo

python3 scripts/tau3/build_protocol_sft_data.py \
  --input ~/SDPO-qwen35/outputs/tau3_protocol_candidates_thinking_5k_b2 \
  --input ~/SDPO-qwen35/outputs/tau3_protocol_candidates_thinking_5k \
  --input ~/SDPO-qwen35/outputs/tau3_protocol_candidates_thinking_v1 \
  --output-dir datasets/tau3_sft_full_traj_smoke \
  --train-only-holdout \
  --include-thinking-traces \
  --sft-format trajectory \
  --max-rows-per-task 20 \
  --overwrite
```

Verify:

```bash
python3 - <<'PY'
import json
m = json.load(open("datasets/tau3_sft_full_traj_smoke/manifest.json"))
print("format", m["sft_format"], "train", m["train_rows"], "test", m["test_rows"])
print("thinking modes", m["thinking_supervision_modes"])
assert m["sft_format"] == "trajectory"
assert m["train_rows"] > 0 and m["test_rows"] > 0
assert m["accepted_rows_containing_reasoning_traces"] == m["accepted_rows"]
print("PASS: full-traj smoke parquet built with thinking traces")
PY
```

---

## Recipe 2F: Pre-Tokenize Full Trajectory With Patched Template

This applies the local preserve-thinking Qwen3.5 chat template and writes
`input_ids`/`loss_mask` for `PretokenizedSFTDataset`.

```bash
cd ~/verl_tau3_sdpo

python3 scripts/tau3/pretokenize_full_traj_sft.py \
  --input datasets/tau3_sft_full_traj_smoke \
  --output datasets/tau3_sft_full_traj_smoke_pretok \
  --model Qwen/Qwen3.5-4B \
  --max-length 32768 \
  --truncation error \
  --workers 8 \
  --audit-samples 5 \
  2>&1 | tee logs/pretokenize_full_traj_sft_smoke.log
```

If overlength rows make the smoke fail, rerun only for debugging with
`--allow-errors` and inspect `manifest.json`. Do not use silent truncation for a
claim run until the row-length distribution is understood.

Pass criteria:

- `errors == 0` for both train/test in the first clean smoke, or a clearly
  documented small overlength-only error count in a debugging run.
- `avg_labeled_tokens > 0`.
- `avg_assistant_count > 1`, proving this is actually full trajectory.
- `*.audit.jsonl` decoded spans include old `<think>` and tool names.
- Each pre-tokenized row includes `segments` metadata with role/type,
  token boundaries, assistant-turn index, loss flag, and tool names. This is
  not consumed by SFT training yet, but it is the component-level index we can
  reuse later for memory/retrieval experiments.

---

## Recipe 3F: Full-Trajectory Token Budget Audit

Run this before choosing RL `MAX_PROMPT_LENGTH` / `MAX_MODEL_LEN`. It measures
the exact token budget from the pre-tokenized `segments` metadata and estimates
how assistant tokens split across historical thinking, tool calls, and visible
assistant text.

```bash
cd ~/verl_tau3_sdpo

python3 scripts/tau3/analyze_full_traj_token_budget.py \
  --dataset datasets/tau3_sft_full_traj_smoke_pretok \
  --split both \
  --model Qwen/Qwen3.5-4B \
  --component-sample-rows 500 \
  --thresholds 4096 8192 16384 24576 32768 \
  --output-json research/diagnostics/full_traj_token_budget_smoke.json \
  2>&1 | tee logs/full_traj_token_budget_smoke.log
```

For the real full-trajectory dataset, point `--dataset` at the production
pre-tokenized directory and raise `--component-sample-rows` only if the decoded
component estimate is still fast enough. Exact role/turn stats always cover the
entire split.

Decision use:

- If `assistant_prompt_len_before_turn.p90` is already near 16K, the one-step
  GRPO smoke should start at `MAX_PROMPT_LENGTH=16384`, not 4096.
- If many rows exceed 24K total context, either reduce response length, use a
  smaller rollout batch, or build a shorter/easier RL smoke subset before a
  full baseline.
- The component estimate is approximate for `<think>` vs tool-call vs visible
  text, but exact for role/turn boundaries.

---

## Recipe 4F: Ten-Step Full-Trajectory SFT Smoke

```bash
cd ~/verl_tau3_sdpo

REPORT_TO='["console"]' \
MODEL_PATH=Qwen/Qwen3.5-4B \
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' \
NUM_GPUS=8 TOTAL_EPOCHS=1 SAVE_FREQ=10 TEST_FREQ=10 \
MAX_LENGTH=32768 MAX_TOKEN_LEN_PER_GPU=32768 \
TRAIN_BATCH_SIZE=8 MICRO_BATCH_SIZE_PER_GPU=1 \
USE_LIGER=true LR=1e-5 TRUNCATION=error \
PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/tau3/run_tau3_verl_sft_full_thinking.sh \
  datasets/tau3_sft_full_traj_smoke_pretok \
  qwen35_4b_vlm_full_traj_sft_10step_smoke \
  +data.custom_cls.path=verl/utils/dataset/pretokenized_sft_dataset.py \
  +data.custom_cls.name=PretokenizedSFTDataset \
  engine.use_torch_compile=False \
  model.use_fused_kernels=False \
  trainer.total_training_steps=10 \
  trainer.max_ckpt_to_keep=1 \
  2>&1 | tee logs/sft_full_traj_10step_smoke.log
```

After checkpoint export, patch the saved tokenizer to the same template:

```bash
export HF_CKPT=$(ls -d checkpoints/SDPO/tau3_verl_sft/*full_traj_sft_10step_smoke*/global_step_*/huggingface | tail -1)
python3 scripts/qwen35/patch_chat_template_preserve_thinking.py "$HF_CKPT"
```

Then run the normal VLM-format check, vLLM smoke, and one-step GRPO smoke. Stop
if the patched checkpoint cannot load or if the rollout prompt drops historical
assistant `<think>`.

---

## Recipe 5F: Real Full-Trajectory SFT (uncapped, 1 epoch, with timing probe)

After Recipe 4F proves the 10-step path is stable end-to-end, run the real
full-param SFT on all accepted successful trajectories. Do not reopen LoRA, do
not switch back to text-only CausalLM checkpoints.

### 5F-A. Build uncapped full-trajectory dataset

Same as Recipe 1F but without `--max-rows-per-task` so every accepted
trajectory becomes one SFT row.

```bash
cd ~/verl_tau3_sdpo

python3 scripts/tau3/build_protocol_sft_data.py \
  --input ~/SDPO-qwen35/outputs/tau3_protocol_candidates_thinking_5k_b2 \
  --input ~/SDPO-qwen35/outputs/tau3_protocol_candidates_thinking_5k \
  --input ~/SDPO-qwen35/outputs/tau3_protocol_candidates_thinking_v1 \
  --output-dir datasets/tau3_sft_full_traj \
  --train-only-holdout \
  --include-thinking-traces \
  --sft-format trajectory \
  --overwrite
```

Verify:

```bash
python3 - <<'PY'
import json
m = json.load(open("datasets/tau3_sft_full_traj/manifest.json"))
print("train", m["train_rows"], "test", m["test_rows"])
assert m["sft_format"] == "trajectory"
assert m["accepted_rows_containing_reasoning_traces"] == m["accepted_rows"]
# Expect roughly the number of successful trajectories from the generator pools
assert m["train_rows"] >= 4000, f"unexpectedly small train set: {m['train_rows']}"
print("PASS: full-traj uncapped dataset built")
PY
```

Expected: ~8–9K train rows, ~1K test rows based on the ~9,198 successful
thinking-on trajectories recorded in `session_sync.md`. If the count is much
lower, re-check which generator pools are being passed in.

### 5F-B. Pre-tokenize at full length

Same as Recipe 2F, pointed at the uncapped dataset.

```bash
python3 scripts/tau3/pretokenize_full_traj_sft.py \
  --input datasets/tau3_sft_full_traj \
  --output datasets/tau3_sft_full_traj_pretok \
  --model Qwen/Qwen3.5-4B \
  --max-length 32768 \
  --truncation error \
  --workers 8 \
  --audit-samples 10 \
  2>&1 | tee logs/pretokenize_full_traj_sft.log
```

Pass criteria: `errors == 0`, `avg_labeled_tokens > 0`, `avg_assistant_count > 1`,
`*.audit.jsonl` decoded spans include old `<think>` and tool names.

### 5F-C. Run Recipe 3F on the uncapped dataset

Before training, measure the exact token budget so we know the P90/P95 tail.

```bash
python3 scripts/tau3/analyze_full_traj_token_budget.py \
  --dataset datasets/tau3_sft_full_traj_pretok \
  --split both \
  --model Qwen/Qwen3.5-4B \
  --component-sample-rows 1000 \
  --thresholds 4096 8192 16384 24576 32768 \
  --output-json research/diagnostics/full_traj_token_budget_real.json \
  2>&1 | tee logs/full_traj_token_budget_real.log
```

### 5F-D. Timing probe (20 steps, no epoch)

Before committing to a full epoch, run 20 steps to check step time at the
target batch/length. This is throw-away; no checkpoint export required.

```bash
REPORT_TO='["console"]' \
MODEL_PATH=Qwen/Qwen3.5-4B \
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra"]' \
NUM_GPUS=8 TOTAL_EPOCHS=1 SAVE_FREQ=20 TEST_FREQ=1000 \
MAX_LENGTH=32768 MAX_TOKEN_LEN_PER_GPU=32768 \
TRAIN_BATCH_SIZE=8 MICRO_BATCH_SIZE_PER_GPU=1 \
USE_LIGER=true LR=1e-5 TRUNCATION=error \
PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/tau3/run_tau3_verl_sft_full_thinking.sh \
  datasets/tau3_sft_full_traj_pretok \
  qwen35_4b_vlm_full_traj_sft_timing_probe \
  +data.custom_cls.path=verl/utils/dataset/pretokenized_sft_dataset.py \
  +data.custom_cls.name=PretokenizedSFTDataset \
  engine.use_torch_compile=False \
  model.use_fused_kernels=False \
  trainer.total_training_steps=20 \
  trainer.max_ckpt_to_keep=1 \
  2>&1 | tee logs/sft_full_traj_timing_probe.log
```

Decision:

- If average step time is **under ~120s**, proceed to full epoch (5F-E).
  Ballpark full-epoch wall clock at `train_batch=8`, 9K rows: ~9000/8=1125
  steps × 120s ≈ **37 hours** single pass.
- If step time is **120–180s**, still proceed but prepare a LIMA-style fallback:
  stratified subsample to ~5K trajectories × 2 epochs (same token budget, half
  the wall clock).
- If step time **> 180s**, stop and diagnose:
  - Confirm `use_liger=True` in the log.
  - Check if `MAX_LENGTH=32768` is forced vs dynamic bucketing.
  - Consider `MICRO_BATCH_SIZE_PER_GPU=2` if memory allows.
  - Fall back to stratified 5K × 2 epochs subsample.

Do not skip this probe. The 10-step smoke used ~500 rows; step timing can shift
when the dataset grows by 20×.

### 5F-E. Full real SFT (1 epoch, uncapped)

```bash
REPORT_TO='["console","wandb"]' \
MODEL_PATH=Qwen/Qwen3.5-4B \
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' \
NUM_GPUS=8 TOTAL_EPOCHS=1 SAVE_FREQ=500 TEST_FREQ=500 \
MAX_LENGTH=32768 MAX_TOKEN_LEN_PER_GPU=32768 \
TRAIN_BATCH_SIZE=8 MICRO_BATCH_SIZE_PER_GPU=1 \
USE_LIGER=true LR=1e-5 TRUNCATION=error \
PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/tau3/run_tau3_verl_sft_full_thinking.sh \
  datasets/tau3_sft_full_traj_pretok \
  qwen35_4b_vlm_full_traj_sft_real_9k \
  +data.custom_cls.path=verl/utils/dataset/pretokenized_sft_dataset.py \
  +data.custom_cls.name=PretokenizedSFTDataset \
  engine.use_torch_compile=False \
  model.use_fused_kernels=False \
  trainer.max_ckpt_to_keep=3 \
  2>&1 | tee logs/sft_full_traj_real_9k.log
```

After the final export, patch the tokenizer:

```bash
export HF_CKPT=$(ls -d checkpoints/SDPO/tau3_verl_sft/*full_traj_sft_real_9k*/global_step_*/huggingface | tail -1)
python3 scripts/qwen35/patch_chat_template_preserve_thinking.py "$HF_CKPT"
echo "REAL_SFT_CKPT=$HF_CKPT"
```

Record `REAL_SFT_CKPT` in `research/session_sync.md`. Use this exact path for
vLLM serve smoke, one-step GRPO smoke, one-step SDPO smoke, and the full
Recipe 8 / Recipe 10 baselines.

### 5F fallback: LIMA-style 5K × 2 epochs

If the timing probe forces a subsample, build a stratified 5K subset that
preserves task balance.

```bash
python3 scripts/tau3/curate_full_traj_subset.py \
  --input datasets/tau3_sft_full_traj_pretok \
  --output datasets/tau3_sft_full_traj_5k \
  --target-rows 5000 \
  --strata task_id \
  --seed 0 \
  --overwrite
```

Then run 5F-E pointing at `datasets/tau3_sft_full_traj_5k` with
`TOTAL_EPOCHS=2`. Total token budget is identical to the 9K × 1 epoch case.

Note: `curate_full_traj_subset.py` does not exist yet. Only write it if the
timing probe forces this fallback.

---

## Recipe 2: Audit Qwen3.5 Template

Run before any SFT training. Validates that turn-per-row data renders correctly
through Qwen3.5's chat template with proper loss masking.

```bash
python3 scripts/qwen35/diagnose_tau3_sft_template.py \
  datasets/tau3_sft_thinking_train_only \
  --model Qwen/Qwen3.5-4B \
  --split train \
  --max-rows 20 \
  --max-length 32768 \
  --format turn \
  2>&1 | tee logs/qwen35_turn_sft_template_audit.log
```

Pass criteria per row:
- Exactly 1 contiguous target span
- `<think>` text present in decoded target (for thinking turns)
- Tool names present in decoded target (for tool-call turns)
- Non-empty loss mask

**Stop if audit fails.**

---

## Recipe 3: Pre-Tokenize Turn SFT Data

Run after audit passes. First curate a balanced raw turn-row subset; then move
Qwen3.5 chat-template work out of the training loop. Use left truncation if any
row exceeds `MAX_LENGTH`, because the current assistant answer is at the end of
the sequence.

### 3A. Curate a Balanced 5K Pilot

This is the current preferred SFT baseline. The full 75K turn-row dataset is
valid, but too expensive for the first VLM-format VERL SFT pass. Sample evenly
from canonical train tasks and exclude task 7, which has known assertion issues.

```bash
cd ~/verl_tau3_sdpo

python3 scripts/tau3/curate_turn_sft_subset.py \
  --input datasets/tau3_sft_thinking_train_only \
  --output datasets/tau3_sft_thinking_balanced_5k \
  --deny-task-ids 7 \
  --train-rows-per-task 170 \
  --val-rows-per-task 20 \
  --max-id-variants-per-source-turn 1 \
  --require-final-reward 1.0 \
  --require-thinking \
  --overwrite \
  2>&1 | tee logs/curate_turn_sft_balanced_5k.log
```

Expected if only task 7 is denied: 29 usable tasks, about 4,930 train rows and
580 validation rows. If task 39 is re-confirmed assertion-bad or unsolvable for
the generator, rerun with `--deny-task-ids 7 39` and expect about 5,040 train
rows with `--train-rows-per-task 180`.

Verify:

```bash
python3 - <<'PY'
import json
m = json.load(open("datasets/tau3_sft_thinking_balanced_5k/manifest.json"))
print("train_rows", m["train_rows"], "test_rows", m["test_rows"])
print("denied", m["deny_task_ids"])
print("shortfall_tasks", m["shortfall_tasks"])
print("train_rows_per_task", m["train_rows_per_task"])
assert "7" not in m["train_rows_per_task"], m["train_rows_per_task"]
assert not m["shortfall_tasks"], m["shortfall_tasks"]
assert min(m["train_rows_per_task"].values()) >= 150, m["train_rows_per_task"]
print("PASS: balanced SFT pilot manifest is sane")
PY
```

Stop if task 7 appears, if many tasks have shortfalls, or if tool/final-text
strata look collapsed in the manifest.

### 3B. Pre-Tokenize the Balanced Pilot

```bash
cd ~/verl_tau3_sdpo

python3 scripts/tau3/pretokenize_turn_sft.py \
  --input datasets/tau3_sft_thinking_balanced_5k \
  --output datasets/tau3_sft_balanced_5k_pretokenized \
  --model Qwen/Qwen3.5-4B \
  --max-length 24576 \
  --truncation left \
  --workers 8 \
  2>&1 | tee logs/pretokenize_turn_sft_balanced_5k.log
```

Verify the manifest before training:

```bash
python3 - <<'PY'
import json
manifest = json.load(open("datasets/tau3_sft_balanced_5k_pretokenized/manifest.json"))
for split in ["train", "test"]:
    stats = manifest.get(split, {})
    print(split, stats)
    assert stats.get("errors", 0) == 0, stats
    assert stats.get("output_rows", 0) > 0, stats
    assert stats.get("avg_labeled_tokens", 0) > 0, stats
print("PASS: pre-tokenized dataset manifest is trainable")
PY
```

**Stop if pre-tokenization reports errors or empty labeled-token stats.**

The training-step/runtime estimate is in Recipe 5 below.

---

## Recipe 4: Two-Step Pre-Tokenized SFT Smoke

This proves the data loader, Liger path, VLM checkpoint export, and dynamic
batching without spending an epoch.

```bash
cd ~/verl_tau3_sdpo

REPORT_TO='["console"]' \
MODEL_PATH=Qwen/Qwen3.5-4B \
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' \
NUM_GPUS=8 TOTAL_EPOCHS=1 SAVE_FREQ=1 TEST_FREQ=1 \
MAX_LENGTH=24576 MAX_TOKEN_LEN_PER_GPU=24576 \
TRAIN_BATCH_SIZE=32 MICRO_BATCH_SIZE_PER_GPU=1 \
USE_LIGER=true LR=1e-5 TRUNCATION=error \
PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/tau3/run_tau3_verl_sft_full_thinking.sh \
  datasets/tau3_sft_balanced_5k_pretokenized \
  qwen35_4b_vlm_export_smoke_balanced5k \
  data.custom_cls.path=verl/utils/dataset/pretokenized_sft_dataset.py \
  data.custom_cls.name=PretokenizedSFTDataset \
  engine.use_torch_compile=False \
  model.use_fused_kernels=False \
  trainer.total_training_steps=2 \
  trainer.max_ckpt_to_keep=1 \
  2>&1 | tee logs/sft_vlm_export_smoke_balanced5k.log
```

Pass criteria:
- Step time should be close to the known VERL baseline, not the 75K all-row run
- Loss is finite and nonzero
- A `huggingface/` checkpoint is saved

**Stop if loss/mask path fails or VLM-format checkpoint export is missing.**

---

## Recipe 5: Balanced Pre-Tokenized SFT Run

Train the balanced 5K pilot first. Use the full 75K row dataset only after the
VLM checkpoint -> vLLM -> GRPO path is proven.

```bash
cd ~/verl_tau3_sdpo

MODEL_PATH=Qwen/Qwen3.5-4B \
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' \
NUM_GPUS=8 TOTAL_EPOCHS=1 SAVE_FREQ=100 TEST_FREQ=100 \
MAX_LENGTH=24576 MAX_TOKEN_LEN_PER_GPU=24576 \
TRAIN_BATCH_SIZE=32 MICRO_BATCH_SIZE_PER_GPU=1 \
USE_LIGER=true LR=1e-5 TRUNCATION=error \
PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/tau3/run_tau3_verl_sft_full_thinking.sh \
  datasets/tau3_sft_balanced_5k_pretokenized \
  qwen35_4b_vlm_sft_balanced5k_turn_pretok \
  data.custom_cls.path=verl/utils/dataset/pretokenized_sft_dataset.py \
  data.custom_cls.name=PretokenizedSFTDataset \
  engine.use_torch_compile=False \
  model.use_fused_kernels=False \
  trainer.max_ckpt_to_keep=2 \
  2>&1 | tee logs/sft_balanced5k_turn_pretok.log
```

Expected: about 4,930 train rows / batch 32 = about 155 optimizer steps.
At the observed latest-VERL baseline of roughly 80s/step, this is about
3.5 hours. That is acceptable for the first VLM-format SFT baseline.

If OOM or too slow:
1. Confirm Liger is active in the log (`use_liger=True`)
2. Try `TRAIN_BATCH_SIZE=16`
3. Try `MAX_LENGTH=16384`, then re-run pre-tokenization

<!-- legacy estimate superseded by Recipe 5 above

Expected: ~2,374 steps, ~1-2 hours on 8×H100.

-->

### Verify VLM-format checkpoint

```bash
export HF_CKPT=$(ls -d checkpoints/SDPO/tau3_verl_sft/*/global_step_*/huggingface | tail -1)
echo "Checkpoint: $HF_CKPT"

python3 - <<'PY'
import json, os
p = os.environ["HF_CKPT"] + "/config.json"
c = json.load(open(p))
print("model_type:", c.get("model_type"))
print("architectures:", c.get("architectures"))
assert c.get("model_type") == "qwen3_5", f"FAIL: model_type={c.get('model_type')}"
assert "Qwen3_5ForConditionalGeneration" in (c.get("architectures") or []), \
    f"FAIL: architectures={c.get('architectures')}"
print("PASS: VLM-format checkpoint confirmed")
PY
```

**Stop if checkpoint is text-only format.**

---

## Recipe 6: Standalone vLLM Serve Smoke

Prove the SFT export loads in vLLM before involving GRPO.

```bash
export HF_CKPT=$(ls -d checkpoints/SDPO/tau3_verl_sft/*/global_step_*/huggingface | tail -1)

python3 -m vllm.entrypoints.openai.api_server \
  --model "$HF_CKPT" \
  --served-model-name tau3-sft-smoke \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --language-model-only \
  --trust-remote-code \
  --port 9100 \
  > logs/vllm_serve_smoke.log 2>&1 &
VLLM_PID=$!

# Wait for ready
for i in $(seq 1 60); do
  curl -s http://127.0.0.1:9100/health > /dev/null 2>&1 && echo "vLLM ready" && break
  sleep 2
done

# Test
curl -s http://127.0.0.1:9100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tau3-sft-smoke",
    "messages": [{"role": "user", "content": "Say one short sentence about airline customer service."}],
    "temperature": 0.2,
    "max_tokens": 80
  }' | python3 -m json.tool

kill $VLLM_PID 2>/dev/null
```

Pass: coherent English, no gibberish, no vision-encoder crash.

---

## Recipe 7: One-Step GRPO Smoke

Prove latest VERL can load the SFT checkpoint, run tau3 live interaction,
and produce one nonzero actor update.

```bash
cd ~/verl_tau3_sdpo

# Build GRPO dataset if not done
python3 examples/data_preprocess/tau3_live_multiturn.py \
  --output-dir datasets/tau3_live_airline_canonical_json \
  --domain airline \
  --feedback-mode json

export HF_CKPT=$(ls -d checkpoints/SDPO/tau3_verl_sft/*/global_step_*/huggingface | tail -1)
export TAU3_LIVE_USER_MODEL=us.anthropic.claude-sonnet-4-6
export TAU3_LIVE_RUNTIME=official_gym
export MODEL_PATH="$HF_CKPT"
export ENABLE_THINKING=true
export VLLM_LANGUAGE_MODEL_ONLY=true
export N_GPUS_PER_NODE=8
export ROLLOUT_TP_SIZE=1
export TRAIN_BATCH_SIZE=4
export ROLLOUT_BATCH_SIZE=2
export PPO_MINI_BATCH_SIZE=4
export VAL_N=1
export TOTAL_TRAINING_STEPS=1
export TOTAL_EPOCHS=1
export TEST_FREQ=1
export SAVE_FREQ=1
export LR=1e-6
export MAX_PROMPT_LENGTH=16384
export MAX_RESPONSE_LENGTH=512
export MAX_MODEL_LEN=24576
export ROLLOUT_TEMPERATURE=0.2
export ROLLOUT_TOP_P=0.95
export VAL_TEMPERATURE=0.2
export VAL_TOP_P=0.95
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs outputs/grpo_vlm_sft_smoke/rollout_data

ROLLOUT_DATA_DIR=outputs/grpo_vlm_sft_smoke/rollout_data \
bash run_local_tau3_grpo_live_p5.sh \
  datasets/tau3_live_airline_canonical_json \
  grpo_vlm_sft_smoke \
  json \
  2>&1 | tee logs/grpo_vlm_sft_smoke.log
```

Pass criteria:
- Decoded assistant output is not gibberish
- `tau3_live_result` appears in rollout/reward fields
- Finite `actor/pg_loss` / `actor/grad_norm`; nonzero update is expected only if the rollout group has reward variance

**Stop if decoded output is corrupted.**

---

## Recipe 8: Full GRPO Baseline

Run only after SFT checkpoint passes vLLM serve + one-step GRPO smoke.
This is the SDPO-paper companion GRPO baseline, not a separate DeepSeekMath-reference
GRPO ablation: rollout `n=8`, rollout IS clip `2`, and KL coefficient `0.0`.
The response/model length budget is larger than the paper's single-shot code setup
because Tau3 airline rollouts are multi-turn.

```bash
cd ~/verl_tau3_sdpo

export HF_CKPT=<path to final SFT global_step_*/huggingface>
export TAU3_LIVE_USER_MODEL=us.anthropic.claude-sonnet-4-6
export TAU3_LIVE_RUNTIME=official_gym
export MODEL_PATH="$HF_CKPT"
export ENABLE_THINKING=true
export VLLM_LANGUAGE_MODEL_ONLY=true
export N_GPUS_PER_NODE=8
export ROLLOUT_TP_SIZE=1
export TRAIN_BATCH_SIZE=8
export ROLLOUT_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export VAL_N=4
export TOTAL_TRAINING_STEPS=300
export TOTAL_EPOCHS=300
export TEST_FREQ=30
export SAVE_FREQ=30
export LR=1e-6
export LR_WARMUP_STEPS=0
export MAX_PROMPT_LENGTH=16384
export MAX_RESPONSE_LENGTH=12288
export MAX_MODEL_LEN=32768
export ROLLOUT_TEMPERATURE=0.4
export ROLLOUT_TOP_P=0.95
export VAL_TEMPERATURE=0.4
export VAL_TOP_P=0.95
export PYTORCH_ALLOC_CONF=expandable_segments:True

ROLLOUT_DATA_DIR=outputs/grpo_full_sft_baseline/rollout_data \
bash run_local_tau3_grpo_live_p5.sh \
  datasets/tau3_live_airline_canonical_json \
  grpo_full_sft_baseline \
  json \
  2>&1 | tee logs/grpo_full_sft_baseline.log
```

No LoRA. No `SFT_LORA_*` env vars. Full checkpoint loaded directly.

---

## Recipe 9: One-Step Vanilla SDPO Smoke

Run only after the SFT checkpoint passes vLLM serve and the one-step GRPO smoke.
This is vanilla SDPO only: no memory/DENSE retrieval.

```bash
cd ~/verl_tau3_sdpo

export HF_CKPT=$(ls -d checkpoints/SDPO/tau3_verl_sft/*/global_step_*/huggingface | tail -1)
export TAU3_LIVE_USER_MODEL=us.anthropic.claude-sonnet-4-6
export TAU3_LIVE_RUNTIME=official_gym
export MODEL_PATH="$HF_CKPT"
export ENABLE_THINKING=true
export VLLM_LANGUAGE_MODEL_ONLY=true
export N_GPUS_PER_NODE=8
export ROLLOUT_TP_SIZE=1
export TRAIN_BATCH_SIZE=4
export ROLLOUT_BATCH_SIZE=2
export PPO_MINI_BATCH_SIZE=4
export VAL_N=1
export TOTAL_TRAINING_STEPS=1
export TOTAL_EPOCHS=1
export TEST_FREQ=1
export SAVE_FREQ=1
export LR=1e-6
export MAX_PROMPT_LENGTH=16384
export MAX_RESPONSE_LENGTH=512
export MAX_MODEL_LEN=24576
export SDPO_MAX_REPROMPT_LEN=8192
export SDPO_REPROMPT_TRUNCATION=right
export ROLLOUT_TEMPERATURE=0.2
export ROLLOUT_TOP_P=0.95
export VAL_TEMPERATURE=0.2
export VAL_TOP_P=0.95
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs outputs/sdpo_vlm_sft_smoke/rollout_data

ROLLOUT_DATA_DIR=outputs/sdpo_vlm_sft_smoke/rollout_data \
bash run_local_tau3_sdpo_live_p5.sh \
  datasets/tau3_live_airline_canonical_json \
  sdpo_vlm_sft_smoke \
  json \
  2>&1 | tee logs/sdpo_vlm_sft_smoke.log
```

Pass criteria:
- Decoded assistant output is not gibberish
- `tau3_live_result` and `feedback` appear in rollout/reward fields for failed samples
- `self_distillation/reprompt_sample_fraction` is nonzero if failed samples have feedback
- `actor/pg_loss`, `self_distillation/token_fraction`, and `actor/grad_norm` are finite

**Stop if `self_distillation/reprompt_sample_fraction == 0` for a failed batch with feedback.**

---

## Recipe 10: Full Vanilla SDPO Baseline

Run only after Recipe 9 passes. Keep all shared rollout settings identical to
Recipe 8 so GRPO and SDPO remain an apples-to-apples comparison; the only
algorithmic difference should be `loss_mode=sdpo` plus the feedback teacher path.

```bash
cd ~/verl_tau3_sdpo

export HF_CKPT=<path to final SFT global_step_*/huggingface>
export TAU3_LIVE_USER_MODEL=us.anthropic.claude-sonnet-4-6
export TAU3_LIVE_RUNTIME=official_gym
export MODEL_PATH="$HF_CKPT"
export ENABLE_THINKING=true
export VLLM_LANGUAGE_MODEL_ONLY=true
export N_GPUS_PER_NODE=8
export ROLLOUT_TP_SIZE=1
export TRAIN_BATCH_SIZE=8
export ROLLOUT_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export VAL_N=4
export TOTAL_TRAINING_STEPS=300
export TOTAL_EPOCHS=300
export TEST_FREQ=30
export SAVE_FREQ=30
export LR=1e-6
export LR_WARMUP_STEPS=0
export MAX_PROMPT_LENGTH=16384
export MAX_RESPONSE_LENGTH=12288
export MAX_MODEL_LEN=32768
export SDPO_ALPHA=1.0
export SDPO_LOSS_COEF=1.0
export SDPO_IS_CLIP=2.0
export SDPO_MAX_REPROMPT_LEN=8192
export SDPO_REPROMPT_TRUNCATION=right
export ROLLOUT_TEMPERATURE=0.4
export ROLLOUT_TOP_P=0.95
export VAL_TEMPERATURE=0.4
export VAL_TOP_P=0.95
export PYTORCH_ALLOC_CONF=expandable_segments:True

ROLLOUT_DATA_DIR=outputs/sdpo_full_sft_baseline/rollout_data \
bash run_local_tau3_sdpo_live_p5.sh \
  datasets/tau3_live_airline_canonical_json \
  sdpo_full_sft_baseline \
  json \
  2>&1 | tee logs/sdpo_full_sft_baseline.log
```

Track at minimum: terminal/nonterminal fractions, budget-exhausted fraction,
turn/tool counts, response clip ratio, success count, tokens per success,
`self_distillation/reprompt_sample_fraction`, `feedback_used_fraction`,
`teacher_prompt_saturation_fraction`, finite `actor/pg_loss`, and sane
`actor/grad_norm`.

---

## Historical Notes

- Old LoRA SFT looked good in standalone eval but produced 50-72% garbage in VERL agent loop rollouts
- Old `model_type=qwen3_5_text` / `Qwen3_5ForCausalLM` checkpoints required manual config patching — abandoned
- Qwen3.5 stock chat template strips `<think>` from non-final assistant messages; for full-trajectory SFT use the custom preserve-thinking template and pre-tokenized full-trajectory dataset
- VERL `MultiTurnSFTDataset` per-turn tokenization is incompatible with Qwen3.5 strict template; use `PretokenizedSFTDataset` for full trajectories or `TurnSFTDataset` only for the earlier turn-per-row ablation
- Qwen3.5-4B chosen as primary seed: same pass@1/best@4 as 27B, 4-15× cheaper
- 1K full-param thinking SFT pilot overfit → use 1 epoch for 5K+ data
- 9,198 successful thinking-on trajectories generated (Bedrock Opus 4.6 agent, Sonnet 4.6 user sim)
- Turn expansion: ~9 assistant turns per trajectory → ~82K turn-rows
