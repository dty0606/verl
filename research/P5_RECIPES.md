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

## Remote Layout

```bash
LATEST_VERL_ROOT=~/verl_tau3_sdpo    # active execution repo
OLD_SDPO_ARCHIVE=~/SDPO-qwen35       # old archive, read-only reference
```

## S3 Sync

```bash
export S3_PREFIX=s3://tianyd-rlvr-research/tau3-sdpo/latest-verl

# Pull generated trajectories from P5
aws s3 sync ~/verl_tau3_sdpo/outputs/generated_sft/ "$S3_PREFIX/generated_sft/"

# Pull checkpoints from P5 (only when needed for cross-machine access)
aws s3 sync ~/verl_tau3_sdpo/checkpoints/ "$S3_PREFIX/checkpoints/" --exclude "*/optim*"

# Push code changes through git, not S3
```

Keep generated trajectories, checkpoints, rollout data, wandb artifacts,
and large logs in S3 or P5-local. Do not commit them to GitHub.

---

## Recipe 0: Environment Truth

Run this before any training or smoke. Record output in `research/session_sync.md`.

```bash
cd ~/verl_tau3_sdpo
echo "=== Git ==="
git rev-parse HEAD
git status --short

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

echo "=== GPU ==="
nvidia-smi
```

---

## Recipe 1: Build GRPO/Smoke Dataset (tau3 airline canonical)

This creates the train/test parquet that GRPO and SFT smoke use.
Must be run once on P5 where `tau2-bench` is installed.

```bash
cd ~/verl_tau3_sdpo

python3 examples/data_preprocess/tau3_live_multiturn.py \
  --output-dir datasets/tau3_live_airline_canonical_json \
  --domain airline \
  --feedback-mode json
```

Verify:

```bash
python3 -c "
import pandas as pd
for split in ['train', 'test']:
    df = pd.read_parquet(f'datasets/tau3_live_airline_canonical_json/{split}.parquet')
    print(f'{split}: {len(df)} rows')
"
# Expected: train=30, test=20
```

---

## Recipe 2: Tiny SFT Export Smoke

Goal: prove latest VERL can full-param SFT Qwen3.5-4B and export VLM-format HF checkpoint.

### 2a. Build a tiny SFT dataset (if 10K generation not yet done)

Use a handful of the old v1 thinking trajectories or a synthetic mini-set.
If the 10K generation is done, use the real data instead and skip this.

```bash
# Option A: use real generated data (preferred)
python3 scripts/tau3/build_protocol_sft_data.py \
  --input outputs/tau3_protocol_candidates_thinking_10k \
  --output-dir datasets/tau3_sft_thinking_train_only \
  --train-only-holdout \
  --include-thinking-traces \
  --id-transform randomize \
  --id-variants 3 \
  --include-original-id-variant \
  --overwrite

# Option B: for smoke only, use a tiny subset
# (manually select 5-10 trajectory JSONs into a temp dir, then run the builder)
```

### 2b. Run tiny SFT

```bash
cd ~/verl_tau3_sdpo

MODEL_PATH=Qwen/Qwen3.5-4B \
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' \
NUM_GPUS=8 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=1 \
TEST_FREQ=1 \
MAX_LENGTH=4096 \
MAX_TOKEN_LEN_PER_GPU=4096 \
TRAIN_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=1 \
bash scripts/tau3/run_tau3_verl_sft_full_thinking.sh \
  datasets/tau3_sft_thinking_train_only \
  qwen35_4b_vlm_export_smoke \
  > logs/sft_vlm_export_smoke.log 2>&1
```

### 2c. Verify VLM-format checkpoint

```bash
# Find the checkpoint
HF_CKPT=$(ls -d checkpoints/SDPO/tau3_verl_sft/*/global_step_*/huggingface | head -1)
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

**Stop if this fails.** Do not patch config.json and pretend it's validated.

---

## Recipe 3: Standalone vLLM Serve Smoke

Goal: prove the SFT export loads in vLLM before involving GRPO.

### 3a. Start vLLM server

```bash
export HF_CKPT=$(ls -d checkpoints/SDPO/tau3_verl_sft/*/global_step_*/huggingface | head -1)

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
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready
for i in $(seq 1 60); do
  curl -s http://127.0.0.1:9100/health && break
  sleep 2
done
```

### 3b. Test inference

```bash
curl http://127.0.0.1:9100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tau3-sft-smoke",
    "messages": [{"role": "user", "content": "Say one short sentence about airline customer service."}],
    "temperature": 0.2,
    "max_tokens": 80
  }'
```

Pass criteria:
- Server starts without `Qwen3_5TextConfig` mismatch
- Output is coherent English
- No multilingual/gibberish corruption
- No vision-encoder crash

```bash
# Cleanup
kill $VLLM_PID 2>/dev/null
```

**Stop if output is gibberish.** Debug SFT export before GRPO.

---

## Recipe 4: One-Step GRPO Smoke

Goal: prove latest VERL can load the SFT checkpoint, run tau3 live interaction,
and produce one nonzero actor update.

### 4a. Build GRPO dataset (if not done in Recipe 1)

```bash
python3 examples/data_preprocess/tau3_live_multiturn.py \
  --output-dir datasets/tau3_live_airline_canonical_json \
  --domain airline \
  --feedback-mode json
```

### 4b. Launch one-step GRPO

```bash
cd ~/verl_tau3_sdpo

export HF_CKPT=$(ls -d checkpoints/SDPO/tau3_verl_sft/*/global_step_*/huggingface | head -1)
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
export MAX_PROMPT_LENGTH=4096
export MAX_RESPONSE_LENGTH=512
export MAX_MODEL_LEN=8192
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
  > logs/grpo_vlm_sft_smoke.log 2>&1
```

### 4c. Verify GRPO smoke

```bash
# Check rollout output is not gibberish
grep -A2 "decoded\|output\|response" outputs/grpo_vlm_sft_smoke/rollout_data/*.jsonl 2>/dev/null | head -20

# Check tau3_live_result appears
grep "tau3_live_result" logs/grpo_vlm_sft_smoke.log | head -5

# Check actor update metrics are nonzero
grep -E "actor/pg_loss|actor/grad_norm" logs/grpo_vlm_sft_smoke.log | tail -5
```

Pass criteria:
- vLLM args include `--language-model-only`
- Served model path is the SFT `huggingface` checkpoint
- Decoded assistant output is not gibberish
- Tau3 tool calls parse through `tau3_qwen`
- `tau3_live_result` appears in rollout/reward fields
- At least one actor update logs nonzero `actor/pg_loss` or `actor/grad_norm`

**Stop if decoded output is corrupted.** Do not run long GRPO/SDPO.

---

## Recipe 5: Full SFT (after smoke passes)

Run only after Recipes 2-4 all pass.

```bash
cd ~/verl_tau3_sdpo

MODEL_PATH=Qwen/Qwen3.5-4B \
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' \
NUM_GPUS=8 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=50 \
TEST_FREQ=25 \
MAX_LENGTH=32768 \
MAX_TOKEN_LEN_PER_GPU=32768 \
TRAIN_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=1 \
USE_LIGER=true \
bash scripts/tau3/run_tau3_verl_sft_full_thinking.sh \
  datasets/tau3_sft_thinking_train_only \
  qwen35_4b_vlm_sft_10k \
  > logs/sft_full_10k.log 2>&1 &
```

If OOM at 32K:
1. Confirm Liger/FLCE is actually active in the log
2. Fall back to `MAX_LENGTH=24576`
3. Record skipped/overlength fraction

After SFT completes, re-run Recipes 2c, 3, and 4 against the full SFT checkpoint
before starting long GRPO.

---

## Recipe 6: Full GRPO Baseline

Run only after full SFT export passes vLLM serve + one-step GRPO smoke.

```bash
cd ~/verl_tau3_sdpo

export HF_CKPT=<path to full SFT global_step_*/huggingface>
export TAU3_LIVE_USER_MODEL=us.anthropic.claude-sonnet-4-6
export TAU3_LIVE_RUNTIME=official_gym
export MODEL_PATH="$HF_CKPT"
export ENABLE_THINKING=true
export VLLM_LANGUAGE_MODEL_ONLY=true
export N_GPUS_PER_NODE=8
export ROLLOUT_TP_SIZE=1
export TRAIN_BATCH_SIZE=12
export ROLLOUT_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=12
export VAL_N=4
export TOTAL_TRAINING_STEPS=300
export TOTAL_EPOCHS=300
export TEST_FREQ=30
export SAVE_FREQ=30
export LR=1e-6
export LR_WARMUP_STEPS=0
export MAX_PROMPT_LENGTH=16384
export MAX_RESPONSE_LENGTH=4096
export MAX_MODEL_LEN=24576
export ROLLOUT_TEMPERATURE=0.4
export ROLLOUT_TOP_P=0.95
export VAL_TEMPERATURE=0.4
export VAL_TOP_P=0.95
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs

ROLLOUT_DATA_DIR=outputs/grpo_full_sft_baseline/rollout_data \
bash run_local_tau3_grpo_live_p5.sh \
  datasets/tau3_live_airline_canonical_json \
  grpo_full_sft_baseline \
  json \
  > logs/grpo_full_sft_baseline.log 2>&1 &
```

Monitor first 3 steps:

```bash
grep -E "actor/pg_loss|actor/grad_norm|tau3_live_result|garbage|gibberish" \
  logs/grpo_full_sft_baseline.log | tail -20
```

No LoRA. No `SFT_LORA_*` env vars. Full checkpoint loaded directly.

---

## Recipe 7: Vanilla SDPO (not yet runnable)

`run_local_tau3_sdpo_live_p5.sh` intentionally fails fast.
See `research/migration/sdpo_latest_verl_port_notes.md` for the port plan.

Do not attempt until GRPO baseline is understood.

---

## Historical Notes

- Old LoRA SFT looked good in standalone eval but produced 50-72% garbage in VERL agent loop rollouts (see `dty0606_SDPO/research/evidence/grpo_n8_b12_33steps_garbage/FINDINGS.md`)
- Old `model_type=qwen3_5_text` / `Qwen3_5ForCausalLM` checkpoints required manual config patching for vLLM — that path is abandoned
- Old VERL fork had a patched `fsdp_workers.py` that converted `Qwen3_5Config` → `Qwen3_5TextConfig` at FSDP load time — latest VERL does not need this
- Qwen3.5-4B was chosen as primary seed: same pass@1/best@4 as 27B on 10-task eval, 4-15× cheaper
- 1K full-param thinking SFT pilot overfit → use 1 epoch for 5K+ data
