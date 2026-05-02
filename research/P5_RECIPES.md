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

## SFT Format: Turn-Per-Row

All SFT recipes use **turn-per-row** format (AReaL Tau2 style):

```text
messages = prior conversation history (thinking stripped from historical assistants)
answer   = current assistant target with thinking/content/tool_calls
```

This is required because Qwen3.5 strips historical assistant reasoning during
full multi-turn chat-template rendering. Each row trains one assistant action
given history — matching rollout inference.

Dataset class: `TurnSFTDataset` via `+data.custom_cls` hydra override.

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

## Recipe 3: Full SFT (9K trajectories, turn-per-row)

Run after audit passes. Uses all ~75K train rows, Liger for memory efficiency,
batch 32 for speed.

```bash
cd ~/verl_tau3_sdpo

MODEL_PATH=Qwen/Qwen3.5-4B \
CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' \
NUM_GPUS=8 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=500 \
TEST_FREQ=250 \
MAX_LENGTH=32768 \
MAX_TOKEN_LEN_PER_GPU=32768 \
TRAIN_BATCH_SIZE=32 \
MICRO_BATCH_SIZE_PER_GPU=1 \
USE_LIGER=true \
LR=1e-5 \
bash scripts/tau3/run_tau3_verl_sft_full_thinking.sh \
  datasets/tau3_sft_thinking_train_only \
  qwen35_4b_vlm_sft_9k_turn \
  +data.custom_cls.path=verl/utils/dataset/turn_sft_dataset.py \
  +data.custom_cls.name=TurnSFTDataset \
  +data.audit_samples=2 \
  +data.truncation=right \
  2>&1 | tee logs/sft_full_9k_turn.log
```

Expected: ~2,374 steps, ~1-2 hours on 8×H100.

If OOM:
1. Confirm Liger is active in the log (`use_liger=True`)
2. Try `TRAIN_BATCH_SIZE=16` (doubles steps to ~4,748)
3. Try `MAX_LENGTH=24576` as last resort

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

## Recipe 4: Standalone vLLM Serve Smoke

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

## Recipe 5: One-Step GRPO Smoke

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
  2>&1 | tee logs/grpo_vlm_sft_smoke.log
```

Pass criteria:
- Decoded assistant output is not gibberish
- `tau3_live_result` appears in rollout/reward fields
- Nonzero `actor/pg_loss` or `actor/grad_norm`

**Stop if decoded output is corrupted.**

---

## Recipe 6: Full GRPO Baseline

Run only after SFT checkpoint passes vLLM serve + one-step GRPO smoke.

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

ROLLOUT_DATA_DIR=outputs/grpo_full_sft_baseline/rollout_data \
bash run_local_tau3_grpo_live_p5.sh \
  datasets/tau3_live_airline_canonical_json \
  grpo_full_sft_baseline \
  json \
  2>&1 | tee logs/grpo_full_sft_baseline.log
```

No LoRA. No `SFT_LORA_*` env vars. Full checkpoint loaded directly.

---

## Recipe 7: Vanilla SDPO (not yet runnable)

`run_local_tau3_sdpo_live_p5.sh` intentionally fails fast.
See `research/migration/sdpo_latest_verl_port_notes.md` for the port plan.

Do not attempt until GRPO baseline is understood.

---

## Historical Notes

- Old LoRA SFT looked good in standalone eval but produced 50-72% garbage in VERL agent loop rollouts
- Old `model_type=qwen3_5_text` / `Qwen3_5ForCausalLM` checkpoints required manual config patching — abandoned
- Qwen3.5 chat template strips `<think>` from non-final assistant messages — turn-per-row SFT is required
- VERL `MultiTurnSFTDataset` per-turn tokenization is incompatible with Qwen3.5 strict template — use `TurnSFTDataset`
- Qwen3.5-4B chosen as primary seed: same pass@1/best@4 as 27B, 4-15× cheaper
- 1K full-param thinking SFT pilot overfit → use 1 epoch for 5K+ data
- 9,198 successful thinking-on trajectories generated (Bedrock Opus 4.6 agent, Sonnet 4.6 user sim)
- Turn expansion: ~9 assistant turns per trajectory → ~82K turn-rows
