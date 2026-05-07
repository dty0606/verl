#!/usr/bin/env bash
# Bundle a GRPO readiness smoke from P5 and optionally upload it to S3.
#
# Usage on P5:
#   SMOKE_NAME=grpo_vllm_v1_readiness_10step \
#   PROFILE_NAME=02_auto_prefix_24k_48k \
#   bash scripts/tau3/bundle_grpo_readiness_from_p5.sh

set -euo pipefail

SMOKE_NAME="${SMOKE_NAME:-grpo_vllm_v1_readiness_10step}"
PROFILE_NAME="${PROFILE_NAME:-02_auto_prefix_24k_48k}"
S3_PREFIX="${S3_PREFIX:-s3://tianyd-rlvr-research/tau3-sdpo/latest-verl}"
STAGE_DIR="${STAGE_DIR:-/tmp/${SMOKE_NAME}_bundle}"
INCLUDE_WANDB_DIR="${INCLUDE_WANDB_DIR:-0}"

LOG_FILE="${LOG_FILE:-logs/${SMOKE_NAME}.log}"
MATRIX_ROOT="${MATRIX_ROOT:-logs/vllm_v1_capacity_matrix}"
ROLLOUT_DIR="${ROLLOUT_DIR:-outputs/vllm_v1_capacity_matrix/${PROFILE_NAME}/rollout_data}"

echo "=== Bundling GRPO readiness artifacts ==="
echo "SMOKE_NAME=$SMOKE_NAME"
echo "PROFILE_NAME=$PROFILE_NAME"
echo "LOG_FILE=$LOG_FILE"
echo "MATRIX_ROOT=$MATRIX_ROOT"
echo "ROLLOUT_DIR=$ROLLOUT_DIR"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

if [ -f "$LOG_FILE" ]; then
    cp "$LOG_FILE" "$STAGE_DIR/full_log.txt"
else
    echo "WARN: $LOG_FILE not found"
fi

LATEST_MATRIX_DIR="$(find "$MATRIX_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
if [ -n "$LATEST_MATRIX_DIR" ]; then
    mkdir -p "$STAGE_DIR/capacity_matrix"
    cp "$LATEST_MATRIX_DIR"/summary.txt "$STAGE_DIR/capacity_matrix/summary.txt" 2>/dev/null || true
    cp "$LATEST_MATRIX_DIR"/00_preflight.log "$STAGE_DIR/capacity_matrix/00_preflight.log" 2>/dev/null || true
    cp "$LATEST_MATRIX_DIR/${PROFILE_NAME}.log" "$STAGE_DIR/capacity_matrix/${PROFILE_NAME}.log" 2>/dev/null || true
    echo "Copied matrix logs from $LATEST_MATRIX_DIR"
fi

if [ -d "$ROLLOUT_DIR" ]; then
    mkdir -p "$STAGE_DIR/rollout_data"
    cp -r "$ROLLOUT_DIR"/. "$STAGE_DIR/rollout_data/"
    echo "Copied rollout_data: $(du -sh "$STAGE_DIR/rollout_data" | cut -f1)"
else
    echo "WARN: $ROLLOUT_DIR not found"
fi

{
    echo "=== git/source ==="
    git rev-parse --short HEAD 2>/dev/null || true
    git status --short 2>/dev/null || true
    echo
    echo "=== runtime env ==="
    env | sort | grep -E "^(CAPACITY_PROFILES|CUDA_HOME|LD_LIBRARY_PATH|MODEL_ALIAS|MODEL_PATH|PROJECT_NAME|TAU2_DATA_DIR|TAU3_|TOTAL_|VLLM_|MAX_|TRAIN_|VAL_|ROLLOUT_|PPO_)" || true
    echo
    echo "=== packages ==="
    python - <<'PY' || true
import importlib.metadata as md
for pkg in ["torch", "vllm", "flashinfer-python", "flash-attn", "transformers", "tokenizers", "verl"]:
    try:
        print(f"{pkg}={md.version(pkg)}")
    except Exception as exc:
        print(f"{pkg}=<missing:{exc}>")
PY
    echo
    echo "=== gpu ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
} > "$STAGE_DIR/runtime_snapshot.txt"

if [ -f "$STAGE_DIR/full_log.txt" ]; then
    {
        echo "=== key smoke lines ==="
        grep -E "PREFLIGHT=PASS|Capacity profiles|Profile:|PASS |FAIL |SKIP |Tau3 all-messages observation|VLLM_USE_V1|VLLM_ENABLE_PREFIX_CACHING|VLLM_KV_CACHE_DTYPE|MAX_RESPONSE_LENGTH|MAX_MODEL_LEN" "$STAGE_DIR/full_log.txt" || true
        echo
        echo "=== metrics ==="
        grep -E "response_length/|prompt_length/|tau3_live/|critic/rewards|critic/score|actor/(pg_loss|grad_norm|entropy|ppo_kl|pg_clipfrac)|training/global_step" "$STAGE_DIR/full_log.txt" || true
        echo
        echo "=== warnings/errors ==="
        grep -iE "error|traceback|warning|oom|out of memory|cuda_runtime|cicc|gdn|page size|undefined symbol|AttributeError|nan|inf" "$STAGE_DIR/full_log.txt" | head -200 || true
    } > "$STAGE_DIR/metrics_summary.txt"
fi

if command -v wandb >/dev/null 2>&1; then
    WANDB_DIR="$(find wandb -maxdepth 1 -type d -name "run-*" 2>/dev/null | sort | tail -1 || true)"
else
    WANDB_DIR="$(find wandb -maxdepth 1 -type d -name "run-*" 2>/dev/null | sort | tail -1 || true)"
fi
if [ -n "$WANDB_DIR" ]; then
    mkdir -p "$STAGE_DIR/wandb_meta"
    cp "$WANDB_DIR/files/config.yaml" "$STAGE_DIR/wandb_meta/config.yaml" 2>/dev/null || true
    cp "$WANDB_DIR/files/wandb-summary.json" "$STAGE_DIR/wandb_meta/summary.json" 2>/dev/null || true
    cp "$WANDB_DIR/files/wandb-metadata.json" "$STAGE_DIR/wandb_meta/metadata.json" 2>/dev/null || true
    cp "$WANDB_DIR/files/output.log" "$STAGE_DIR/wandb_meta/output.log" 2>/dev/null || true
    find "$WANDB_DIR" -maxdepth 3 -type f \( -name "wandb-history*.jsonl" -o -name "debug*.log" -o -name "requirements.txt" \) \
        -exec cp {} "$STAGE_DIR/wandb_meta/" \; 2>/dev/null || true
    if [ "$INCLUDE_WANDB_DIR" = "1" ]; then
        mkdir -p "$STAGE_DIR/wandb_run"
        tar -C "$WANDB_DIR" -cf "$STAGE_DIR/wandb_run/latest_wandb_run.tar" . 2>/dev/null || true
    fi
    echo "Copied wandb metadata from $WANDB_DIR"
fi

if [ -f scripts/tau3/analyze_grpo_smoke_bundle.py ]; then
    python scripts/tau3/analyze_grpo_smoke_bundle.py "$STAGE_DIR" \
        --output-json "$STAGE_DIR/analysis_summary.json" \
        > "$STAGE_DIR/analysis_summary.txt" || true
fi

BUNDLE_TGZ="/tmp/${SMOKE_NAME}_bundle.tgz"
rm -f "$BUNDLE_TGZ"
tar -C "$STAGE_DIR" -czf "$BUNDLE_TGZ" .
echo "Created bundle: $(du -sh "$BUNDLE_TGZ" | cut -f1)"

if command -v aws >/dev/null 2>&1; then
    aws s3 cp "$BUNDLE_TGZ" "$S3_PREFIX/diagnostics/${SMOKE_NAME}_bundle.tgz"
    echo "Bundle at: $S3_PREFIX/diagnostics/${SMOKE_NAME}_bundle.tgz"
else
    echo "aws CLI not found; bundle is local only: $BUNDLE_TGZ"
fi
