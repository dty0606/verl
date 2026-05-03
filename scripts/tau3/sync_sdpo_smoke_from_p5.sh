#!/usr/bin/env bash
# Run on P5 after the SDPO smoke completes.
# Zips rollout data + the key log tail and pushes to S3 for Kiro pull.
#
# Usage (on P5):
#   bash scripts/tau3/sync_sdpo_smoke_from_p5.sh

set -euo pipefail

SMOKE_NAME="${SMOKE_NAME:-sdpo_vlm_sft_smoke}"
S3_PREFIX="${S3_PREFIX:-s3://tianyd-rlvr-research/tau3-sdpo/latest-verl}"
STAGE_DIR="${STAGE_DIR:-/tmp/${SMOKE_NAME}_bundle}"

ROLLOUT_DIR="outputs/${SMOKE_NAME}/rollout_data"
LOG_FILE="logs/${SMOKE_NAME}.log"

echo "=== Bundling SDPO smoke artifacts ==="
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

# 1. Rollout data (actual prompt/response/score per rollout)
if [ -d "$ROLLOUT_DIR" ]; then
    cp -r "$ROLLOUT_DIR" "$STAGE_DIR/rollout_data"
    echo "Copied rollout_data: $(du -sh $STAGE_DIR/rollout_data | cut -f1)"
else
    echo "WARN: $ROLLOUT_DIR not found"
fi

# 2. Full training log
if [ -f "$LOG_FILE" ]; then
    cp "$LOG_FILE" "$STAGE_DIR/full_log.txt"
    echo "Copied log: $(du -sh $STAGE_DIR/full_log.txt | cut -f1)"
fi

# 3. Metric tail — grep just the lines we care about
if [ -f "$LOG_FILE" ]; then
    {
        echo "=== SDPO-specific metrics ==="
        grep -E "self_distillation/" "$LOG_FILE" || true
        echo ""
        echo "=== actor metrics ==="
        grep -E "actor/(pg_loss|grad_norm|loss|entropy|ppo_kl|lr|pg_clipfrac)" "$LOG_FILE" | head -50 || true
        echo ""
        echo "=== tau3_live rewards ==="
        grep -E "tau3_live/(reward|score|acc)" "$LOG_FILE" | head -30 || true
        echo ""
        echo "=== feedback sanity ==="
        grep -E "s_sdpo/(diagnostic|failure_type)" "$LOG_FILE" | head -30 || true
        echo ""
        echo "=== errors/warnings ==="
        grep -iE "(error|traceback|warning|nan|inf)" "$LOG_FILE" | grep -v "resource_tracker\|KeyError: '/psm\|KeyError: '/mp-\|BrokenPipe" | head -30 || true
    } > "$STAGE_DIR/metrics_summary.txt"
    echo "Wrote metrics_summary.txt"
fi

# 4. wandb run dir (small, but useful for step-by-step history)
WANDB_DIR=$(ls -d wandb/run-*sdpo_vlm_sft_smoke* 2>/dev/null | tail -1 || echo "")
if [ -n "$WANDB_DIR" ]; then
    # Just the config and summary, not the full artifacts
    mkdir -p "$STAGE_DIR/wandb_meta"
    cp "$WANDB_DIR/files/config.yaml" "$STAGE_DIR/wandb_meta/config.yaml" 2>/dev/null || true
    cp "$WANDB_DIR/files/wandb-summary.json" "$STAGE_DIR/wandb_meta/summary.json" 2>/dev/null || true
    cp "$WANDB_DIR/files/wandb-metadata.json" "$STAGE_DIR/wandb_meta/metadata.json" 2>/dev/null || true
    echo "Copied wandb_meta"
fi

# 5. Zip it
BUNDLE_ZIP="/tmp/${SMOKE_NAME}_bundle.zip"
rm -f "$BUNDLE_ZIP"
(cd "$STAGE_DIR" && zip -rq "$BUNDLE_ZIP" .)
echo "Created bundle: $(du -sh $BUNDLE_ZIP | cut -f1)"

# 6. Upload
aws s3 cp "$BUNDLE_ZIP" "$S3_PREFIX/diagnostics/${SMOKE_NAME}_bundle.zip"
echo ""
echo "=== DONE ==="
echo "Bundle at: $S3_PREFIX/diagnostics/${SMOKE_NAME}_bundle.zip"
echo "Kiro can pull with:"
echo "  aws s3 cp $S3_PREFIX/diagnostics/${SMOKE_NAME}_bundle.zip /tmp/"
echo "  unzip -o /tmp/${SMOKE_NAME}_bundle.zip -d /tmp/${SMOKE_NAME}_bundle/"
