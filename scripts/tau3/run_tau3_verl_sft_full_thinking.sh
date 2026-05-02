#!/usr/bin/env bash
# Full-parameter tau3 thinking SFT using latest VERL's native SFT trainer.
#
# Usage:
#   ./scripts/tau3/run_tau3_verl_sft_full_thinking.sh <dataset_dir> [experiment_suffix]
#
# The dataset directory must contain train.parquet and test.parquet with raw
# multi-turn columns. Preferred turn-level format has:
# messages, answer, tools, and enable_thinking.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dataset_dir> [experiment_suffix]"
    exit 1
fi

DATASET_DIR="$1"
SUFFIX="${2:-tau3_verl_sft_full_thinking}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_FILE="$DATASET_DIR/test.parquet"
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VAL_FILE" ]; then
    echo "Error: expected train.parquet and test.parquet under $DATASET_DIR"
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
NUM_GPUS="${NUM_GPUS:-8}"
REPORT_TO="${REPORT_TO:-[\"console\",\"wandb\"]}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"optimizer\",\"extra\",\"hf_model\"]}"
EXP_NAME="TAU3-VERL-SFT-FULL-$(echo "$MODEL_PATH" | tr '/:' '--')-${SUFFIX}"
CHECKPOINT_ROOT="${SFT_CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints/SDPO/tau3_verl_sft}"
RUN_DIR="$CHECKPOINT_ROOT/$EXP_NAME"

mkdir -p "$RUN_DIR" logs
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

CMD=(
    torchrun --standalone --nnodes="${NNODES:-1}" --nproc_per_node="$NUM_GPUS"
    -m verl.trainer.sft_trainer
    "data.train_files=$TRAIN_FILE"
    "data.val_files=$VAL_FILE"
    "data.messages_key=messages"
    "+data.answer_key=${ANSWER_KEY:-answer}"
    "data.tools_key=tools"
    "data.enable_thinking_key=enable_thinking"
    "data.enable_thinking_default=True"
    "data.pad_mode=no_padding"
    "data.max_length=${MAX_LENGTH:-32768}"
    "data.truncation=${TRUNCATION:-error}"
    "data.ignore_input_ids_mismatch=${IGNORE_INPUT_IDS_MISMATCH:-True}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE:-8}"
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-1}"
    "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-32768}"
    "data.use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}"
    "model.path=$MODEL_PATH"
    "model.trust_remote_code=${TRUST_REMOTE_CODE:-True}"
    "model.use_remove_padding=${USE_REMOVE_PADDING:-True}"
    "model.use_liger=${USE_LIGER:-False}"
    "model.lora_rank=0"
    "engine=fsdp"
    "engine.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
    "optim.lr=${LR:-1e-5}"
    "trainer.project_name=${PROJECT_NAME:-SDPO-${USER:-$(whoami)}}"
    "trainer.experiment_name=$EXP_NAME"
    "trainer.default_local_dir=$RUN_DIR"
    "trainer.total_epochs=${TOTAL_EPOCHS:-1}"
    "trainer.save_freq=${SAVE_FREQ:-50}"
    "trainer.test_freq=${TEST_FREQ:-25}"
    "trainer.logger=$REPORT_TO"
    "trainer.n_gpus_per_node=$NUM_GPUS"
    "checkpoint.save_contents=$CHECKPOINT_SAVE_CONTENTS"
)

echo "----------------------------------------------------------------"
echo "Starting latest-VERL tau3 full-parameter thinking SFT"
echo "Experiment: $EXP_NAME"
echo "Dataset: $DATASET_DIR"
echo "Model: $MODEL_PATH"
echo "Output: $RUN_DIR"
echo "----------------------------------------------------------------"

"${CMD[@]}" "${@:3}"
