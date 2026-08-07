#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-"$ROOT_DIR/data/webshop"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT_DIR/outputs/trial_webshop"}
ENGINE=${ENGINE:-vllm}
NUM_GPUS=${NUM_GPUS:-8}
TRAIN_SIZE=${TRAIN_SIZE:-16}
VAL_SIZE=${VAL_SIZE:-128}
TOTAL_STEPS=${TOTAL_STEPS:-150}

if [[ -z "$MODEL_PATH" ]]; then
  printf 'MODEL_PATH must point to a local or Hugging Face model.\n' >&2
  exit 2
fi

cd "$ROOT_DIR"
python -m trial.preflight --environment webshop
python -m trial.prepare_data --output-dir "$DATA_DIR" --train-size "$TRAIN_SIZE" --val-size "$VAL_SIZE"

python -m trial.train --config-name webshop \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.name="$ENGINE" \
  trainer.default_local_dir="$OUTPUT_ROOT" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.total_epochs="$TOTAL_STEPS" \
  "$@"
