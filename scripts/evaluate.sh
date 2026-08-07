#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BENCHMARK=${BENCHMARK:-}
CHECKPOINT=${CHECKPOINT:-}
MODEL_PATH=${MODEL_PATH:-}
ENGINE=${ENGINE:-vllm}
NUM_GPUS=${NUM_GPUS:-8}
EVAL_SEED=${EVAL_SEED:-0}
EVAL_OUTPUT=${EVAL_OUTPUT:-"$ROOT_DIR/outputs/evaluation"}

if [[ -z "$MODEL_PATH" || -z "$CHECKPOINT" ]]; then
  printf 'MODEL_PATH and CHECKPOINT (a global_step_* directory) are required.\n' >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT" || "$(basename "$CHECKPOINT")" != global_step_* ]]; then
  printf 'CHECKPOINT must name an existing global_step_* directory.\n' >&2
  exit 2
fi

case "$BENCHMARK" in
  alfworld)
    CONFIG=alfworld
    EVAL_SIZE=134
    EXTRA=(env.alfworld.fixed_eval_games=true)
    ;;
  webshop)
    CONFIG=webshop
    EVAL_SIZE=500
    EXTRA=(env.webshop.fixed_eval_goals=true)
    ;;
  *)
    printf 'Set BENCHMARK to alfworld or webshop.\n' >&2
    exit 2
    ;;
esac

DATA_DIR="$ROOT_DIR/data/evaluation/$BENCHMARK"
RUN_DIR="$EVAL_OUTPUT/$BENCHMARK"
cd "$ROOT_DIR"
python -m trial.preflight --environment "$BENCHMARK"
python -m trial.prepare_data --output-dir "$DATA_DIR" --train-size 1 --val-size "$EVAL_SIZE"

python -m trial.train --config-name "$CONFIG" \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size=1 \
  data.val_batch_size="$EVAL_SIZE" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.name="$ENGINE" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.default_local_dir="$RUN_DIR" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CHECKPOINT" \
  trainer.val_before_train=true \
  trainer.val_only=true \
  trainer.test_freq=-1 \
  trainer.validation_data_dir="$RUN_DIR/generations" \
  env.seed="$EVAL_SEED" \
  "${EXTRA[@]}" \
  "$@"
