#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-local_training_example}"
GLOBAL_LOG="${GLOBAL_LOG:-$ROOT_DIR/centralized_runs/${EXPERIMENT_NAME}_${RUN_TS}.log}"
TRAINER="${TRAINER:-$ROOT_DIR/centralized/train_centralized_dp.py}"
CONFIG_DIR="${CONFIG_DIR:-$ROOT_DIR/centralized/experiment_configs}"

CONFIGS=(
  "example_config_name_without_yml"
)

mkdir -p "$ROOT_DIR/centralized_runs"

{
  echo "[$(date --iso-8601=seconds)] Starting ${EXPERIMENT_NAME}"
  echo "ROOT_DIR=$ROOT_DIR"
  echo "PYTHON=$PYTHON"
  echo "RUN_TS=$RUN_TS"
  echo "GLOBAL_LOG=$GLOBAL_LOG"
  echo "TRAINER=$TRAINER"
  echo "CONFIG_DIR=$CONFIG_DIR"
  echo "CONFIGS=${CONFIGS[*]}"
  echo
} >> "$GLOBAL_LOG"

failed=0
for name in "${CONFIGS[@]}"; do
  config="$CONFIG_DIR/${name}.yml"
  workdir="$ROOT_DIR/centralized_runs/${name}_${RUN_TS}"

  {
    echo "================================================================"
    echo "[$(date --iso-8601=seconds)] START $name"
    echo "CONFIG=$config"
    echo "TRAINER=$TRAINER"
    echo "WORKDIR=$workdir"
    echo "================================================================"
  } >> "$GLOBAL_LOG"

  "$PYTHON" "$TRAINER" -c "$config" --workdir "$workdir" >> "$GLOBAL_LOG" 2>&1
  status=$?

  {
    echo "[$(date --iso-8601=seconds)] END $name status=$status"
    echo
  } >> "$GLOBAL_LOG"

  if [ "$status" -ne 0 ]; then
    failed=1
    echo "[$(date --iso-8601=seconds)] $name failed; continuing to next config." >> "$GLOBAL_LOG"
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "[$(date --iso-8601=seconds)] ${EXPERIMENT_NAME} completed with failures." >> "$GLOBAL_LOG"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] ${EXPERIMENT_NAME} completed." >> "$GLOBAL_LOG"
