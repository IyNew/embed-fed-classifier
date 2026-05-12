#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
GLOBAL_LOG="${GLOBAL_LOG:-$ROOT_DIR/centralized_runs/dp_adam_experiments_${RUN_TS}.log}"

# put the configs in an array for easy iteration
CONFIGS=(
  "sanity_off_adam_lr5e4_b16:$ROOT_DIR/centralized/experiment_configs/sanity_off_adam_lr5e4_b16.yml"
  "dp_adam_lr1e4_b32_norm05_noise025:$ROOT_DIR/centralized/experiment_configs/dp_adam_lr1e4_b32_norm05_noise025.yml"
  "dp_adam_lr1e4_b32_norm05_noise05:$ROOT_DIR/centralized/experiment_configs/dp_adam_lr1e4_b32_norm05_noise05.yml"
)

mkdir -p "$ROOT_DIR/centralized_runs"

{
  echo "[$(date --iso-8601=seconds)] Starting DP-Adam experiment sweep"
  echo "ROOT_DIR=$ROOT_DIR"
  echo "PYTHON=$PYTHON"
  echo "RUN_TS=$RUN_TS"
  echo "GLOBAL_LOG=$GLOBAL_LOG"
  echo
} >> "$GLOBAL_LOG"

for entry in "${CONFIGS[@]}"; do
  name="${entry%%:*}"
  config="${entry#*:}"
  workdir="$ROOT_DIR/centralized_runs/${name}_${RUN_TS}"

  {
    echo "================================================================"
    echo "[$(date --iso-8601=seconds)] START $name"
    echo "CONFIG=$config"
    echo "WORKDIR=$workdir"
    echo "================================================================"
  } >> "$GLOBAL_LOG"

  "$PYTHON" "$ROOT_DIR/centralized/train_centralized.py" \
    -c "$config" \
    --workdir "$workdir" \
    >> "$GLOBAL_LOG" 2>&1
  status=$?

  {
    echo "[$(date --iso-8601=seconds)] END $name status=$status"
    echo
  } >> "$GLOBAL_LOG"

  if [ "$status" -ne 0 ]; then
    echo "[$(date --iso-8601=seconds)] Sweep stopped after $name failed." >> "$GLOBAL_LOG"
    exit "$status"
  fi
done

echo "[$(date --iso-8601=seconds)] DP-Adam experiment sweep completed." >> "$GLOBAL_LOG"
