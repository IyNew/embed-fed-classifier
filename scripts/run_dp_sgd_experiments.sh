#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
GLOBAL_LOG="${GLOBAL_LOG:-$ROOT_DIR/centralized_runs/dp_sgd_experiments_${RUN_TS}.log}"

CONFIGS=(
  "opacus_headonly_noise10_b256_clip1_e15_seed42:$ROOT_DIR/centralized/experiment_configs/opacus_headonly_noise10_b256_clip1_e15_seed42.yml"
  "opacus_headonly_noise10_b512_clip05_autocw_e15_seed42:$ROOT_DIR/centralized/experiment_configs/opacus_headonly_noise10_b512_clip05_autocw_e15_seed42.yml"
  "opacus_headonly_noise12_b256_clip1_e15_seed42:$ROOT_DIR/centralized/experiment_configs/opacus_headonly_noise12_b256_clip1_e15_seed42.yml"
)

mkdir -p "$ROOT_DIR/centralized_runs"

{
  echo "[$(date --iso-8601=seconds)] Starting head-only Opacus DP-SGD experiment sweep"
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
  trainer="$ROOT_DIR/centralized/train_centralized_dp.py"

  {
    echo "================================================================"
    echo "[$(date --iso-8601=seconds)] START $name"
    echo "CONFIG=$config"
    echo "TRAINER=$trainer"
    echo "WORKDIR=$workdir"
    echo "================================================================"
  } >> "$GLOBAL_LOG"

  "$PYTHON" "$trainer" \
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

echo "[$(date --iso-8601=seconds)] Head-only Opacus DP-SGD experiment sweep completed." >> "$GLOBAL_LOG"
