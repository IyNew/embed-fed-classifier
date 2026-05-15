#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
GLOBAL_LOG="${GLOBAL_LOG:-$ROOT_DIR/centralized_runs/dp_adam_pilots_${RUN_TS}.log}"

CONFIGS=(
  "adam_headonly_nondp_b128_e50_seed42:$ROOT_DIR/centralized/experiment_configs/adam_headonly_nondp_b128_e50_seed42.yml"
  "adam_fullbackbone_nondp_b128_e50_seed42:$ROOT_DIR/centralized/experiment_configs/adam_fullbackbone_nondp_b128_e50_seed42.yml"
  "adam_headonly_dp_noise02_clip5_b128_e10_seed42:$ROOT_DIR/centralized/experiment_configs/adam_headonly_dp_noise02_clip5_b128_e10_seed42.yml"
  "adam_headonly_dp_noise05_clip5_b128_e10_seed42:$ROOT_DIR/centralized/experiment_configs/adam_headonly_dp_noise05_clip5_b128_e10_seed42.yml"
  "adam_headonly_dp_noise05_clip10_b128_e10_seed42:$ROOT_DIR/centralized/experiment_configs/adam_headonly_dp_noise05_clip10_b128_e10_seed42.yml"
  "adam_headonly_dp_sgdrecipe_noise10_clip1_b256_e15_seed42:$ROOT_DIR/centralized/experiment_configs/adam_headonly_dp_sgdrecipe_noise10_clip1_b256_e15_seed42.yml"
  "adam_headonly_dp_sgdrecipe_noise12_clip1_b256_e15_seed42:$ROOT_DIR/centralized/experiment_configs/adam_headonly_dp_sgdrecipe_noise12_clip1_b256_e15_seed42.yml"
)

mkdir -p "$ROOT_DIR/centralized_runs"

{
  echo "[$(date --iso-8601=seconds)] Starting Adam classifier pilot sweep"
  echo "ROOT_DIR=$ROOT_DIR"
  echo "PYTHON=$PYTHON"
  echo "RUN_TS=$RUN_TS"
  echo "GLOBAL_LOG=$GLOBAL_LOG"
  echo
} >> "$GLOBAL_LOG"

failed=0
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
    failed=1
    echo "[$(date --iso-8601=seconds)] $name failed; continuing to next config." >> "$GLOBAL_LOG"
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "[$(date --iso-8601=seconds)] Adam classifier pilot sweep completed with failures." >> "$GLOBAL_LOG"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] Adam classifier pilot sweep completed." >> "$GLOBAL_LOG"
