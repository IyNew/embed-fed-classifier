#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
CONFIG_DIR="$ROOT_DIR/conditional_dp/experiment_configs/tiny_ratio_dp_nondp_sweep"
MANIFEST="${MANIFEST:-$CONFIG_DIR/manifest.csv}"
GLOBAL_LOG="${GLOBAL_LOG:-$ROOT_DIR/conditional_dp/runs/tiny_ratio_dp_nondp_sweep_${RUN_TS}.log}"
TRAINER="$ROOT_DIR/centralized/train_centralized_dp.py"

mkdir -p "$ROOT_DIR/conditional_dp/runs"

{
  echo "[$(date --iso-8601=seconds)] Starting conditional DP tiny ratio sweep"
  echo "ROOT_DIR=$ROOT_DIR"
  echo "PYTHON=$PYTHON"
  echo "RUN_TS=$RUN_TS"
  echo "MANIFEST=$MANIFEST"
  echo "TRAINER=$TRAINER"
  echo "GLOBAL_LOG=$GLOBAL_LOG"
  echo
} >> "$GLOBAL_LOG"

if [ ! -f "$MANIFEST" ]; then
  echo "[$(date --iso-8601=seconds)] Manifest not found: $MANIFEST" >> "$GLOBAL_LOG"
  exit 1
fi

failed=0
{
  read -r _header
  while IFS=, read -r config_path run_name n_plus rho variant epsilon clip workdir_stem; do
    if [ -z "$config_path" ]; then
      continue
    fi
    config="$ROOT_DIR/$config_path"
    workdir="$ROOT_DIR/conditional_dp/runs/tiny_ratio_dp_nondp_sweep/${workdir_stem}_${RUN_TS}"

    {
      echo "================================================================"
      echo "[$(date --iso-8601=seconds)] START $run_name"
      echo "CONFIG=$config"
      echo "WORKDIR=$workdir"
      echo "N_PLUS=$n_plus"
      echo "RHO=$rho"
      echo "VARIANT=$variant"
      echo "EPSILON=$epsilon"
      echo "CLIP=$clip"
      echo "================================================================"
    } >> "$GLOBAL_LOG"

    "$PYTHON" "$TRAINER" \
      -c "$config" \
      --workdir "$workdir" \
      >> "$GLOBAL_LOG" 2>&1
    status=$?

    {
      echo "[$(date --iso-8601=seconds)] END $run_name status=$status"
      echo
    } >> "$GLOBAL_LOG"

    if [ "$status" -ne 0 ]; then
      failed=1
      echo "[$(date --iso-8601=seconds)] $run_name failed; continuing to next config." >> "$GLOBAL_LOG"
    fi
  done
} < "$MANIFEST"

if [ "$failed" -ne 0 ]; then
  echo "[$(date --iso-8601=seconds)] Conditional DP tiny ratio sweep completed with failures." >> "$GLOBAL_LOG"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] Conditional DP tiny ratio sweep completed." >> "$GLOBAL_LOG"
