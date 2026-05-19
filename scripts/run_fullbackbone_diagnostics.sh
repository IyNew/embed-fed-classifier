#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
GLOBAL_LOG="${GLOBAL_LOG:-$ROOT_DIR/centralized_runs/fullbackbone_diagnostics_${RUN_TS}.log}"

CONFIGS=(
  "adam_fullbackbone_dp_targeteps10_clip200_b128_e50_seed42"
  "adam_fullbackbone_dp_targeteps5_clip200_b128_e50_seed42"
  "adam_fullbackbone_dp_targeteps3_clip200_b128_e50_seed42"
  "adam_fullbackbone_dp_targeteps1_clip200_b128_e50_seed42"
  "adam_fullbackbone_dp_targeteps0p8_clip200_b128_e50_seed42"
  "adam_fullbackbone_dp_noise06_clip200_b128_e50_seed42"
  "adam_fullbackbone_dp_noise08_clip250_b128_e50_seed42"
  "adam_fullbackbone_dp_noise10_clip220_b128_e50_seed42"
)

mkdir -p "$ROOT_DIR/centralized_runs"

{
  echo "[$(date --iso-8601=seconds)] Starting full-backbone DP diagnostic experiments"
  echo "ROOT_DIR=$ROOT_DIR"
  echo "PYTHON=$PYTHON"
  echo "RUN_TS=$RUN_TS"
  echo "GLOBAL_LOG=$GLOBAL_LOG"
  echo "CONFIGS=${CONFIGS[*]}"
  echo
} >> "$GLOBAL_LOG"

failed=0
for name in "${CONFIGS[@]}"; do
  config="$ROOT_DIR/centralized/experiment_configs/${name}.yml"
  workdir="$ROOT_DIR/centralized_runs/${name}_${RUN_TS}"
  trainer="$ROOT_DIR/centralized/train_centralized_dp.py"
  grad_args=()
  batch_size="$("$PYTHON" - "$config" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(config.get("dataloader", {}).get("batch_size", ""))
PY
)"
  if [ "${ENABLE_GRAD_OUTPUT:-0}" = "1" ]; then
    grad_args=(--grad-output)
  fi

  {
    echo "================================================================"
    echo "[$(date --iso-8601=seconds)] START $name"
    echo "CONFIG=$config"
    echo "TRAINER=$trainer"
    echo "WORKDIR=$workdir"
    echo "BATCH_SIZE=$batch_size"
    echo "GRAD_OUTPUT_ARGS=${grad_args[*]:-none}"
    echo "================================================================"
  } >> "$GLOBAL_LOG"

  "$PYTHON" "$trainer" -c "$config" --workdir "$workdir" "${grad_args[@]}" >> "$GLOBAL_LOG" 2>&1
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
  echo "[$(date --iso-8601=seconds)] Full-backbone DP diagnostics completed with failures." >> "$GLOBAL_LOG"
  exit 1
fi

echo "[$(date --iso-8601=seconds)] Full-backbone DP diagnostics completed." >> "$GLOBAL_LOG"
