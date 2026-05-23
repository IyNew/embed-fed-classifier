#!/usr/bin/env bash
#SBATCH --job-name=embed-shadow-resume
#SBATCH --output=shadow_runs/full_shadow_training/slurm-%j.out
#SBATCH --error=shadow_runs/full_shadow_training/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail

# Submit from the transferred repository root:
#   sbatch scripts/resume_full_shadow_training.sbatch
#
# This resumes at job granularity. Do not add --force: completed job directories
# are skipped when metrics.json, best_model.pth, and last_model.pth exist.

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-shadow/experiments/full_shadow_training.yml}"
PYTHON="${PYTHON:-$ROOT_DIR/venv/bin/python}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
WORKDIR="${WORKDIR:-shadow_runs/full_shadow_training}"
GLOBAL_LOG="${GLOBAL_LOG:-$WORKDIR/full_shadow_training_resume_${RUN_TS}_slurm${SLURM_JOB_ID:-manual}.log}"

mkdir -p "$WORKDIR"

{
  echo "[$(date --iso-8601=seconds)] START full_shadow_training resume"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
  echo "ROOT_DIR=$ROOT_DIR"
  echo "CONFIG=$CONFIG"
  echo "PYTHON=$PYTHON"
  echo "WORKDIR=$WORKDIR"
  echo "GLOBAL_LOG=$GLOBAL_LOG"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "Resume behavior: skip completed jobs; no --force"
  echo
} >> "$GLOBAL_LOG"

if [ ! -x "$PYTHON" ]; then
  {
    echo "[$(date --iso-8601=seconds)] ERROR Python executable not found or not executable: $PYTHON"
    echo "Set PYTHON=/path/to/python or create $ROOT_DIR/venv before submitting."
  } >> "$GLOBAL_LOG"
  exit 2
fi

if [ ! -f "$CONFIG" ]; then
  echo "[$(date --iso-8601=seconds)] ERROR config not found: $CONFIG" >> "$GLOBAL_LOG"
  exit 2
fi

if [ ! -f "$WORKDIR/split/attack_manifest.csv" ] || [ ! -f "$WORKDIR/split/split_plan.json" ]; then
  {
    echo "[$(date --iso-8601=seconds)] ERROR expected transferred split files are missing."
    echo "Required:"
    echo "  $WORKDIR/split/attack_manifest.csv"
    echo "  $WORKDIR/split/split_plan.json"
    echo "Transfer the existing run directory, or intentionally regenerate splits before resuming."
  } >> "$GLOBAL_LOG"
  exit 2
fi

completed_before="$(find "$WORKDIR" -name metrics.json | wc -l)"
{
  echo "[$(date --iso-8601=seconds)] Completed metrics before resume: $completed_before"
  echo "[$(date --iso-8601=seconds)] Running data check"
} >> "$GLOBAL_LOG"

"$PYTHON" shadow/train/run.py -c "$CONFIG" --check-data-only >> "$GLOBAL_LOG" 2>&1

{
  echo "[$(date --iso-8601=seconds)] Launching resume command"
  echo "COMMAND: $PYTHON shadow/train/run.py -c $CONFIG"
  echo
} >> "$GLOBAL_LOG"

set +e
"$PYTHON" shadow/train/run.py -c "$CONFIG" >> "$GLOBAL_LOG" 2>&1
status=$?
set -e

completed_after="$(find "$WORKDIR" -name metrics.json | wc -l)"
{
  echo
  echo "[$(date --iso-8601=seconds)] END full_shadow_training resume status=$status"
  echo "Completed metrics after resume: $completed_after"
} >> "$GLOBAL_LOG"

if [ "$status" -eq 0 ]; then
  echo "DONE full_shadow_training resume completed at $(date --iso-8601=seconds)" >> "$GLOBAL_LOG"
else
  echo "ERROR full_shadow_training resume failed with status=$status at $(date --iso-8601=seconds)" >> "$GLOBAL_LOG"
fi

exit "$status"
