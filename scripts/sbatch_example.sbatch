#!/bin/bash
#SBATCH -A r01939                         # RT project account
#SBATCH --partition=gpu                   # partition
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=1                        # 1 task
#SBATCH --gres=gpu:1                      # 1 GPU
#SBATCH --cpus-per-task=10                # 10 cpu cores
#SBATCH --mem=64G                         # memory
#SBATCH --time=24:00:00                   # max run time
#SBATCH --job-name=centralized_dp         # job name
#SBATCH --output=path/to/logs/run_name_%j.out           # stdout (%j = jobID)
#SBATCH --error=path/to/logs/run_name_%j.err            # stderr 

# other useful options
# --array=0-2

# When 

module load python/3.11.5 

source venv/bin/activate

mkdir -p sbatch_logs
# Config list
CONFIGS=(
    "centralized/config_model1.yml"
    "centralized/config_model2.yml"
    "centralized/config_model3.yml"
)
# ===============
# Sequential Mode
# ===============
for config in "${CONFIGS[@]}"; do
    echo "=== Using: $config ==="
    python train.py -c "centralized/$config"
    echo "=== Done: $config ==="
done

# ===================================
# Array Mode with #SBATCH --array=0-2
# ===================================
# 
# IDX=${SLURM_ARRAY_TASK_ID}           # 0, 1, 2
# CURRENT_CONFIG=${CONFIGS[$IDX]}
# 
# 
# echo "=============================="
# echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
# echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
# echo "Node: $(hostname)"
# echo "Config: ${CURRENT_CONFIG}"
# echo "=============================="

# python centralized/train_centralized_dp.py -c "${CURRENT_CONFIG}"

