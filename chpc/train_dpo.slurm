#!/bin/bash
# Args: [config]
#SBATCH --job-name=dpo-train
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=chpc/results/logs/slurm-train-dpo-%j.out

module load cuda/12.2
module load python/3.11

cd "${CHPC_REPO:-$HOME/learning-when-to-think}"
mkdir -p chpc/results/logs
source .venv/bin/activate

export HF_HOME=/scratch/general/vast/$USER/hf_cache
export PYTHONUNBUFFERED=1

CONFIG=${1:-configs/chpc_dpo.yaml}

echo "=== DPO training ==="
echo "Config: $CONFIG"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

python scripts/train_dpo.py --config "$CONFIG"

echo "=== Done: $(date) ==="
