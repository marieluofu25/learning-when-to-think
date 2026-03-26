#!/bin/bash
#SBATCH --job-name=grpo-eval
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=slurm-%j.out

# --- Modules ---
module load cuda/12.2
module load python/3.11

# --- Environment ---
cd ~/learning-when-to-think
source .venv/bin/activate

export HF_HOME=/scratch/general/vast/$USER/hf_cache
export PYTHONUNBUFFERED=1

# --- Config ---
CONFIG=${1:-configs/chpc_grpo.yaml}
RESULTS_DIR=${2:-results/eval_chpc}

echo "=== Evaluation ==="
echo "Config:  $CONFIG"
echo "Results: $RESULTS_DIR"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:   $(date)"
echo "========================"

python scripts/eval.py --config "$CONFIG" --output-dir "$RESULTS_DIR"

echo "=== Done: $(date) ==="
