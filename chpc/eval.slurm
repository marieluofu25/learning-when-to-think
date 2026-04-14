#!/bin/bash
# Args: [config] [checkpoint] [results_dir]
#   sbatch chpc/eval.slurm configs/chpc_grpo.yaml checkpoints/grpo_chpc/final results/eval_chpc
#SBATCH --job-name=grpo-eval
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=chpc/results/logs/slurm-eval-%j.out

# --- Modules ---
module load cuda/12.2
module load python/3.11

# --- Environment ---
cd "${CHPC_REPO:-$HOME/learning-when-to-think}"
mkdir -p chpc/results/logs
source .venv/bin/activate

export HF_HOME=/scratch/general/vast/$USER/hf_cache
export PYTHONUNBUFFERED=1

CONFIG=${1:-configs/chpc_grpo.yaml}
CHECKPOINT=${2:-chpc/results/checkpoints/grpo/final}
RESULTS_DIR=${3:-chpc/results/eval/latest}

echo "=== Evaluation ==="
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Results:    $RESULTS_DIR"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:   $(date)"
echo "========================"

python scripts/eval.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --output-dir "$RESULTS_DIR"

echo "=== Done: $(date) ==="
