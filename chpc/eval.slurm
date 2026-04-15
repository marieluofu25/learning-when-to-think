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

set -euo pipefail
export CHPC_SLURM_SCRIPT="${BASH_SOURCE[0]}"
source "$(cd "$(dirname "${CHPC_SLURM_SCRIPT}")" && pwd)/slurm_prologue.sh"

CONFIG=${1:-configs/chpc_grpo.yaml}
CHECKPOINT=${2:-chpc/results/checkpoints/grpo/final}
RESULTS_DIR=${3:-chpc/results/eval/latest}

echo "=== Evaluation ==="
echo "REPO_ROOT:  ${REPO_ROOT}"
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Results:    $RESULTS_DIR"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:   $(date)"
echo "========================"

python scripts/eval.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --output-dir "$RESULTS_DIR"

echo "=== Done: $(date) ==="
