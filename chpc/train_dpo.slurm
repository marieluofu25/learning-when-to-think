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

set -euo pipefail
export CHPC_SLURM_SCRIPT="${BASH_SOURCE[0]}"
source "$(cd "$(dirname "${CHPC_SLURM_SCRIPT}")" && pwd)/slurm_prologue.sh"

CONFIG=${1:-configs/chpc_dpo.yaml}

echo "=== DPO training ==="
echo "REPO_ROOT: ${REPO_ROOT}"
echo "Config: $CONFIG"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

python scripts/train_dpo.py --config "$CONFIG"

echo "=== Done: $(date) ==="
