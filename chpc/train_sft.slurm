#!/bin/bash
# Args: [config] [output_dir]
#SBATCH --job-name=sft-3action
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=chpc/results/logs/slurm-train-sft-%j.out

set -euo pipefail
export CHPC_SLURM_SCRIPT="${BASH_SOURCE[0]}"
source "$(cd "$(dirname "${CHPC_SLURM_SCRIPT}")" && pwd)/slurm_prologue.sh"

CONFIG=${1:-configs/chpc_sft_3action.yaml}
OUTPUT_DIR=${2:-chpc/results/sft/3action}

echo "=== SFT (3-action JSONL) ==="
echo "REPO_ROOT: ${REPO_ROOT}"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Node:   $(hostname)"
echo "Start:  $(date)"

python scripts/train_sft_3action.py --config "$CONFIG" --output-dir "$OUTPUT_DIR"

echo "=== Done: $(date) ==="
