#!/bin/bash
# Override account/partition if needed, e.g.:
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_PART chpc/train_grpo.slurm configs/chpc_debug.yaml
#SBATCH --job-name=grpo-train
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=chpc/results/logs/slurm-train-grpo-%j.out

# --- Modules ---
module load cuda/12.2
module load python/3.11

# --- Environment ---
cd "${CHPC_REPO:-$HOME/learning-when-to-think}"
mkdir -p chpc/results/logs
source .venv/bin/activate

export HF_HOME=/scratch/general/vast/$USER/hf_cache
export PYTHONUNBUFFERED=1

# --- Config ---
CONFIG=${1:-configs/chpc_grpo.yaml}
OUTPUT_DIR=${2:-chpc/results/checkpoints/grpo}

echo "=== GRPO Training ==="
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:  $(date)"
echo "========================"

python scripts/train.py --config "$CONFIG" --output-dir "$OUTPUT_DIR"

echo "=== Done: $(date) ==="
