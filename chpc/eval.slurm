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
# Slurm copies this script to spool; dirname is not repo chpc/. Resolve prologue from env or submit dir.
_SLURM_PROLOGUE=""
for _d in "${CHPC_REPO:-}" "${SLURM_SUBMIT_DIR:-}"; do
  if [[ -n "$_d" && -f "$_d/chpc/slurm_prologue.sh" ]]; then
    _SLURM_PROLOGUE="$_d/chpc/slurm_prologue.sh"
    break
  fi
done
if [[ -z "$_SLURM_PROLOGUE" ]]; then
  _chpc_dir="$(cd "$(dirname "${CHPC_SLURM_SCRIPT}")" && pwd)"
  [[ -f "${_chpc_dir}/slurm_prologue.sh" ]] && _SLURM_PROLOGUE="${_chpc_dir}/slurm_prologue.sh"
fi
if [[ -z "$_SLURM_PROLOGUE" ]]; then
  echo "lwtt: cannot find chpc/slurm_prologue.sh (use bash chpc/run_chpc.sh or set CHPC_REPO)." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$_SLURM_PROLOGUE"
unset _d _chpc_dir _SLURM_PROLOGUE

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
