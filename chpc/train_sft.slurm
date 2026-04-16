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

CONFIG=${1:-configs/chpc_sft_3action.yaml}
OUTPUT_DIR=${2:-chpc/results/sft/3action}

echo "=== SFT (3-action JSONL) ==="
echo "REPO_ROOT: ${REPO_ROOT}"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Node:   $(hostname)"
echo "Account:   ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "QOS:       ${SLURM_JOB_QOS:-${SLURM_QOS:-unknown}}"
echo "Start:  $(date)"

python scripts/train_sft_3action.py --config "$CONFIG" --output-dir "$OUTPUT_DIR"

echo "=== Done: $(date) ==="
