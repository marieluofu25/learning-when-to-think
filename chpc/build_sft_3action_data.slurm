#!/bin/bash
# Build SFT warmup JSONL (3-action tokens) on CHPC.
# Args: [rollouts_jsonl] [output_jsonl]
#
# Defaults match configs/chpc_sft_3action.yaml:
#   rollouts: data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl
#   output:   data/sft_3action_math_train.jsonl
#
# Optional env overrides:
#   SFT_REWRITER   (default: Qwen/Qwen3-14B)
#   SFT_TP         (default: 1)
#   SFT_GPU_MEM    (default: 0.90)
#   SFT_MAX_TOKENS (default: 4096)
#   SFT_CLEAN_RATIO(default: 0.6)
#   SFT_LIMIT      (default: unset)
#
#SBATCH --job-name=build-sft-3action-data
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=chpc/results/logs/slurm-build-sft-data-%j.out

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

# vLLM may call int(CUDA_VISIBLE_DEVICES); Slurm sometimes exports a MIG UUID here.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

ROLLOUTS_JSONL="${1:-data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl}"
OUTPUT_JSONL="${2:-data/sft_3action_math_train.jsonl}"

SFT_REWRITER="${SFT_REWRITER:-Qwen/Qwen3-14B}"
SFT_TP="${SFT_TP:-1}"
SFT_GPU_MEM="${SFT_GPU_MEM:-0.90}"
SFT_MAX_TOKENS="${SFT_MAX_TOKENS:-4096}"
SFT_CLEAN_RATIO="${SFT_CLEAN_RATIO:-0.6}"
SFT_LIMIT="${SFT_LIMIT:-}"

echo "=== Build SFT data (3-action) ==="
echo "REPO_ROOT: ${REPO_ROOT}"
echo "Rollouts:  ${ROLLOUTS_JSONL}"
echo "Output:    ${OUTPUT_JSONL}"
echo "Rewriter:  ${SFT_REWRITER}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"

if [[ ! -f "${ROLLOUTS_JSONL}" ]]; then
  echo "lwtt: rollouts JSONL not found: ${ROLLOUTS_JSONL}" >&2
  echo "lwtt: build rollouts first (see scripts/generate_rollouts_pivot.py) or pass a different path." >&2
  exit 1
fi

args=(
  -m scripts.generate_sft_3action
  --rollouts "${ROLLOUTS_JSONL}"
  --output "${OUTPUT_JSONL}"
  --rewriter "${SFT_REWRITER}"
  --tp "${SFT_TP}"
  --gpu-mem "${SFT_GPU_MEM}"
  --max-tokens "${SFT_MAX_TOKENS}"
  --clean-ratio "${SFT_CLEAN_RATIO}"
)
if [[ -n "${SFT_LIMIT}" ]]; then
  args+=(--limit "${SFT_LIMIT}")
fi

python "${args[@]}"

echo "=== Done: $(date) ==="
