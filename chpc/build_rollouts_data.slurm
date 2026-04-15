#!/bin/bash
# Build grouped rollout JSONL on CHPC.
# Args: [output_jsonl] [config_yaml]
#
# Defaults:
#   output: data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl
#   config: configs/generate_rollouts_qwen25.yaml
#
# Optional env overrides:
#   ROLLOUT_MODEL
#   ROLLOUT_SUBSET
#   ROLLOUT_NUM_SAMPLES
#   ROLLOUT_MAX_TOKENS
#   ROLLOUT_TEMPERATURE
#   ROLLOUT_TP
#   ROLLOUT_GPU_MEM
#   ROLLOUT_SEED
#
#SBATCH --job-name=build-rollouts-data
#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=chpc/results/logs/slurm-build-rollouts-%j.out

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

OUTPUT_JSONL="${1:-data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl}"
CONFIG_YAML="${2:-configs/generate_rollouts_qwen25.yaml}"

echo "=== Build rollouts data ==="
echo "REPO_ROOT: ${REPO_ROOT}"
echo "Config:    ${CONFIG_YAML}"
echo "Output:    ${OUTPUT_JSONL}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

args=(
  -m scripts.generate_rollouts_pivot
  --config "${CONFIG_YAML}"
  --output "${OUTPUT_JSONL}"
)

[[ -n "${ROLLOUT_MODEL:-}" ]] && args+=(--model "${ROLLOUT_MODEL}")
[[ -n "${ROLLOUT_SUBSET:-}" ]] && args+=(--subset "${ROLLOUT_SUBSET}")
[[ -n "${ROLLOUT_NUM_SAMPLES:-}" ]] && args+=(--num-samples "${ROLLOUT_NUM_SAMPLES}")
[[ -n "${ROLLOUT_MAX_TOKENS:-}" ]] && args+=(--max-tokens "${ROLLOUT_MAX_TOKENS}")
[[ -n "${ROLLOUT_TEMPERATURE:-}" ]] && args+=(--temperature "${ROLLOUT_TEMPERATURE}")
[[ -n "${ROLLOUT_TP:-}" ]] && args+=(--tp "${ROLLOUT_TP}")
[[ -n "${ROLLOUT_GPU_MEM:-}" ]] && args+=(--gpu-mem "${ROLLOUT_GPU_MEM}")
[[ -n "${ROLLOUT_SEED:-}" ]] && args+=(--seed "${ROLLOUT_SEED}")

python "${args[@]}"

echo "=== Done: $(date) ==="
