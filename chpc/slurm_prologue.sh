#!/usr/bin/env bash
# shellcheck shell=bash
# Sourced from chpc/*.slurm after:
#   export CHPC_SLURM_SCRIPT="${BASH_SOURCE[0]}"
#
# Resolves REPO_ROOT reliably on CHPC: Slurm often does not forward the submit
# shell's env; exported CHPC_REPO from another machine/path can be wrong.
# Priority: REPO_ROOT (explicit) → SLURM_SUBMIT_DIR if it looks like the repo →
#          path derived from this script (chpc/foo.slurm → parent of chpc/).

: "${CHPC_SLURM_SCRIPT:?slurm_prologue: set CHPC_SLURM_SCRIPT to the Slurm batch script path}"

# Slurm may copy the batch script to spool; dirname(script) is NOT the repo. Prefer
# SLURM_SUBMIT_DIR / CHPC_REPO when they contain pyproject.toml; use script parent
# only as a fallback when the batch file still lives under chpc/ in the checkout.
_script="$(readlink -f "${CHPC_SLURM_SCRIPT}" 2>/dev/null || echo "${CHPC_SLURM_SCRIPT}")"
_chpc_dir="$(cd "$(dirname "$_script")" && pwd)"
_derived_root="$(cd "${_chpc_dir}/.." && pwd)"

if [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/pyproject.toml" ]]; then
  :
else
  REPO_ROOT=""
  for d in "${SLURM_SUBMIT_DIR:-}" "${CHPC_REPO:-}" "${_derived_root}"; do
    [[ -n "$d" && -f "${d}/pyproject.toml" ]] && { REPO_ROOT="$d"; break; }
  done
fi
if [[ -z "${REPO_ROOT}" ]]; then
  echo "lwtt: Cannot find repo root (no pyproject.toml)." >&2
  echo "lwtt:   SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-} CHPC_REPO=${CHPC_REPO:-} derived=${_derived_root}" >&2
  echo "lwtt: Run sbatch from the repository root (same as your other project), e.g.:" >&2
  echo "lwtt:   cd .../learning-when-to-think && sbatch chpc/train_grpo.slurm" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export CHPC_REPO="${REPO_ROOT}"

# Non-array jobs do not set SLURM_ARRAY_*; snippets under set -u must not assume they exist.
: "${SLURM_JOB_ID:=0}"
export SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# GH200 / Grace (aarch64): venv built on Granite (x86_64) cannot run here.
_drop_path_entry() {
  local drop="$1" p cleaned="" first=1
  local -a parts=()
  IFS=':' read -r -a parts <<< "${PATH:-}"
  for p in "${parts[@]}"; do
    [[ -z "$p" || "$p" == "$drop" ]] && continue
    if (( first )); then cleaned="$p"; first=0; else cleaned+=":$p"; fi
  done
  PATH="$cleaned"
  export PATH
}
if [[ "$(uname -m)" == aarch64 ]]; then
  _drop_path_entry "${HOME}/venvs/teaching-llms-errors/bin"
  if [[ "${VIRTUAL_ENV:-}" == "${HOME}/venvs/teaching-llms-errors" ]]; then
    unset VIRTUAL_ENV
  fi
fi
unset -f _drop_path_entry

# shellcheck source=/dev/null
[[ -f "${REPO_ROOT}/chpc/env_local.sh" ]] && source "${REPO_ROOT}/chpc/env_local.sh"

# Optional: export CHPC_SLURM_MODULES="cuda/12.2 python/3.11" (space-separated) in env_local.sh
if [[ -n "${CHPC_SLURM_MODULES:-}" ]] && command -v module &>/dev/null; then
  read -r -a _lwtt_mods <<< "${CHPC_SLURM_MODULES}"
  module load "${_lwtt_mods[@]}"
  unset _lwtt_mods
else
  module load cuda/12.2
  module load python/3.11
fi
hash -r

mkdir -p chpc/results/logs
CHPC_VENV="${CHPC_VENV:-$HOME/venvs/teaching-llms-errors}"
# shellcheck source=/dev/null
source "${CHPC_VENV}/bin/activate"

export HF_HOME="${HF_HOME:-/scratch/general/vast/$USER/hf_cache}"
export PYTHONUNBUFFERED=1
