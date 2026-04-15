#!/usr/bin/env bash
# Single CHPC entrypoint: submit Slurm jobs with artifacts under chpc/results/
#
# Usage (from repo root):
#   bash chpc/run_chpc.sh train-grpo [config] [output_dir]
#   bash chpc/run_chpc.sh eval [config] [checkpoint] [results_dir]
#   bash chpc/run_chpc.sh build-rollouts-data [output_jsonl] [config_yaml]
#   bash chpc/run_chpc.sh build-sft-data [rollouts_jsonl] [output_jsonl]
#   bash chpc/run_chpc.sh train-sft [config] [output_dir]
#   bash chpc/run_chpc.sh train-dpo [config]
#   bash chpc/run_chpc.sh run-all
#
# run-all submits four jobs: GRPO and SFT start in parallel; DPO waits for SFT;
# eval waits for GRPO (eval uses the GRPO checkpoint). Override paths with env:
#   RUN_ALL_SFT_CFG RUN_ALL_SFT_OUT RUN_ALL_GRPO_CFG RUN_ALL_GRPO_OUT
#   RUN_ALL_DPO_CFG RUN_ALL_EVAL_CFG RUN_ALL_EVAL_CKPT RUN_ALL_EVAL_OUT
# Skip branches: RUN_ALL_SKIP_SFT=1, RUN_ALL_SKIP_DPO=1, RUN_ALL_SKIP_GRPO=1,
#   RUN_ALL_SKIP_EVAL=1 (dependencies adjust automatically).
#
# Override Slurm account/partition (defaults: UU Granite guest examples):
#   export CHPC_ACCOUNT=cs6966 CHPC_PARTITION=granite-gpu-guest CHPC_QOS=granite-gpu-guest
# Python venv on compute nodes (default: ~/venvs/teaching-llms-errors); override:
#   export CHPC_VENV=/path/to/venv
# With sbatch --export=ALL, CHPC_VENV is inherited if set before running this script.
#
# Compute nodes: Slurm copies the batch script to spool, so *.slurm sources
# chpc/slurm_prologue.sh via CHPC_REPO (exported below) or SLURM_SUBMIT_DIR, not
# dirname(BASH_SOURCE). Prologue then sets REPO_ROOT from pyproject.toml.
#
# train-sft / run-all (SFT branch): if sft_data_path JSONL is missing, this script
# submits build jobs first (rollouts -> sft-jsonl) and chains dependencies.
# Use RUN_ALL_SKIP_SFT=1 if you only run GRPO+eval.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export CHPC_REPO="$REPO_ROOT"

# Optional: per-user Slurm account/partition/qos (and HF_HOME, etc.) without retyping.
if [[ -f "$SCRIPT_DIR/env_local.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env_local.sh"
fi

mkdir -p chpc/results/logs chpc/results/checkpoints chpc/results/eval \
  chpc/results/sft chpc/results/dpo

CHPC_ACCOUNT="${CHPC_ACCOUNT:-cs6966}"
CHPC_PARTITION="${CHPC_PARTITION:-granite-gpu-guest}"
CHPC_QOS="${CHPC_QOS:-granite-gpu-guest}"

_resolve_sft_jsonl_path() {
  # Prints resolved absolute JSONL path from YAML sft_data_path.
  # Exits 0 if path printed; non-zero on error.
  local cfg="${1:-configs/chpc_sft_3action.yaml}"
  python3 - "$REPO_ROOT" "$cfg" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "lwtt: PyYAML not importable; activate project venv (pip install -e .) and retry.",
        file=sys.stderr,
    )
    sys.exit(1)

root = Path(sys.argv[1])
cfg_path = Path(sys.argv[2])
if not cfg_path.is_absolute():
    cfg_path = root / cfg_path
if not cfg_path.is_file():
    print(f"lwtt: SFT config not found: {cfg_path}", file=sys.stderr)
    sys.exit(1)
with open(cfg_path, encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
raw = data.get("sft_data_path")
if not raw:
    print("lwtt: sft_data_path missing in YAML", file=sys.stderr)
    sys.exit(1)
p = Path(raw)
if not p.is_absolute():
    p = root / p
print(str(p))
PY
}

_resolve_rollouts_jsonl_path() {
  # Prints resolved absolute rollouts JSONL path from config output key.
  # Exits 0 if path printed; non-zero on error.
  local cfg="${1:-configs/generate_rollouts_qwen25.yaml}"
  python3 - "$REPO_ROOT" "$cfg" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "lwtt: PyYAML not importable; activate project venv (pip install -e .) and retry.",
        file=sys.stderr,
    )
    sys.exit(1)

root = Path(sys.argv[1])
cfg_path = Path(sys.argv[2])
if not cfg_path.is_absolute():
    cfg_path = root / cfg_path
if not cfg_path.is_file():
    print(f"lwtt: rollouts config not found: {cfg_path}", file=sys.stderr)
    sys.exit(1)
with open(cfg_path, encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
raw = data.get("output")
if not raw:
    print("lwtt: output missing in rollouts YAML", file=sys.stderr)
    sys.exit(1)
p = Path(raw)
if not p.is_absolute():
    p = root / p
print(str(p))
PY
}

_sft_jsonl_exists() {
  local cfg="${1:-configs/chpc_sft_3action.yaml}"
  local p
  p="$(_resolve_sft_jsonl_path "$cfg")" || return 1
  [[ -f "$p" ]]
}

_rollouts_jsonl_exists() {
  local cfg="${1:-configs/generate_rollouts_qwen25.yaml}"
  local p
  p="$(_resolve_rollouts_jsonl_path "$cfg")" || return 1
  [[ -f "$p" ]]
}

usage() {
  echo "Usage: bash chpc/run_chpc.sh <train-grpo|eval|build-rollouts-data|build-sft-data|train-sft|train-dpo|run-all> [args...]" >&2
  echo "See chpc/README.md" >&2
  exit 1
}

CMD="${1:-}"
shift || true

# --chdir: job cwd = repo root (Slurm 20.11+). Drop if your sbatch rejects it.
SBATCH_CHDIR=(--chdir="$REPO_ROOT")

case "$CMD" in
  train-grpo)
    sbatch "${SBATCH_CHDIR[@]}" --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
      --export=ALL,CHPC_REPO="$REPO_ROOT" \
      chpc/train_grpo.slurm \
      "${1:-configs/chpc_grpo.yaml}" \
      "${2:-chpc/results/checkpoints/grpo}"
    ;;
  eval)
    sbatch "${SBATCH_CHDIR[@]}" --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
      --export=ALL,CHPC_REPO="$REPO_ROOT" \
      chpc/eval.slurm \
      "${1:-configs/chpc_grpo.yaml}" \
      "${2:-chpc/results/checkpoints/grpo/final}" \
      "${3:-chpc/results/eval/latest}"
    ;;
  build-rollouts-data)
    # Build grouped rollouts JSONL used by generate_sft_3action.
    sbatch "${SBATCH_CHDIR[@]}" --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
      --export=ALL,CHPC_REPO="$REPO_ROOT" \
      chpc/build_rollouts_data.slurm \
      "${1:-data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl}" \
      "${2:-configs/generate_rollouts_qwen25.yaml}"
    ;;
  build-sft-data)
    # Build the JSONL used by SFT (runs on GPU via vLLM rewriter).
    sbatch "${SBATCH_CHDIR[@]}" --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
      --export=ALL,CHPC_REPO="$REPO_ROOT" \
      chpc/build_sft_3action_data.slurm \
      "${1:-data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl}" \
      "${2:-data/sft_3action_math_train.jsonl}"
    ;;
  train-sft)
    SFT_CFG="${1:-configs/chpc_sft_3action.yaml}"
    SFT_OUT="${2:-chpc/results/sft/3action}"
    ROLLOUTS_CFG="${RUN_ALL_ROLLOUTS_CFG:-configs/generate_rollouts_qwen25.yaml}"
    ROLLOUTS_OUT="${RUN_ALL_ROLLOUTS_OUT:-}"
    if _sft_jsonl_exists "$SFT_CFG"; then
      sbatch "${SBATCH_CHDIR[@]}" --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
        --export=ALL,CHPC_REPO="$REPO_ROOT" \
        chpc/train_sft.slurm \
        "$SFT_CFG" \
        "$SFT_OUT"
    else
      [[ -n "$ROLLOUTS_OUT" ]] || ROLLOUTS_OUT="$(_resolve_rollouts_jsonl_path "$ROLLOUTS_CFG")"
      SFT_JSONL="$(_resolve_sft_jsonl_path "$SFT_CFG")"
      echo "SFT JSONL missing for $SFT_CFG (${SFT_JSONL}); preparing build dependencies."
      J_BUILD_ROLLOUTS=""
      J_BUILD_SFT=""
      if _rollouts_jsonl_exists "$ROLLOUTS_CFG"; then
        echo "Rollouts JSONL exists: ${ROLLOUTS_OUT}"
      else
        J_BUILD_ROLLOUTS=$(sbatch --parsable "${SBATCH_CHDIR[@]}" --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
          --export=ALL,CHPC_REPO="$REPO_ROOT" \
          chpc/build_rollouts_data.slurm "$ROLLOUTS_OUT" "$ROLLOUTS_CFG")
        echo "Submitted build-rollouts-data job $J_BUILD_ROLLOUTS"
      fi
      if [[ -n "$J_BUILD_ROLLOUTS" ]]; then
        J_BUILD_SFT=$(sbatch --parsable "${SBATCH_CHDIR[@]}" --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
          --dependency=afterok:"$J_BUILD_ROLLOUTS" \
          --export=ALL,CHPC_REPO="$REPO_ROOT" \
          chpc/build_sft_3action_data.slurm "$ROLLOUTS_OUT" "$SFT_JSONL")
      else
        J_BUILD_SFT=$(sbatch --parsable "${SBATCH_CHDIR[@]}" --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
          --export=ALL,CHPC_REPO="$REPO_ROOT" \
          chpc/build_sft_3action_data.slurm "$ROLLOUTS_OUT" "$SFT_JSONL")
      fi
      echo "Submitted build-sft-data job $J_BUILD_SFT"
      sbatch "${SBATCH_CHDIR[@]}" --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
        --dependency=afterok:"$J_BUILD_SFT" \
        --export=ALL,CHPC_REPO="$REPO_ROOT" \
        chpc/train_sft.slurm \
        "$SFT_CFG" \
        "$SFT_OUT"
    fi
    ;;
  train-dpo)
    sbatch "${SBATCH_CHDIR[@]}" --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
      --export=ALL,CHPC_REPO="$REPO_ROOT" \
      chpc/train_dpo.slurm \
      "${1:-configs/chpc_dpo.yaml}"
    ;;
  run-all)
    SFT_CFG="${RUN_ALL_SFT_CFG:-configs/chpc_sft_3action.yaml}"
    SFT_OUT="${RUN_ALL_SFT_OUT:-chpc/results/sft/3action}"
    ROLLOUTS_CFG="${RUN_ALL_ROLLOUTS_CFG:-configs/generate_rollouts_qwen25.yaml}"
    ROLLOUTS_OUT="${RUN_ALL_ROLLOUTS_OUT:-}"
    GRPO_CFG="${RUN_ALL_GRPO_CFG:-configs/chpc_grpo.yaml}"
    GRPO_OUT="${RUN_ALL_GRPO_OUT:-chpc/results/checkpoints/grpo}"
    DPO_CFG="${RUN_ALL_DPO_CFG:-configs/chpc_dpo.yaml}"
    EVAL_CFG="${RUN_ALL_EVAL_CFG:-configs/chpc_grpo.yaml}"
    EVAL_CKPT="${RUN_ALL_EVAL_CKPT:-chpc/results/checkpoints/grpo/final}"
    EVAL_OUT="${RUN_ALL_EVAL_OUT:-chpc/results/eval/latest}"

    SKIP_SFT="${RUN_ALL_SKIP_SFT:-0}"
    SKIP_DPO="${RUN_ALL_SKIP_DPO:-0}"
    SKIP_GRPO="${RUN_ALL_SKIP_GRPO:-0}"
    SKIP_EVAL="${RUN_ALL_SKIP_EVAL:-0}"

    SBATCH_BASE=(--chdir="$REPO_ROOT" --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS"
      --export=ALL,CHPC_REPO="$REPO_ROOT")

    J_SFT=""
    J_GRPO=""
    if [[ "$SKIP_GRPO" != "1" ]]; then
      J_GRPO=$(sbatch --parsable "${SBATCH_BASE[@]}" chpc/train_grpo.slurm "$GRPO_CFG" "$GRPO_OUT")
      echo "Submitted train-grpo job $J_GRPO"
    else
      echo "Skipped train-grpo (RUN_ALL_SKIP_GRPO=1)"
    fi

    if [[ "$SKIP_SFT" != "1" ]]; then
      if _sft_jsonl_exists "$SFT_CFG"; then
        J_SFT=$(sbatch --parsable "${SBATCH_BASE[@]}" chpc/train_sft.slurm "$SFT_CFG" "$SFT_OUT")
        echo "Submitted train-sft job $J_SFT"
      else
        [[ -n "$ROLLOUTS_OUT" ]] || ROLLOUTS_OUT="$(_resolve_rollouts_jsonl_path "$ROLLOUTS_CFG")"
        SFT_JSONL="$(_resolve_sft_jsonl_path "$SFT_CFG")"
        echo "SFT JSONL missing for $SFT_CFG (${SFT_JSONL}); preparing build dependencies."
        J_BUILD_ROLLOUTS=""
        J_BUILD_SFT=""
        if _rollouts_jsonl_exists "$ROLLOUTS_CFG"; then
          echo "Rollouts JSONL exists: ${ROLLOUTS_OUT}"
        else
          J_BUILD_ROLLOUTS=$(sbatch --parsable "${SBATCH_BASE[@]}" chpc/build_rollouts_data.slurm "$ROLLOUTS_OUT" "$ROLLOUTS_CFG")
          echo "Submitted build-rollouts-data job $J_BUILD_ROLLOUTS"
        fi
        if [[ -n "$J_BUILD_ROLLOUTS" ]]; then
          J_BUILD_SFT=$(sbatch --parsable "${SBATCH_BASE[@]}" --dependency=afterok:"$J_BUILD_ROLLOUTS" \
            chpc/build_sft_3action_data.slurm "$ROLLOUTS_OUT" "$SFT_JSONL")
          echo "Submitted build-sft-data job $J_BUILD_SFT (after rollouts $J_BUILD_ROLLOUTS)"
        else
          J_BUILD_SFT=$(sbatch --parsable "${SBATCH_BASE[@]}" chpc/build_sft_3action_data.slurm "$ROLLOUTS_OUT" "$SFT_JSONL")
          echo "Submitted build-sft-data job $J_BUILD_SFT"
        fi
        J_SFT=$(sbatch --parsable "${SBATCH_BASE[@]}" --dependency=afterok:"$J_BUILD_SFT" \
          chpc/train_sft.slurm "$SFT_CFG" "$SFT_OUT")
        echo "Submitted train-sft job $J_SFT (after build $J_BUILD_SFT)"
      fi
    else
      echo "Skipped train-sft (RUN_ALL_SKIP_SFT=1)"
    fi

    if [[ "$SKIP_DPO" != "1" ]]; then
      if [[ -n "$J_SFT" ]]; then
        J_DPO=$(sbatch --parsable "${SBATCH_BASE[@]}" --dependency=afterok:"$J_SFT" \
          chpc/train_dpo.slurm "$DPO_CFG")
        echo "Submitted train-dpo job $J_DPO (after SFT $J_SFT)"
      else
        echo "WARN: train-dpo skipped (no SFT job; set RUN_ALL_SKIP_SFT=0 or run train-dpo manually)" >&2
      fi
    else
      echo "Skipped train-dpo (RUN_ALL_SKIP_DPO=1)"
    fi

    if [[ "$SKIP_EVAL" != "1" ]]; then
      if [[ -n "$J_GRPO" ]]; then
        J_EVAL=$(sbatch --parsable "${SBATCH_BASE[@]}" --dependency=afterok:"$J_GRPO" \
          chpc/eval.slurm "$EVAL_CFG" "$EVAL_CKPT" "$EVAL_OUT")
        echo "Submitted eval job $J_EVAL (after GRPO $J_GRPO)"
      else
        echo "WARN: eval skipped (no GRPO job; set RUN_ALL_SKIP_GRPO=0 or run eval manually)" >&2
      fi
    else
      echo "Skipped eval (RUN_ALL_SKIP_EVAL=1)"
    fi
    ;;
  ""|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    usage
    ;;
esac
