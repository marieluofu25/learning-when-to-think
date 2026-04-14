#!/usr/bin/env bash
# Single CHPC entrypoint: submit Slurm jobs with artifacts under chpc/results/
#
# Usage (from repo root):
#   bash chpc/run_chpc.sh train-grpo [config] [output_dir]
#   bash chpc/run_chpc.sh eval [config] [checkpoint] [results_dir]
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
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export CHPC_REPO="$REPO_ROOT"

mkdir -p chpc/results/logs chpc/results/checkpoints chpc/results/eval \
  chpc/results/sft chpc/results/dpo

CHPC_ACCOUNT="${CHPC_ACCOUNT:-cs6966}"
CHPC_PARTITION="${CHPC_PARTITION:-granite-gpu-guest}"
CHPC_QOS="${CHPC_QOS:-granite-gpu-guest}"

usage() {
  echo "Usage: bash chpc/run_chpc.sh <train-grpo|eval|train-sft|train-dpo|run-all> [args...]" >&2
  echo "See chpc/README.md" >&2
  exit 1
}

CMD="${1:-}"
shift || true

case "$CMD" in
  train-grpo)
    sbatch --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
      --export=ALL,CHPC_REPO="$REPO_ROOT" \
      chpc/train_grpo.slurm \
      "${1:-configs/chpc_grpo.yaml}" \
      "${2:-chpc/results/checkpoints/grpo}"
    ;;
  eval)
    sbatch --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
      --export=ALL,CHPC_REPO="$REPO_ROOT" \
      chpc/eval.slurm \
      "${1:-configs/chpc_grpo.yaml}" \
      "${2:-chpc/results/checkpoints/grpo/final}" \
      "${3:-chpc/results/eval/latest}"
    ;;
  train-sft)
    sbatch --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
      --export=ALL,CHPC_REPO="$REPO_ROOT" \
      chpc/train_sft.slurm \
      "${1:-configs/chpc_sft_3action.yaml}" \
      "${2:-chpc/results/sft/3action}"
    ;;
  train-dpo)
    sbatch --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS" \
      --export=ALL,CHPC_REPO="$REPO_ROOT" \
      chpc/train_dpo.slurm \
      "${1:-configs/chpc_dpo.yaml}"
    ;;
  run-all)
    SFT_CFG="${RUN_ALL_SFT_CFG:-configs/chpc_sft_3action.yaml}"
    SFT_OUT="${RUN_ALL_SFT_OUT:-chpc/results/sft/3action}"
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

    SBATCH_BASE=(--account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" --qos="$CHPC_QOS"
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
      J_SFT=$(sbatch --parsable "${SBATCH_BASE[@]}" chpc/train_sft.slurm "$SFT_CFG" "$SFT_OUT")
      echo "Submitted train-sft job $J_SFT"
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
