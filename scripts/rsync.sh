#!/usr/bin/env bash
# Rsync chpc/, configs/, and data/ between this machine and CHPC.
# Run from your local terminal (or anywhere with rsync + SSH).
#
# Defaults match chpc/README.md (Granite + repo in home as learning-when-to-think).
#
# Usage:
#   bash scripts/rsync_chpc_dirs.sh              # push local -> remote
#   bash scripts/rsync_chpc_dirs.sh push
#   bash scripts/rsync_chpc_dirs.sh pull       # remote -> local
#   RSYNC_EXTRA="--dry-run" bash scripts/rsync_chpc_dirs.sh push
#
# Required once per shell (or export in ~/.bashrc):
#   export CHPC_USER=u1234567
#
# Optional overrides:
#   CHPC_HOST=granite.chpc.utah.edu
#   CHPC_REMOTE_ROOT=~/learning-when-to-think
#   RSYNC_EXTRA="--dry-run"   # or "-n", extra rsync flags after -avz
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

CHPC_HOST="${CHPC_HOST:-granite.chpc.utah.edu}"
CHPC_REMOTE_ROOT="${CHPC_REMOTE_ROOT:-~/llm_finetune_demo/advanced_ai/learning-when-to-think}"
RSYNC_BASE=(rsync -avz --human-readable --progress)

if [[ -n "${RSYNC_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  RSYNC_BASE+=($RSYNC_EXTRA)
fi

if [[ -z "${CHPC_USER:-}" ]]; then
  echo "Set CHPC_USER to your uNID, e.g. export CHPC_USER=u1234567" >&2
  exit 1
fi

REMOTE="${CHPC_USER}@${CHPC_HOST}"
# Expand leading ~/ on remote for display only; rsync/ssh accept ~/ in paths.
REMOTE_SPEC="${REMOTE}:${CHPC_REMOTE_ROOT}"

DIRS=(chpc configs data)

DIR="${1:-push}"
case "$DIR" in
  push)
    for d in "${DIRS[@]}"; do
      if [[ ! -d "$REPO_ROOT/$d" ]]; then
        echo "Missing directory: $REPO_ROOT/$d" >&2
        exit 1
      fi
      echo "==> push $REPO_ROOT/$d/ -> ${REMOTE_SPEC}/$d/"
      "${RSYNC_BASE[@]}" "$REPO_ROOT/$d/" "${REMOTE_SPEC}/$d/"
    done
    ;;
  pull)
    for d in "${DIRS[@]}"; do
      mkdir -p "$REPO_ROOT/$d"
      echo "==> pull ${REMOTE_SPEC}/$d/ -> $REPO_ROOT/$d/"
      "${RSYNC_BASE[@]}" "${REMOTE_SPEC}/$d/" "$REPO_ROOT/$d/"
    done
    ;;
  *)
    echo "Usage: $0 [push|pull]" >&2
    exit 1
    ;;
esac

