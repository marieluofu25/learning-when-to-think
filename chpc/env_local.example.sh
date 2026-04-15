# Copy to chpc/env_local.sh (gitignored) and edit for your allocation.
# run_chpc.sh sources env_local.sh automatically when it exists.
#
#   cp chpc/env_local.example.sh chpc/env_local.sh

export CHPC_ACCOUNT=cs6966
export CHPC_PARTITION=soc-gpu-class-grn
export CHPC_QOS=soc-gpu-class-grn

# Optional (uncomment on CHPC):
# export CHPC_VENV="${CHPC_VENV:-$HOME/venvs/teaching-llms-errors}"
# export HF_HOME="/scratch/general/vast/$USER/hf_cache"
# Override modules loaded in slurm_prologue.sh (space-separated):
# export CHPC_SLURM_MODULES="cuda/12.2 python/3.11"
