# Running on CHPC (University of Utah)

This guide is for running the CHPC GRPO training/eval in this repo.
Goal: copy/paste commands and only fill in `uNID`, `account`, `partition`, and (if needed) `qos`.

## 0. Connect to CHPC

```bash
ssh <uNID>@granite.chpc.utah.edu
```

If your course/project uses a different cluster (e.g. `notchpeak`/`kingspeak`), replace the hostname accordingly.

## 0.1 Check which GPU partitions/accounts you can use (recommended)

Run:

```bash
mychpc batch
```

This shows valid combinations of `--account` / `--partition` (and `--qos` if required).
Use the values you are allowed to access.

## 0.2 (Optional) Quick checks

```bash
sacctmgr show assoc user=$USER format=account%20,partition%30
sinfo -p <your-partition> -o "%N %G %m %f"
scontrol show node <node-name>
```

## 1. Load modules

Module versions depend on the cluster’s available environment. Start with:

```bash
module load cuda/12.2
module load python/3.11
```

If those don’t exist, run:

```bash
module avail cuda
module avail python
```

## 2. Set up the repo (clone or copy)

### Option A: `git clone` (simplest)
```bash
cd ~
git clone <your-repo-url> learning-when-to-think
cd learning-when-to-think
```

### Option B: copy with `rsync` (if repo is already on your machine)
```bash
rsync -avz . <uNID>@granite.chpc.utah.edu:~/learning-when-to-think/
```

## 2.1 Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

If `uv` is available:
```bash
uv sync
```

## 3. HuggingFace cache directory (important)

CHPC home dirs have limited space. Point the cache to scratch:

```bash
export HF_HOME=/scratch/general/vast/$USER/hf_cache
mkdir -p "$HF_HOME"
```

## 4. Training on CHPC (SLURM)

Two provided configs:
- `configs/chpc_debug.yaml` (quick sanity check)
- `configs/chpc_grpo.yaml` (larger run per the plan)

### 4.1 Debug run

First, ensure your `--account` and `--partition` are valid via `mychpc batch`.

```bash
sbatch --account=<your-account> --partition=<your-partition> \
  chpc/train_grpo.slurm configs/chpc_debug.yaml
```

### 4.2 Full training run

```bash
sbatch --account=<your-account> --partition=<your-partition> \
  chpc/train_grpo.slurm configs/chpc_grpo.yaml
```

Notes:
- Our SLURM scripts include defaults, but passing `--account/--partition` on the `sbatch` command overrides them.
- Output goes to `slurm-%j.out` in your submission directory.

## 5. Monitor

```bash
squeue -u $USER
tail -f slurm-<jobid>.out
```

Sanity check that GPU was used:
- The SLURM script prints `nvidia-smi` GPU name at startup.

## 6. Evaluation on CHPC (SLURM)

```bash
sbatch --account=<your-account> --partition=<your-partition> \
  chpc/eval.slurm
```

## 7. Download results

From your local machine:

```bash
rsync -avz <uNID>@granite.chpc.utah.edu:~/learning-when-to-think/results/ ./results/
rsync -avz <uNID>@granite.chpc.utah.edu:~/learning-when-to-think/checkpoints/ ./checkpoints/
```

## 8. Optional interactive debugging (faster than re-submitting jobs)

```bash
srun --account=<your-account> --partition=<your-partition> --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time=01:00:00 --pty bash
```

Then:

```bash
cd ~/learning-when-to-think
source .venv/bin/activate

export HF_HOME=/scratch/general/vast/$USER/hf_cache
mkdir -p "$HF_HOME"

python scripts/train.py --config configs/chpc_debug.yaml --output-dir checkpoints/grpo_debug
```

## Tips
- If you get OOM: reduce `batch_size` (YAML) and/or increase `--mem` (SLURM).
- If you get killed: increase `--time` in the `.slurm` file.
