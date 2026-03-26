# Running on CHPC (University of Utah)

## 1. Connect

```bash
ssh <uNID>@notchpeak.chpc.utah.edu
```

## 2. Check GPU availability

```bash
# See which partitions you have access to
sacctmgr show assoc user=$USER format=account%20,partition%30

# Check GPU nodes in your partition
sinfo -p <your-partition> -o "%N %G %m %f"

# Inspect a specific node
scontrol show node <node-name>
```

## 3. Load modules

```bash
module load cuda/12.2
module load python/3.11
```

> Exact versions depend on what's available. Run `module avail cuda` and `module avail python` to check.

## 4. Clone & set up the project

```bash
cd ~
git clone <your-repo-url> learning-when-to-think
cd learning-when-to-think

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e .
```

If `uv` is available:

```bash
uv sync
```

## 5. Set HuggingFace cache directory

CHPC home dirs have limited space. Point the cache to your scratch space:

```bash
export HF_HOME=/scratch/general/vast/$USER/hf_cache
mkdir -p $HF_HOME
```

Add this to your `~/.bashrc` or put it in the SLURM scripts (already included in the provided scripts).

## 6. Submit training job

```bash
# Quick debug run (small data, 1 epoch)
sbatch chpc/train_grpo.slurm --config configs/chpc_debug.yaml

# Full training run
sbatch chpc/train_grpo.slurm
```

Monitor:

```bash
squeue -u $USER                   # Check job status
tail -f slurm-<jobid>.out         # Watch live output
```

## 7. Submit evaluation job

```bash
sbatch chpc/eval.slurm
```

## 8. Download results

From your local machine:

```bash
rsync -avz <uNID>@notchpeak.chpc.utah.edu:~/learning-when-to-think/results/ ./results/
rsync -avz <uNID>@notchpeak.chpc.utah.edu:~/learning-when-to-think/checkpoints/ ./checkpoints/
```

## Tips

- **Walltime**: If the job gets killed, increase `--time` in the `.slurm` file.
- **Memory**: If OOM, reduce `batch_size` in the YAML config or increase `--mem` in SLURM.
- **Logs**: All stdout/stderr goes to `slurm-<jobid>.out` by default.
- **Interactive session** (for debugging):
  ```bash
  salloc --account=<your-account> --partition=<your-partition> --gres=gpu:1 --mem=32G --time=1:00:00
  ```
