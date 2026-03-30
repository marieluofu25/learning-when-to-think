# Running on CHPC (University of Utah)

Copy/paste workflow for **Qwen2.5-Math-7B-Instruct** + **4-bit QLoRA** GRPO training and eval. Adjust `uNID`, account, and partition using `mychpc batch` (or your course instructions).

## Connect

```bash
ssh <uNID>@granite.chpc.utah.edu
```

Use your assigned cluster hostname if different (e.g. notchpeak login node).

## Valid account / partition

```bash
mychpc batch
```

Pick a GPU partition you are allowed to use. Submit with explicit flags if the defaults in the `.slurm` files are wrong:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  chpc/train_grpo.slurm configs/chpc_debug.yaml checkpoints/grpo_debug
```

## Modules

```bash
module load cuda/12.2
module load python/3.11
```

If versions differ, run `module avail cuda` and `module avail python`.

## Repo and environment

```bash
cd ~
git clone <your-repo-url> learning-when-to-think
cd learning-when-to-think

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

`bitsandbytes` is required for **QLoRA** (`use_qlora: true` in CHPC configs). If install fails on the node, open a CHPC ticket or use a module-provided CUDA stack that matches the wheels.

## Hugging Face cache (use scratch)

Home quotas are small; point the cache at scratch:

```bash
export HF_HOME=/scratch/general/vast/$USER/hf_cache
mkdir -p "$HF_HOME"
```

Gated models (e.g. Llama) need a token:

```bash
export HF_TOKEN=...
```

## Training (SLURM)

Configs:

- `configs/chpc_debug.yaml` — short sanity run (same **Math-7B** backbone, small subset).
- `configs/chpc_grpo.yaml` — longer GSM8K run.
- `configs/chpc_qwen25_7b_instruct.yaml` — optional **Qwen2.5-7B-Instruct** track for stronger code/general transfer.

```bash
# Debug
sbatch chpc/train_grpo.slurm configs/chpc_debug.yaml checkpoints/grpo_debug

# Full (edit #SBATCH account/partition or override on sbatch line)
sbatch chpc/train_grpo.slurm configs/chpc_grpo.yaml checkpoints/grpo_chpc
```

Positional arguments to the batch script: `[config_path] [output_dir]`.

## Evaluation (SLURM)

Arguments: `[config] [checkpoint] [results_dir]`.

```bash
sbatch chpc/eval.slurm \
  configs/chpc_grpo.yaml \
  checkpoints/grpo_chpc/final \
  results/eval_chpc
```

HumanEval example (config must set `dataset: humaneval`):

```bash
sbatch chpc/eval.slurm configs/eval_humaneval.yaml checkpoints/grpo_chpc/final results/eval_he
```

## Monitor

```bash
squeue -u $USER
tail -f slurm-<jobid>.out
```

## Pull results back

```bash
rsync -avz <uNID>@granite.chpc.utah.edu:~/learning-when-to-think/results/ ./results/
rsync -avz <uNID>@granite.chpc.utah.edu:~/learning-when-to-think/checkpoints/ ./checkpoints/
```

## Interactive GPU shell

```bash
srun --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time=01:00:00 --pty bash
```

Then activate `.venv`, set `HF_HOME`, and run e.g. `python scripts/train.py --config configs/chpc_debug.yaml --output-dir checkpoints/grpo_debug`.

## OOM / stability

- Reduce `batch_size`, `num_rollouts_per_problem`, or `max_tokens_per_step` in the YAML.
- Keep `use_qlora: true` for Math-7B on a single GPU.
- Increase `#SBATCH --mem` before shrinking the model.
