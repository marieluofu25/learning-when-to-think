# Running on CHPC (University of Utah)

Single operator entrypoint: **`chpc/run_chpc.sh`**. All Slurm stdout and recommended checkpoint/eval paths live under **`chpc/results/`** so you can copy one tree off the cluster:

```bash
rsync -avz <uNID>@granite.chpc.utah.edu:~/learning-when-to-think/chpc/results/ ./chpc_results_backup/
```

## Connect

```bash
ssh <uNID>@granite.chpc.utah.edu
```

## One script: `run_chpc.sh`

From the **repository root** on CHPC (after `git clone`):

```bash
cd ~/learning-when-to-think

# Optional: match your Slurm allocation (see `mychpc batch`)
export CHPC_ACCOUNT=cs6966
export CHPC_PARTITION=granite-gpu-guest
export CHPC_QOS=granite-gpu-guest

bash chpc/run_chpc.sh train-grpo   [config_yaml] [output_dir]
bash chpc/run_chpc.sh eval         [config_yaml] [checkpoint_dir] [results_dir]
bash chpc/run_chpc.sh train-sft    [config_yaml] [output_dir]
bash chpc/run_chpc.sh train-dpo    [config_yaml]
```

### Run the full pipeline (`run-all`)

One command submits **four** Slurm jobs with dependencies:

- **train-grpo** and **train-sft** start **in parallel** (two GPUs if the scheduler assigns them, otherwise they queue).
- **train-dpo** starts only **after SFT succeeds** (`afterok`).
- **eval** starts only **after GRPO succeeds**; it uses the GRPO checkpoint (same defaults as `eval` above).

```bash
bash chpc/run_chpc.sh run-all
```

Optional environment overrides (all optional):

- Paths/configs: `RUN_ALL_SFT_CFG`, `RUN_ALL_SFT_OUT`, `RUN_ALL_GRPO_CFG`, `RUN_ALL_GRPO_OUT`, `RUN_ALL_DPO_CFG`, `RUN_ALL_EVAL_CFG`, `RUN_ALL_EVAL_CKPT`, `RUN_ALL_EVAL_OUT`
- Skip steps: `RUN_ALL_SKIP_SFT=1`, `RUN_ALL_SKIP_DPO=1`, `RUN_ALL_SKIP_GRPO=1`, `RUN_ALL_SKIP_EVAL=1`

Ensure `configs/chpc_dpo.yaml` points `sft_checkpoint` at the same directory as `RUN_ALL_SFT_OUT` (default: `chpc/results/sft/3action/final`).

Defaults point at `configs/chpc_grpo.yaml`, `configs/chpc_sft_3action.yaml`, and `configs/chpc_dpo.yaml` with outputs under `chpc/results/`.

`run_chpc.sh` exports **`CHPC_REPO`** to the repo root so `.slurm` scripts `cd` to the correct path even if your home layout differs.

## Modules

```bash
module load cuda/12.2
module load python/3.11
```

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

`bitsandbytes` is required for **QLoRA** (`use_qlora: true` in CHPC configs).

## Hugging Face cache (scratch)

```bash
export HF_HOME=/scratch/general/vast/$USER/hf_cache
mkdir -p "$HF_HOME"
```

Gated models need `HF_TOKEN` (do not commit tokens).

## Artifact layout

| Path | Contents |
|------|----------|
| `chpc/results/logs/` | Slurm `slurm-*.out` files |
| `chpc/results/checkpoints/grpo/` | GRPO LoRA saves (`final/`) |
| `chpc/results/eval/latest/` | `eval.py` JSON + `comparison.json` |
| `chpc/results/sft/3action/` | SFT on 3-action JSONL |
| `chpc/results/dpo/run1/` | DPO adapter |

## Configs

- `configs/chpc_grpo.yaml` — GRPO + QLoRA on MATH-500 subset  
- `configs/chpc_debug.yaml` — short sanity run  
- `configs/chpc_sft_3action.yaml` — SFT on `generate_sft_3action` JSONL  
- `configs/chpc_dpo.yaml` — DPO (**not** DAPO); ensure `sft_checkpoint` exists  

## Interactive GPU

```bash
srun --account=cs6966 --partition=granite-gpu-guest --qos=granite-gpu-guest \
  --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=01:00:00 --pty bash
```

Then activate `.venv`, set `HF_HOME`, e.g.:

```bash
python scripts/train.py --config configs/chpc_debug.yaml --output-dir chpc/results/checkpoints/debug
```

## OOM / stability

- Reduce `batch_size`, `num_rollouts_per_problem`, or `max_tokens_per_step` in the YAML.  
- Keep `use_qlora: true` for Math-7B on a single GPU.  
- Increase `#SBATCH --mem` in the `.slurm` files if needed.

## Slurm files (advanced)

`chpc/*.slurm` jobs can be submitted manually with the same `sbatch` flags as `run_chpc.sh`; defaults use **notchpeak** names in `#SBATCH` — override with `CHPC_ACCOUNT`, `CHPC_PARTITION`, and `CHPC_QOS` when calling `run_chpc.sh`, or edit the headers once per cluster.
