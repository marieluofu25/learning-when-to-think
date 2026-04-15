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

# Optional: save Slurm defaults so you do not retype every session
# cp chpc/env_local.example.sh chpc/env_local.sh
# (edit chpc/env_local.sh; it is gitignored — run_chpc.sh sources it when present)

# Or export manually before each run:
# export CHPC_ACCOUNT=cs6966
# export CHPC_PARTITION=granite-gpu-guest
# export CHPC_QOS=granite-gpu-guest

# Optional: override Python venv (default below). Export before sbatch so jobs inherit it.
# export CHPC_VENV=/path/to/other/venv

bash chpc/run_chpc.sh train-grpo   [config_yaml] [output_dir]
bash chpc/run_chpc.sh eval         [config_yaml] [checkpoint_dir] [results_dir]
bash chpc/run_chpc.sh build-sft-data [rollouts_jsonl] [output_jsonl]
bash chpc/run_chpc.sh train-sft    [config_yaml] [output_dir]
bash chpc/run_chpc.sh train-dpo    [config_yaml]
```

### Run the full pipeline (`run-all`)

One command submits **four** Slurm jobs with dependencies:

- **train-grpo** and **train-sft** start **in parallel** (two GPUs if the scheduler assigns them, otherwise they queue).
- **train-dpo** starts only **after SFT succeeds** (`afterok`).
- **eval** starts only **after GRPO succeeds**; it uses the GRPO checkpoint (same defaults as `eval` above).
-
- If the SFT JSONL (`sft_data_path` inside `configs/chpc_sft_3action.yaml`) is **missing**, `run-all` will **first submit** a GPU job to build it (`build-sft-data`), then submit `train-sft` with an `afterok` dependency on that build job.

```bash
bash chpc/run_chpc.sh run-all
```

Optional environment overrides (all optional):

- Paths/configs: `RUN_ALL_SFT_CFG`, `RUN_ALL_SFT_OUT`, `RUN_ALL_GRPO_CFG`, `RUN_ALL_GRPO_OUT`, `RUN_ALL_DPO_CFG`, `RUN_ALL_EVAL_CFG`, `RUN_ALL_EVAL_CKPT`, `RUN_ALL_EVAL_OUT`
- Skip steps: `RUN_ALL_SKIP_SFT=1`, `RUN_ALL_SKIP_DPO=1`, `RUN_ALL_SKIP_GRPO=1`, `RUN_ALL_SKIP_EVAL=1`

Ensure `configs/chpc_dpo.yaml` points `sft_checkpoint` at the same directory as `RUN_ALL_SFT_OUT` (default: `chpc/results/sft/3action/final`).

Defaults point at `configs/chpc_grpo.yaml`, `configs/chpc_sft_3action.yaml`, and `configs/chpc_dpo.yaml` with outputs under `chpc/results/`.

### Build SFT JSONL (3-action tokens)

SFT training expects a `messages` JSONL at `sft_data_path` (default: `data/sft_3action_math_train.jsonl`). You can build it explicitly:

```bash
bash chpc/run_chpc.sh build-sft-data \
  data/rollouts_grouped_math_Qwen2.5-Math-7B.jsonl \
  data/sft_3action_math_train.jsonl
```

Or just run `train-sft` / `run-all`: if the JSONL is missing, they will **auto-submit** `build-sft-data` first and chain the dependency.

`run_chpc.sh` still exports **`CHPC_REPO`**, but compute jobs **do not rely on it alone**: `chpc/slurm_prologue.sh` resolves **`REPO_ROOT`** from the batch script path (parent of `chpc/`) whenever `SLURM_SUBMIT_DIR` / `CHPC_REPO` point at a directory without `pyproject.toml`. That matches the usual CHPC pattern (`SLURM_SUBMIT_DIR` + fallback). Optional: **`CHPC_SLURM_MODULES`** in `env_local.sh` overrides default `module load` lines. Grace/aarch64 nodes drop the Granite x86 `teaching-llms-errors` venv from `PATH` before `activate` (same idea as your other project).

## Modules

```bash
module load cuda/12.2
module load python/3.11
```

## Environment (shared venv)

Slurm scripts and batch jobs use this **default** interpreter (override with `CHPC_VENV`):

```bash
source ~/venvs/teaching-llms-errors/bin/activate
```

Equivalent:

```bash
export CHPC_VENV="${CHPC_VENV:-$HOME/venvs/teaching-llms-errors}"
source "${CHPC_VENV}/bin/activate"
```

**Bisa dipakai bareng-bareng?** Ya, **selama** venv itu ada di path yang sama di node compute dan semua anggota tim punya **izin baca + execute** (venv kursus / shared install). Semua job membaca **paket yang sama**; yang tidak dibagi adalah **isi repo + `HF_HOME` + hasil** di home/scratch masing-masing. Jangan `pip install` ke venv shared kecuali tim/dosen setuju (bisa bentrok versi).

Dari root repo, pasang project dalam mode editable **sekali** (jika belum):

```bash
cd ~/learning-when-to-think
source ~/venvs/teaching-llms-errors/bin/activate
pip install --upgrade pip
pip install -e .
```

`bitsandbytes` harus ada di venv itu untuk **QLoRA** (`use_qlora: true` di config CHPC).

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

Then activate the same venv (`source ~/venvs/teaching-llms-errors/bin/activate` or `CHPC_VENV`), set `HF_HOME`, e.g.:

```bash
python scripts/train.py --config configs/chpc_debug.yaml --output-dir chpc/results/checkpoints/debug
```

## OOM / stability

- Reduce `batch_size`, `num_rollouts_per_problem`, or `max_tokens_per_step` in the YAML.  
- Keep `use_qlora: true` for Math-7B on a single GPU.  
- Increase `#SBATCH --mem` in the `.slurm` files if needed.

## Slurm files (advanced)

`chpc/*.slurm` jobs can be submitted manually with the same `sbatch` flags as `run_chpc.sh`; defaults use **notchpeak** names in `#SBATCH` — override with `CHPC_ACCOUNT`, `CHPC_PARTITION`, and `CHPC_QOS` when calling `run_chpc.sh`, or edit the headers once per cluster.
