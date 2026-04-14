# Learning when to think

RL (**GRPO**) + **LoRA** trains an adaptive reasoning policy with three meta-actions: **continue**, **refine** (self-correction nudge), and **terminate**. Training uses **ALP-style** group rewards: \(r_{\text{acc}} - \beta \max(0, \text{SR}) \cdot n_{\text{tokens}} / L_{\max}\), where **SR** is the empirical solve rate across rollouts for the same prompt. **DeGRPO** optionally up-weights log-probability gradients on the short **control** prefix (action token) versus the rest of each step (`degrpo`, `w_ctrl`, `w_resp`); set `degrpo: false` for vanilla GRPO on all generated tokens.

The default benchmark is **MATH-500** (`HuggingFaceH4/MATH-500`): answers are graded after normalizing LaTeX, using `#### <answer>` and/or `\boxed{...}` in model outputs.

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Train (see `configs/default.yaml`; backbone **Qwen2.5-Math-7B-Instruct**):

```bash
python scripts/train.py --config configs/default.yaml --output-dir checkpoints/grpo
```

Evaluate MATH-500 (CoT + direct + adaptive with and without refine):

```bash
python scripts/eval.py --config configs/default.yaml --checkpoint checkpoints/grpo/final
```

HumanEval (separate config):

```bash
python scripts/eval.py --config configs/eval_humaneval.yaml
```

Plots from `comparison.json`:

```bash
python scripts/plot_results.py --comparison results/eval/comparison.json
```

## Two eval entrypoints

| Script | Stack | Use |
|--------|--------|-----|
| `scripts/eval.py` | Hugging Face `generate`, adaptive policy in `src/policy/adaptive.py` | Main GRPO / LoRA checkpoint eval |
| `scripts/eval_pivot.py` | vLLM | Fast MATH-500 throughput; see `configs/eval_qwen_math_7b.yaml` |

## Decoding & eval knobs (`configs/default.yaml`)

| Key | Typical value | Notes |
|-----|----------------|-------|
| `eval_subset_size` | `100` | Set to **`500`** for full MATH-500 |
| `max_steps` | `5` | Adaptive rollout step cap |
| `max_tokens_per_step` | `256` | Per adaptive step |
| `direct_max_tokens` | `64` | Direct baseline cap |
| `allow_refine` | `true` | `eval.py` also runs a no-refine adaptive pass |
| `constrain_action_first_token` | `false` | If `true`, first token of each adaptive step is restricted to action-prefix IDs (forced interface) |
| `beta`, `L_max`, `degrpo`, `n_control_tokens`, `w_ctrl`, `w_resp` | — | GRPO / ALP / DeGRPO |

CoT and direct baselines use temperatures inside `cot_generate` / `direct_generate` in `src/policy/adaptive.py` unless you extend those call sites to read YAML.

## SFT (3-action JSONL) and DPO

- Build data: `scripts/generate_sft_3action.py` (writes `messages` JSONL).
- SFT: `python scripts/train_sft_3action.py --config configs/chpc_sft_3action.yaml --output-dir checkpoints/sft_3action`
- **DPO** (Direct Preference Optimization, `src/train/dpo.py`): `python scripts/train_dpo.py --config configs/chpc_dpo.yaml`

## CHPC (University of Utah)

Use **one** helper script from the repo root; all batch artifacts go under **`chpc/results/`** (logs, checkpoints, eval JSON) for easy `rsync`.

```bash
# Example: submit GRPO training (override account/partition if needed)
export CHPC_ACCOUNT=cs6966 CHPC_PARTITION=granite-gpu-guest CHPC_QOS=granite-gpu-guest
bash chpc/run_chpc.sh train-grpo configs/chpc_grpo.yaml chpc/results/checkpoints/grpo

bash chpc/run_chpc.sh eval configs/chpc_grpo.yaml chpc/results/checkpoints/grpo/final chpc/results/eval/latest
bash chpc/run_chpc.sh train-sft configs/chpc_sft_3action.yaml chpc/results/sft/3action
bash chpc/run_chpc.sh train-dpo configs/chpc_dpo.yaml

# Full pipeline: GRPO + SFT in parallel, then DPO after SFT, eval after GRPO
bash chpc/run_chpc.sh run-all
```

Full notes: [chpc/README.md](chpc/README.md).

## Metrics

- **MATH-500:** accuracy (normalized string match), avg tokens per problem, cost per correct answer, **format_success_rate** (`####` or `\boxed{}`), aggregated **action_counts_total**.

## Layout

- `src/policy/adaptive.py` — shared rollout loop for train and eval  
- `src/train/grpo.py` — GRPO + ALP + DeGRPO + LoRA  
- `src/train/dpo.py` — DPO (preference pairs)  
- `src/train/model_loading.py` — optional 4-bit QLoRA base load  
- `src/data/math_500.py` — MATH-500 load + extraction + grading  
- `src/pivot/` — vLLM eval + ALP helpers (parallel track)  
