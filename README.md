# Learning when to think

RL (**GRPO**) + **LoRA** trains an adaptive reasoning policy with three meta-actions: **continue**, **refine** (self-correction nudge), and **terminate**. Training uses **ALP-style** group rewards: \(r_{\text{acc}} - \beta \max(0, \text{SR}) \cdot n_{\text{tokens}} / L_{\max}\), where **SR** is the empirical solve rate across rollouts for the same prompt. **DeGRPO** optionally up-weights log-probability gradients on the short **control** prefix (action token) versus the rest of each step (`degrpo`, `w_ctrl`, `w_resp`); set `degrpo: false` for vanilla GRPO on all generated tokens.

The default benchmark is **MATH-500** (`HuggingFaceH4/MATH-500`): answers are graded after normalizing LaTeX, using `#### <answer>` and/or `\boxed{...}` in model outputs. **GSM8K** is not the primary track in this repo (optional sanity checks can use the same grading helpers if you add a loader).

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Train (see `configs/default.yaml`; default backbone is **Qwen2.5-Math-7B-Instruct**):

```bash
python scripts/train.py --config configs/default.yaml --output-dir checkpoints/grpo
```

Evaluate MATH-500 (CoT + Direct baselines + adaptive with and without refine):

```bash
python scripts/eval.py --config configs/default.yaml --checkpoint checkpoints/grpo/final
```

HumanEval eval (separate config; adaptive uses the same three actions with a code-specific system prompt in `src/eval/evaluate.py`):

```bash
python scripts/eval.py --config configs/eval_humaneval.yaml
```

Plots from `comparison.json`:

```bash
python scripts/plot_results.py --comparison results/eval/comparison.json
```

## Metrics

- **MATH-500:** accuracy (normalized string match), avg tokens per problem, cost per correct answer, **format_success_rate** (`####` or `\boxed{}`), aggregated **action_counts_total**.

## CHPC (University of Utah)

See [chpc/README.md](chpc/README.md) for SLURM, `HF_HOME` on scratch, and **Qwen2.5-Math-7B-Instruct + QLoRA** configs.

## Layout

- `src/policy/adaptive.py` — shared rollout loop for train and eval  
- `src/train/grpo.py` — GRPO + ALP + DeGRPO + LoRA  
- `src/train/model_loading.py` — optional 4-bit QLoRA base load  
- `src/data/math_500.py` — MATH-500 load + extraction + grading  
- `src/data/humaneval.py` — HumanEval load + grading  

# learning-when-to-think

- enable_thinking: true
- max token: 2048
- temperature: 0.0 (greedy)
- repetition penalty: 1.3
- model: Qwen3.5-4B

Training: 3 epochs

LoRA parameters:

- epochs: 3
- learning rate: 2e-5
- batch size: 1
- grad accum: 4
- max seq: 2048
- lora-r: 16
- lora-alpha: 32
- dtype: bfloat16
- target modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
- lora dropout: 0.05
- bias: None
- task type: "CASUAL_LM"
- lr scheduler: "cosine"
- warmup ratio: 0.03

## Forced Masking (Deprecated)

Trained using LoRA SFT for warm up.

`python -m scripts.eval_phase1 --base-model Qwen/Qwen3.5-4B --checkpoint checkpoints/phase1_4B/final_fixed --n 500`

SUMMARY (500 problems)

Accuracy:     99/500 (20%)
- `<continue>`:   1180 total (2.4/problem)
- `<backoff>`:    207 total (0.4/problem)
- `Terminated`:   498/500

## Base Model 

`python -m scripts.eval_phase1 --base-model Qwen/Qwen3.5-4B --no-adapter --n 500`
