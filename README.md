# Learning when to think

RL (GRPO) + LoRA trains an adaptive reasoning policy over five meta-actions: **continue**, **verify**, **sample_alt**, **call_tool** (sandboxed Python), **terminate**. Rewards follow the proposal-style form \(r_{\text{correct}} + \text{format bonus} - \lambda\cdot n_{\text{tokens}} - \mu\cdot n_{\text{tool}}\).

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Train (see `configs/default.yaml`; default backbone is **Qwen2.5-Math-7B-Instruct**, `use_qlora: false` for machines without bitsandbytes / small GPU):

```bash
python scripts/train.py --config configs/default.yaml --output-dir checkpoints/grpo
```

Evaluate GSM8K (baselines + adaptive + direct answer):

```bash
python scripts/eval.py --config configs/default.yaml --checkpoint checkpoints/grpo/final
```

HumanEval-only eval (expects a trained LoRA at `checkpoint_path` in the YAML):

```bash
python scripts/eval.py --config configs/eval_humaneval.yaml
```

Plots from `comparison.json`:

```bash
python scripts/plot_results.py --comparison results/eval/comparison.json
```

## Metrics

- **GSM8K:** accuracy (numeric match), avg tokens per problem, cost per correct answer, aggregated **action_counts_total**, **total_tool_calls**.
- **HumanEval:** pass@1 via official tests in an isolated subprocess.

## CHPC (University of Utah)

See [chpc/README.md](chpc/README.md) for SLURM, `HF_HOME` on scratch, and **Qwen2.5-Math-7B-Instruct + QLoRA** configs (`configs/chpc_grpo.yaml`, `configs/chpc_debug.yaml`).

## Layout

- `src/policy/adaptive.py` — shared rollout loop for train and eval  
- `src/policy/tool_exec.py` — fenced-code extraction + sandboxed execution  
- `src/train/grpo.py` — GRPO + LoRA  
- `src/train/model_loading.py` — optional 4-bit QLoRA base load  
- `src/data/humaneval.py` — HumanEval load + grading  
