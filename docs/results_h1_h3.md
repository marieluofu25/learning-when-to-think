# H1–H3 Results (MATH-500, minimal scope)

Short results doc tied to the Final proposal (§4.1 Hypotheses, §4.2 Setup).

Setup actually used in this report:

- Model: Qwen2.5-Math-7B-Instruct with LoRA (r=16, α=32) on attention + MLP.
- Training: GRPO + ALP on MATH-500 train subset (n=32) for 2 epochs; main variant `degrpo=true`, baseline variant `degrpo=false`.
- Eval: MATH-500 subset n=100 per checkpoint (job cap 2h on CHPC).
- Action set at training time: `allow_refine: false` → policy effectively `continue / terminate` (2-action). H3 is therefore reported in reduced form.
- Resume: per-problem JSONL so interrupted eval jobs can be resumed without recomputation.

> Numbers below are placeholders — fill them after the CHPC jobs complete and analysis scripts have been run.

## Pipeline to reproduce

All commands below run from repo root. CHPC jobs live under `chpc/results/`.

1. Submit vanilla GRPO training (baseline for H1):

```bash
bash chpc/run_chpc.sh train-grpo configs/chpc_grpo_vanilla.yaml chpc/results/checkpoints/grpo_vanilla
```

2. Eval both checkpoints (2h each):

```bash
bash chpc/run_chpc.sh eval \
  configs/chpc_grpo.yaml \
  chpc/results/checkpoints/grpo/final \
  chpc/results/eval/main

bash chpc/run_chpc.sh eval \
  configs/chpc_grpo_vanilla.yaml \
  chpc/results/checkpoints/grpo_vanilla/final \
  chpc/results/eval/vanilla
```

Re-submit the same command if a job is killed at the 2h limit — `*_results.partial.jsonl` lets it resume per-problem.

3. Analyses:

```bash
python scripts/analyze_h1_pareto.py \
  --main chpc/results/eval/main/comparison.json \
  --vanilla chpc/results/eval/vanilla/comparison.json \
  --out chpc/results/eval/analysis

python scripts/analyze_h2_h3.py \
  --input chpc/results/eval/main/adaptive_with_refine_results.json \
  --out chpc/results/eval/analysis \
  --tag adaptive_with_refine

python scripts/analyze_h2_h3.py \
  --input chpc/results/eval/main/adaptive_no_refine_results.json \
  --out chpc/results/eval/analysis \
  --tag adaptive_no_refine
```

## H1 — ALP + DeGRPO improves the accuracy–efficiency Pareto frontier

Metric table (fill after running `scripts/analyze_h1_pareto.py`):

| Method | Accuracy (95% CI) | Avg tokens | Cost / correct | Pareto status |
|--------|-------------------|------------|-----------------|----------------|
| CoT baseline | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Direct baseline | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Adaptive DeGRPO (+refine) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Adaptive DeGRPO (-refine) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Adaptive Vanilla (+refine) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Adaptive Vanilla (-refine) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

**Decision (fill):** supported / not supported. Point to dominated set in `h1_table.csv`.

Notes:

- CIs at n=100 span roughly ±10 points near accuracy 0.5. Differences smaller than that are not statistically distinguishable.
- Vanilla GRPO checkpoint was trained with the same LoRA rank, optimizer, KL, and ALP reward as DeGRPO; only `degrpo` differs. Config diff: [configs/chpc_grpo.yaml](../configs/chpc_grpo.yaml) vs [configs/chpc_grpo_vanilla.yaml](../configs/chpc_grpo_vanilla.yaml).

## H2 — learned policy allocates more compute to harder prompts

Read `summary_adaptive_with_refine.json`:

- Spearman(level, tokens): _tbd_
- Spearman(level, steps): _tbd_

Per-level averages (from `h2_adaptive_with_refine_per_level.csv`):

| Level | n | mean tokens | mean steps | accuracy |
|-------|---|-------------|------------|----------|
| 1 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 2 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 3 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 4 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 5 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

**Decision (fill):** supported if Spearman(level, tokens) is meaningfully positive and tokens trend up with level. Plot: `h2_adaptive_with_refine_tokens_vs_level.png`.

## H3 — action usage shifts with difficulty

Because the main checkpoint was trained with `allow_refine: false`, the observed action set is `{continue, terminate}`. H3 is reported in reduced form here.

Per-level action fractions (from `h3_adaptive_with_refine_action_dist_per_level.csv`):

| Level | n | frac continue | frac refine | frac terminate |
|-------|---|----------------|--------------|------------------|
| 1 | _tbd_ | _tbd_ | — | _tbd_ |
| 2 | _tbd_ | _tbd_ | — | _tbd_ |
| 3 | _tbd_ | _tbd_ | — | _tbd_ |
| 4 | _tbd_ | _tbd_ | — | _tbd_ |
| 5 | _tbd_ | _tbd_ | — | _tbd_ |

**Decision (fill):** supported in reduced form if `frac terminate` drops as level rises (equivalently, continue share rises with difficulty). Full 3-action H3 requires retraining with `allow_refine: true`; not done in this scope.

## Limitations

- **Training subset small (n=32):** GRPO may be under-trained; `training_history.json` shows train accuracy 0.59 → 0.49 across epochs, suggesting instability. No retrain performed in this scope.
- **Eval subset n=100:** CI 95% at accuracy 0.5 is about ±10 points. Comparisons within this gap are inconclusive.
- **Reduced action space at train time:** `allow_refine: false` means the main checkpoint never exercised the refine action during training. H3 covers continue/terminate only.
- **Vanilla GRPO small-scope:** matched to main config for a clean ablation, but inherits the same small-data caveats.
- **No external PRM, no AIME / full MATH, no self-consistency Pass@k baseline.** These are out of scope for the Final proposal §4.2.

## Files produced

- `chpc/results/eval/main/{cot,direct,adaptive_with_refine,adaptive_no_refine}_results.json`
- `chpc/results/eval/main/comparison.json`
- `chpc/results/eval/vanilla/...` (same layout)
- `chpc/results/eval/analysis/h1_table.csv` and `h1_pareto.png`
- `chpc/results/eval/analysis/h2_<tag>_per_level.csv` and `h2_<tag>_tokens_vs_level.png`
- `chpc/results/eval/analysis/h3_<tag>_action_dist_per_level.csv` and `h3_<tag>_action_dist_per_level.png`
- `chpc/results/eval/analysis/summary_<tag>.json` (Spearman + per-level stats)
