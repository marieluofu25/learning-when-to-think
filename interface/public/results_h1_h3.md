# H1–H3 Results (MATH-500, minimal scope)

Short results doc tied to the Final proposal (§4.1 Hypotheses, §4.2 Setup).

Setup actually used in this report:

- Model: Qwen2.5-Math-7B-Instruct with LoRA (r=16, α=32) on attention + MLP.
- Training: GRPO + ALP on MATH-500 train subset (n=32) for 2 epochs; main variant `degrpo=true`, baseline variant `degrpo=false`.
- Eval: MATH-500 subset n=100 per checkpoint (job cap 2h on CHPC).
- Action set at training time: `allow_refine: false` → policy effectively `continue / terminate` (2-action). H3 is therefore reported in reduced form.
- Resume: per-problem JSONL so interrupted eval jobs can be resumed without recomputation.

Results below are from one eval run (`eval_subset_size=100`), merged for H1 via `scripts/analyze_h1_pareto.py` (main + vanilla `comparison.json`). Baselines CoT/Direct are deduped from the main eval file in that script.

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

Metric table (`python scripts/analyze_h1_pareto.py`; n=100 per adaptive row; CoT/Direct from main eval):

| Method | Correct / total | Accuracy (95% CI) | Avg tokens | Cost / correct | Pareto status |
|--------|-----------------|---------------------|------------|------------------|----------------|
| CoT baseline | 53 / 100 | 0.530 [0.433, 0.625] | 404.4 | 763.0 | non-dominated |
| Direct baseline | 0 / 100 | 0.000 [0.000, 0.037] | 64.0 | 6400.0 | non-dominated |
| Adaptive DeGRPO (+refine) | 46 / 100 | 0.460 [0.366, 0.557] | 754.4 | 1640.1 | dominated |
| Adaptive DeGRPO (-refine) | 59 / 100 | 0.590 [0.492, 0.681] | 613.1 | 1039.2 | non-dominated |
| Adaptive Vanilla (+refine) | 41 / 100 | 0.410 [0.319, 0.508] | 767.3 | 1871.4 | dominated |
| Adaptive Vanilla (-refine) | 50 / 100 | 0.500 [0.404, 0.596] | 594.1 | 1188.3 | dominated |

**Decision:** **Partially supported, nuanced.** The best adaptive point on this slice is **Adaptive DeGRPO (-refine)**: highest accuracy among adaptive runs, non-dominated in the script’s strict 2-D (accuracy vs avg tokens) check, and **higher accuracy than vanilla (-refine)** (0.59 vs 0.50) at similar token cost (~613 vs ~594). Decode with **+refine** is **dominated** for both DeGRPO and vanilla (lower accuracy, more tokens than better points). CIs overlap between CoT (0.53) and DeGRPO (-refine) (0.59), so the accuracy gain is **not** statistically tight at n=100.

Dominated set printed by the analyzer: Adaptive (+refine) [DeGRPO], Adaptive (+refine) [vanilla], Adaptive (-refine) [vanilla]. See `chpc/results/eval/analysis/h1_table.csv` and `h1_pareto.png`.

Notes:

- CIs at n=100 span roughly ±10 points near accuracy 0.5. Differences smaller than that are not statistically distinguishable.
- Vanilla GRPO checkpoint was trained with the same LoRA rank, optimizer, KL, and ALP reward as DeGRPO; only `degrpo` differs. Config diff: [configs/chpc_grpo.yaml](../configs/chpc_grpo.yaml) vs [configs/chpc_grpo_vanilla.yaml](../configs/chpc_grpo_vanilla.yaml).

## H2 — learned policy allocates more compute to harder prompts

### Adaptive (+refine) decode (`adaptive_with_refine_results.json`)

From `summary_adaptive_with_refine.json` / terminal:

- Spearman(level, tokens): **0.360**
- Spearman(level, steps): **0.138**

Per-level averages (`h2_adaptive_with_refine_per_level.csv`):

| Level | n | mean tokens | mean steps | accuracy |
|-------|---|-------------|------------|----------|
| 1 | 11 | 594.4 | 2.64 | 0.545 |
| 2 | 25 | 601.7 | 2.76 | 0.720 |
| 3 | 19 | 672.8 | 2.74 | 0.474 |
| 4 | 22 | 794.2 | 2.77 | 0.273 |
| 5 | 23 | 1026.5 | 2.87 | 0.304 |

**Decision:** **Weakly supported for tokens vs difficulty.** Mean tokens rise from level 1 through 5 (594 → 1027), and Spearman(level, tokens) is positive but moderate. **Not supported for steps:** mean steps are flat (~2.7–2.9) and Spearman(level, steps) ≈ 0.14. Plot: `chpc/results/eval/analysis/h2_adaptive_with_refine_tokens_vs_level.png`.

### Adaptive (-refine) decode (`adaptive_no_refine_results.json`)

- Spearman(level, tokens): **0.288**
- Spearman(level, steps): **0.149**

| Level | n | mean tokens | mean steps | accuracy |
|-------|---|-------------|------------|----------|
| 1 | 11 | 513.3 | 2.82 | 0.909 |
| 2 | 25 | 560.0 | 2.88 | 0.760 |
| 3 | 19 | 570.2 | 2.74 | 0.789 |
| 4 | 22 | 566.2 | 2.95 | 0.500 |
| 5 | 23 | 798.9 | 2.96 | 0.174 |

**Decision:** Same pattern: **token budget grows toward level 5** (513 → 799) with modest Spearman; steps nearly flat.

## H3 — action usage shifts with difficulty

Training used `allow_refine: false`, but **eval can still request +refine**, so the (+refine) decode path can emit refine tokens even if the policy was not trained for them.

### Adaptive (+refine) — action fractions by level (`h3_adaptive_with_refine_action_dist_per_level.csv`)

| Level | n | frac continue | frac refine | frac terminate |
|-------|---|---------------|-------------|----------------|
| 1 | 11 | 0.862 | 0.000 | 0.138 |
| 2 | 25 | 0.841 | 0.029 | 0.130 |
| 3 | 19 | 0.769 | 0.096 | 0.135 |
| 4 | 22 | 0.836 | 0.066 | 0.098 |
| 5 | 23 | 0.879 | 0.030 | 0.091 |

**Decision:** **Partially supported.** `frac(terminate)` is **higher on easier levels (1–2)** than on harder levels (4–5), which matches the qualitative hypothesis (more terminate when the instance is easier). Continue share is noisy and not monotonic in level. Refine usage is small overall but peaks at level 3 in this slice.

### Adaptive (-refine) — refine forced off at decode

| Level | n | frac continue | frac refine | frac terminate |
|-------|---|---------------|-------------|----------------|
| 1 | 11 | 0.839 | 0.000 | 0.161 |
| 2 | 25 | 0.903 | 0.000 | 0.097 |
| 3 | 19 | 0.885 | 0.000 | 0.115 |
| 4 | 22 | 0.985 | 0.000 | 0.015 |
| 5 | 23 | 0.912 | 0.000 | 0.088 |

**Decision:** **Terminate fraction still higher on easy (1) than on hard (5)** in this run; level 4 is an outlier (very few terminates). Interpret as **weak / slice-dependent** evidence for H3 in the 2-action regime. Full 3-action H3 after training with `allow_refine: true` was out of scope.

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
