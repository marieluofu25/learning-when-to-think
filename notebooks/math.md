# Math: how we score, reward, and aggregate

This note matches the implementation in `src/data/gsm8k.py`, `src/train/grpo.py`, and `src/eval/evaluate.py`. We use **GSM8K**: questions are natural language; the **final answer is a single number** compared under fixed rules.

---

## 1. Gold answer (dataset)

Official GSM8K solutions end with `#### <number>`. We parse that substring and convert to `float` (commas removed).

- **Code:** `extract_answer_number()` in `gsm8k.py`
- **Notation:** gold value \(g\)

---

## 2. Predicted answer (model output)

We do **not** parse `####` from the model. We scan the full output with a number regex and take the **last** match as the predicted answer (common heuristic for “final number” in CoT).

- If no number is found: prediction is `None`.
- **Code:** `extract_predicted_number()` in `gsm8k.py`
- **Notation:** \(\hat{y}\) or “no prediction”

---

## 3. Correctness for evaluation (exact match on reals)

A problem is **correct** iff a prediction exists and is close to gold in absolute terms:

\[
\text{correct} \iff \hat{y} \neq \text{None} \;\wedge\; |\hat{y} - g| < \varepsilon,\quad \varepsilon = 10^{-3}
\]

- **Code:** `grade_answer(predicted, gold, tol=1e-3)` in `gsm8k.py`
- **Instantiated:** \(\varepsilon = 0.001\) (fixed in code, not YAML).
- **Accuracy** in JSON results is \(\dfrac{\#\text{correct}}{\#\text{problems}}\).

---

## 4. GRPO reward (training only)

Let:

- \(t\) = total generated tokens in the rollout  
- \(T_{\max} = \texttt{max\_steps} \times \texttt{max\_tokens\_per\_step}\) (cap used for normalization)  
- \(\lambda\) = `lambda_cost` from `configs/default.yaml` (and `grpo.py` default if not overridden)

### Correctness score \(c\)

1. If \(\hat{y}\) is missing: \(c = 0\).
2. Else if `grade_answer` is true: \(c = 1\).
3. Else (wrong but numeric), relative error  
   \[
   r = \frac{|\hat{y} - g|}{\max(|g|,\,10^{-6})}
   \]
   then:
   - \(r < 0.1 \Rightarrow c = 0.5\)
   - \(r < 0.5 \Rightarrow c = 0.2\)
   - else \(c = 0.05\)

### Cost penalty

\[
\text{penalty} = \lambda \cdot \frac{t}{T_{\max}}
\]

### Reward

\[
R = c - \text{penalty}
\]

- **Code:** `compute_reward()` in `grpo.py`

---

## 5. GRPO advantage and policy loss

For one question, sample \(K\) rollouts with rewards \(R_1,\ldots,R_K\). **Instantiated:** \(K = 4\) (`num_rollouts_per_problem` in `configs/default.yaml`).

**Advantages** (group normalization):

\[
A_i =
\begin{cases}
\dfrac{R_i - \bar{R}}{\mathrm{std}(R)} & \text{if } \mathrm{std}(R) > 10^{-8} \\[6pt]
0 & \text{otherwise}
\end{cases}
\]

**Per-rollout loss** (policy gradient style on the generated span):

\[
\mathcal{L}_i = -A_i \cdot \log \pi_\theta(\text{generated}_i \mid \text{prompt})
\]

Implementation sums log-probabilities over all generated tokens for that rollout (`recompute_log_probs`), then backprops \(\mathcal{L}_i\). **Instantiated:** gradient clip `max_norm = 1.0`; optimizer LR `1 \times 10^{-4}` (`configs/default.yaml`).

- **Code:** `grpo_step()` in `grpo.py`

---

## 6. Self-consistency (eval)

For each problem, sample \(k\) independent CoT answers. Extract \(\hat{y}\) for each. Among non-`None` predictions, take the **majority vote** (mode); ties follow `Counter.most_common` ordering. Grade that majority with the same `grade_answer` rule as above.

- **Total tokens** for the row = sum of tokens across the \(k\) samples.
- **Instantiated (quick eval):** \(k = 3\) (`self_consistency_k` in `configs/eval_quick.yaml`).
- **Code:** `run_self_consistency()` in `evaluate.py`

---

## 7. Aggregate metrics (e.g. `comparison.json`)

For a list of per-problem results:

| Metric | Formula |
|--------|--------|
| `accuracy` | \(\text{correct} / \text{total}\) |
| `avg_tokens_per_problem` | \(\big(\sum t_i\big) / \text{total}\) |
| `cost_per_correct_answer` | \(\big(\sum t_i\big) / \max(\text{correct}, 1)\) |
| `avg_steps` | \(\big(\sum \text{num\_steps}_i\big) / \text{total}\) |

- **Code:** `evaluate_results()` in `evaluate.py`

---

## 8. Instantiated parameters (this repo)

| Symbol / setting | Value | Source |
|------------------|-------|--------|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` | `configs/default.yaml`, `configs/eval_quick.yaml` |
| \(\varepsilon\) (grading) | `0.001` | `gsm8k.grade_answer(..., tol=1e-3)` |
| `max_steps` | `5` | YAML |
| `max_tokens_per_step` | `256` | YAML |
| \(T_{\max}\) | `5 × 256 = 1280` | Used in `grpo_step` as `max_total_tokens` |
| \(\lambda\) (`lambda_cost`) | `0.1` | `configs/default.yaml` |
| GRPO rollouts \(K\) | `4` | `num_rollouts_per_problem` |
| GRPO `learning_rate` | `1e-4` | `configs/default.yaml` |
| LoRA `r` / `alpha` | `16` / `32` | `configs/default.yaml` |
| Advantage std floor | `1e-8` | `grpo.py` |
| Eval subset (quick runs) | `n = 30` test problems | `configs/eval_quick.yaml` |
| Self-consistency \(k\) (quick eval) | `3` | `configs/eval_quick.yaml` |

Example reward at \(\lambda=0.1\), \(T_{\max}=1280\): if \(t=256\) tokens and \(c=1\), then  
\(R = 1 - 0.1 \times (256/1280) = 1 - 0.02 = 0.98\).

### Distill pipeline hyperparameters (run that produced `eval_distill`)

| Stage | Parameters |
|-------|------------|
| Teacher data | `100` traces in `data/teacher_traces.jsonl` (GSM8K train gold solutions; `src/data/teacher.py` `--source gsm8k`) |
| SFT | `3` epochs, LR `2 \times 10^{-5}`, LoRA `r=16`, `max_length=1024`, `per_device_batch_size=1`, `gradient_accumulation_steps=4` (`src/train/sft.py`) |
| DPO pairs | `8` rows in `data/dpo_pairs.jsonl` (from `30` train problems × `3` samples / pair generation run) |
| DPO | `2` epochs, LR `5 \times 10^{-6}`, \(\beta = 0.1\), `max_length=1024` (`src/train/dpo.py`) |

---

## 9. Measured results (real runs, `n = 30`)

Same benchmark slice and grading as above. Token counts are **sum of generated tokens** per method as logged by the eval scripts.

### 9.1 SFT + DPO distill eval (`configs/eval_quick.yaml`)

Output file: `results/eval_distill/comparison.json`.

| Method | Correct / Total | Accuracy | Avg tokens / problem | Cost / correct answer | Avg steps |
|--------|-----------------|----------|----------------------|------------------------|-----------|
| CoT baseline | 8 / 30 | 0.2667 | 266.23 | 998.38 | 1.0 |
| Self-consistency (\(k=3\)) | 10 / 30 | 0.3333 | 810.93 | 2432.80 | 3.0 |
| SFT CoT | 8 / 30 | 0.2667 | 281.70 | 1056.38 | 1.0 |
| Adaptive (SFT+DPO) | 1 / 30 | 0.0333 | 210.93 | 6328.00 | 2.3 |
| SFT+DPO CoT | 7 / 30 | 0.2333 | 259.03 | 1110.14 | 1.0 |

### 9.2 GRPO checkpoint eval (`results/eval_pulse/comparison.json`)

Same \(n=30\) quick setting; adaptive uses trained GRPO adapter (see that run’s `--checkpoint`).

| Method | Correct / Total | Accuracy | Avg tokens / problem | Cost / correct answer | Avg steps |
|--------|-----------------|----------|----------------------|------------------------|-----------|
| CoT | 5 / 30 | 0.1667 | 271.23 | 1627.40 | 1.0 |
| Self-consistency (\(k=3\)) | 11 / 30 | 0.3667 | 805.20 | 2196.00 | 3.0 |
| Adaptive (GRPO) | 1 / 30 | 0.0333 | 253.70 | 7611.00 | 3.83 |

---

## 10. Caveats (for writeups)

1. **Last-number heuristic** can be wrong if the model prints extra numbers after the final answer.
2. **Partial credit** exists only in **GRPO reward**, not in reported **eval accuracy** (eval is strict \( \varepsilon \)-match).
3. **DPO** in this repo uses preference pairs; its training objective is handled inside TRL’s `DPOTrainer`, not the formulas above.
4. **Small \(n\)** (30 problems): accuracies have high variance; use CIs or a larger split for a paper-quality table.
