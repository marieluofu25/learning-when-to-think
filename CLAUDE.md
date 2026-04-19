# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project: Learning When to Think

LLMs waste compute by thinking the same amount on every problem. This project trains a model that learns *when to keep thinking, self-correct, or stop* via RL.

## Core Approach

We model reasoning as an MDP with three actions:
- **`continue`**: Extend the current reasoning trajectory.
- **`refine`**: Re-check and fix recent reasoning before continuing (lightweight self-correction).
- **`terminate`**: Stop reasoning and output the final answer.

**Base model**: Qwen2.5-Math-7B-Instruct with LoRA (r=16, α=32) on attention + MLP projections.

**Reward**: Adaptive Length Penalty (ALP) — correctness reward minus a length penalty scaled by group solve rate SR(q). Easy problems (high SR) get penalized more for long solutions; hard problems (low SR) get slack.

```
R_k = r_acc,k − β · max(0, SR(q)) · n_tokens,k / L_max
```

**Training**: GRPO (Group Relative Policy Optimization) with DeGRPO weighting — control tokens (action decisions) get higher gradient weight than response tokens to prevent control-signal washout.

```
L = −A_k (w_ctrl · log p_ctrl + w_resp · log p_resp) + λ_KL · KL
```
with w_ctrl > w_resp (default: 2.0 vs 1.0).

**Evaluation domain**: MATH-500 with `\boxed{}` and `####` answer extraction.

**Baselines**: CoT single-pass, direct-answer (no reasoning), vanilla GRPO (degrpo=false).

**Key metrics**: accuracy, avg tokens/problem, cost per correct answer, action usage by difficulty.

**Ablation switches**: `allow_refine` (disables refine action), `degrpo` (toggles DeGRPO weighting).

## Hypotheses

- **H1**: ALP + DeGRPO improves accuracy–efficiency Pareto frontier over baselines.
- **H2**: Learned policy allocates more tokens/steps to harder prompts.
- **H3**: Action usage shifts with difficulty (more terminate on easy, more continue/refine on hard).

## Work Split

- **Member 1 (Data/Reward)**: MATH-500 loading, answer extraction & grading, evaluation harness, ALP reward function.
- **Member 2 (RL Training)**: LoRA setup, 3-action policy with control tokens, GRPO update, DeGRPO gradient weighting.
- **Member 3 (Generation/Experiments)**: Rollout sampling loop, baseline implementations, experiment execution, results & plots.

## Timeline

- Apr 7–11: Parallel development (data pipeline, RL core, generation infra)
- Apr 12–14: Integration
- Apr 15–18: Experiments and ablations
- Apr 19–20: Presentation prep (present Apr 20)

---

# Dataset Generation Rules

SFT data is built from Qwen3-1.7B's rollouts on MATH train. The core objective is injecting `<backoff_N>` tokens into CoT trajectories so the model learns to recognize errors mid-reasoning, emit backoff token(s), and self-correct toward the right answer. Every trajectory ends with the correct answer.

## How Rollouts Are Used

**Wrong rollouts → backoff examples:**
1. Subagent reads a wrong trajectory and locates the error point (at a semantic boundary).
2. Everything before the error point is kept verbatim (real error pattern from the model).
3. Insert `<backoff_N>` + corrective directive at the error point.
4. Rewrite only the portion after the error point into correct reasoning leading to the correct answer.

**Correct rollouts → clean examples (no backoff).**

## Process
- NEVER use script-based bulk transforms on the dataset. Scripts introduce false positives/negatives that pollute training signal.
- Use subagents **(one entry per subagent)** for all per-entry modifications. Do not batch multiple entries into one subagent.

## Semantic Boundaries

A **semantic boundary** is a natural transition point in the CoT where the model shifts to a distinct reasoning step. Backoff tokens must be placed at these boundaries, never mid-sentence or mid-calculation.

Boundary detection heuristics (three categories from `02_method.md`):
1. **Punctuation**: sentence-ending `.` `。`, paragraph breaks `\n\n`, list item transitions
2. **Logical connectives / discourse markers**: "Therefore", "So", "But", "However", "Next", "Now", "Let me", "This means", "Alternatively", "Wait", "Actually" — these signal a reasoning-step transition
3. **Structural markers**: transition between problem setup → equation formulation → solving → verification → answer extraction

Valid placement points (place `<backoff_N>` here):
- After a sentence-ending period where the next sentence starts a new reasoning step
- Before a logical connective that introduces a new direction ("So now I need to...", "Next, let's...")
- Between "set up the equation" and "solve the equation"
- Between "compute intermediate result" and "apply it to the next step"
- Between "consider case A" and "consider case B"
- After a paragraph break (`\n\n`) where the topic shifts
- After a completed sub-calculation before its result is used downstream

Invalid placement points (do NOT place `<backoff_N>` here):
- Mid-sentence or mid-formula (e.g., inside `x^2 + <backoff_1> 3x = 0`)
- Between two lines of the same algebraic manipulation (e.g., between consecutive simplification steps)
- Inside a single logical step that hasn't concluded yet
- Between a connective and its clause (e.g., `Therefore <backoff_1> x = 5`)

## Chunks

A **chunk** is one atomic unit of reasoning between two consecutive semantic boundaries. It corresponds to a single reasoning step (e.g., one equation setup, one case analysis, one sub-computation). The `N` in `<backoff_N>` equals the number of wrong chunks to rewind — i.e., rewind depth = N wrong chunks back from the backoff point.

## Backoff Token Format
```
[chunk 1 (wrong)] [chunk 2 (wrong)] ... [chunk N (wrong)] <backoff_N> [directive] [correct reasoning continues]
```
- `<backoff_1>`: the immediately preceding 1 chunk was wrong
- `<backoff_2>`: the preceding 2 chunks were wrong
- `<backoff_3>`: the preceding 3 chunks were wrong

### One-shot example (`<backoff_1>`, gold=18)

Question: *How many integers between 100 and 150 have three different digits in increasing order?*

```
<think>
...                                          ← (earlier correct reasoning, kept verbatim)

All steps seem solid. Therefore, confidence   ← chunk: wrong conclusion
high that the answer is 28.                     (forgot the ≤150 upper-bound constraint)

<backoff_1> Wait, I got 28, but that is       ← directive: names wrong answer,
wrong. I computed C(8,2)=28 by choosing any     identifies the specific error,
two digits from {2,3,...,9} for the tens and    does NOT reveal gold answer
units places, but I forgot the constraint
that the number must be at most 150. Let me
recount by enumerating valid tens digits
under this upper bound.

Since the number must be between 100 and      ← correct reasoning: redoes the
150, the hundreds digit is 1, and we need       computation properly
100 + 10T + U <= 150, so 10T + U <= 50.
With T > 1 and U > T:
- T = 2: U can be 3,...,9 → 7 numbers
- T = 3: U can be 4,...,9 → 6 numbers
- T = 4: U can be 5,...,9 → 5 numbers
- T = 5: 10(5)=50, U > 5 → invalid
Total = 7 + 6 + 5 = 18.
</think>

\boxed{18}
```

## Backoff Token Distribution
Target distribution across perturbed entries: `<backoff_1>` 60%, `<backoff_2>` 30%, `<backoff_3>` 10%.
After finishing the perturbed entries, we may add clean trajectories with no `<backoff_N>` tokens to build the whole dataset.
When an entry has multiple backoff tokens, each must govern a **separate wrong reasoning chunk**. Between any two backoff tokens there must be at least one complete correct reasoning step (not just a directive). Example of valid `<backoff_2>` layout:
```
[wrong chunk A, ≥1 semantic boundary long] <backoff_1> [directive A] [correct chunk, ≥1 complete step] ... [wrong chunk B] <backoff_2> [directive B] [correct reasoning to final answer]
```

## Directive Rules
- Directives must describe the error without leaking the gold answer.
- Reference the wrong answer explicitly ("Wait, I got X but..."), then name the specific mistake and suggest a corrective direction.

## Requirements
- **Length-preserving**: Keep the full original CoT chain before the error point. The rewritten portion after backoff should be comparable in detail/verbosity.
- **Semantic-preserving**: The final reasoning chain must be coherent and arrive at the correct answer.
- **Correct final answer**: Every entry's `\boxed{}` must match the gold answer.
- **Correct think boundary**: A CoT chain should be correctly boxed by `<think>` and `</think>` tokens.

## Paper 

For a paper summary at `papers/read/`, its related paper is just at `papers/`, categorized into a subdirectory by its topic. You may want to also read the paper before make any conclusion

---

# Repository Layout

Two parallel implementation tracks share the same `src/` tree:

- **Main HF / training track** — `src/policy/adaptive.py` (3-action rollout loop), `src/train/grpo.py` (GRPO + ALP + DeGRPO + LoRA), `src/train/dpo.py`, `src/train/sft.py`, `src/train/model_loading.py` (optional 4-bit QLoRA), `src/eval/evaluate.py`. Used by `scripts/train.py`, `scripts/eval.py`, `scripts/train_sft_3action.py`, `scripts/train_dpo.py`.
- **Pivot / vLLM track** — `src/pivot/{generate,reward,eval_harness,eval_types,tokens,vllm_cuda_env}.py`. Faster MATH-500 throughput; used by `scripts/eval_pivot.py` and `scripts/generate_rollouts_pivot.py`.

Data adapters live in `src/data/` (`math_500.py` is canonical; `math.py`, `gsm8k.py`, `humaneval.py`, `math_competition.py`, `teacher.py` are auxiliary).

Configs in `configs/`: `default.yaml` is the local recipe; `chpc_*.yaml` are cluster recipes; `eval_qwen_math_7b.yaml` drives the vLLM pivot eval. Action knobs live in YAML — most flags (`degrpo`, `allow_refine`, `constrain_action_first_token`, `w_ctrl`/`w_resp`, `beta`, `L_max`, `n_control_tokens`) are read from the config, not CLI.

# Common commands

```bash
pip install -e .                                              # editable install (defines `learning-when-to-think` package)

# Train + evaluate (HF stack)
python scripts/train.py --config configs/default.yaml --output-dir checkpoints/grpo
python scripts/eval.py  --config configs/default.yaml --checkpoint checkpoints/grpo/final

# Fast MATH-500 eval (vLLM pivot)
python -m scripts.eval_pivot --config configs/eval_qwen_math_7b.yaml

# SFT / DPO
python scripts/generate_sft_3action.py            # build messages JSONL
python scripts/train_sft_3action.py --config configs/chpc_sft_3action.yaml --output-dir checkpoints/sft_3action
python scripts/train_dpo.py        --config configs/chpc_dpo.yaml

# Plots and analysis
python scripts/plot_results.py     --comparison results/eval/comparison.json
python scripts/analyze_h1_pareto.py            # H1 (accuracy/efficiency Pareto)
python scripts/analyze_h2_h3.py                # H2/H3 (difficulty allocation)

# Tests (pytest is not in pyproject deps; install separately if missing)
pytest tests/                                  # all
pytest tests/test_pivot_eval.py::<test_name>   # single test
```

# CHPC (Utah cluster)

Single entrypoint: `bash chpc/run_chpc.sh <subcmd> ...` from repo root. Subcommands: `train-grpo | eval | build-rollouts-data | build-sft-data | train-sft | train-dpo | run-all`. All artifacts go under `chpc/results/`. Override Slurm with `CHPC_ACCOUNT` / `CHPC_PARTITION` / `CHPC_QOS`; venv with `CHPC_VENV` (default `~/venvs/teaching-llms-errors`). `run-all` chains GRPO+SFT in parallel, then DPO after SFT and eval after GRPO, auto-submitting `build-rollouts-data` → `build-sft-data` if the SFT JSONL is missing. See `chpc/README.md` for the full matrix.

# Notes for changes that touch the rollout loop

- The 3-action interface is defined by string tokens (`<continue>`, `<refine>`, `<terminate>`) in `src/policy/adaptive.py`. Legacy checkpoints used `<verify>` for `<refine>` — the detector handles both. Don't rename without updating both `adaptive.py` and the SFT/DPO data builders.
- DeGRPO weights `n_control_tokens` action-prefix tokens with `w_ctrl` and the rest with `w_resp` (`src/train/grpo.py:recompute_log_probs_weighted`). Setting `degrpo: false` falls back to vanilla GRPO with uniform weighting.
- ALP reward uses **group solve rate `SR(q)`** computed over `num_rollouts_per_problem` rollouts per prompt — changes to grouping must keep that grouping intact, or the length penalty becomes meaningless.

# Project-specific docs

- `project_docs/02_method.md` — semantic-boundary heuristics referenced by the dataset rules above.
- `papers/` — referenced literature; summaries under `papers/read/`.
- `.claude/agents/backoff-sample-refiner.md` — subagent prompt used for one-entry-at-a-time backoff dataset perturbations (see "Process" rule above).