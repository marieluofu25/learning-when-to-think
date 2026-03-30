# Project Summary: Learning When to Think

## Team
- **Ivan Andhika** (u1590903) - Distillation + DPO/GRPO training approaches
- **Hao Ren** (u1527543) - Classical RL formulation and training
- **Ammon Gleason** (u1221580) - Difficulty-aware evaluation and analysis

## The Problem

Current LLMs treat every problem the same: they always generate the same amount
of reasoning regardless of difficulty. "What is 2+2?" gets the same chain-of-thought
treatment as a complex multi-step calculus problem. This wastes compute on easy
problems and may under-serve hard ones.

## Our Solution

Use **Reinforcement Learning** to teach a small LLM (Qwen 2.5 0.5B) to dynamically
decide *how much* to think. At each reasoning step, the model chooses one of these
actions:

| Action | What It Does |
|---|---|
| **Continue** | Keep extending the chain-of-thought reasoning |
| **Verify** | Pause and check if the current intermediate result is correct |
| **Alternative Sample** | Scrap current approach, try a different solution path |
| **Terminate** | Stop thinking and output the final answer |

### The Reward Function

```
R = correctness - lambda * (tokens_used / max_tokens)
```

- **correctness**: 1.0 if answer is right, 0.0 if wrong
- **format bonus**: +0.1 if output follows `#### <number>` format
- **lambda**: controls the penalty for using more tokens (default 0.1)
- Cost penalty only applies when the answer is correct (don't punish exploration on wrong answers)

The model learns to be both **accurate** and **efficient**.

## Architecture (from the Flowchart)

```
Start --> Problem Batch (GSM8K) --> Base LLM + LoRA --> [Action Selection]
                                                            |
                                              +-------------+-------------+
                                              |             |             |
                                         Continue      Verify      Alternative
                                         Reasoning    Immediate      Sample
                                              |             |             |
                                              +-------------+-------------+
                                                            |
                                                      Terminate?
                                                       /       \
                                                     No         Yes
                                                     |           |
                                                (loop back)   Final Answer
                                                                 |
                                                          Reward Computation
                                                      (R = correct - lambda*cost)
                                                                 |
                                                          Policy Update
                                                        (PPO/GRPO + LoRA)
                                                                 |
                                                       Iterate until converged
```

## Key Concepts

### LoRA (Low-Rank Adaptation)
Instead of fine-tuning ALL the weights of a large model (billions of parameters),
LoRA freezes the original weights and adds small trainable "adapter" matrices.
This makes training feasible on limited hardware. Our project uses rank=16 LoRA
on Qwen 2.5 0.5B's attention layers (q_proj, v_proj, k_proj, o_proj).

### SFT (Supervised Fine-Tuning)
Standard fine-tuning where you show the model good examples and train it to
replicate them. Ivan uses teacher traces (from Gemini 2.5 Flash or GSM8K gold
solutions) to teach the student model how to reason step-by-step.

### DPO (Direct Preference Optimization)
After SFT, you generate multiple answers per problem. The correct answers become
"chosen" examples and wrong answers become "rejected" examples. DPO trains the
model to prefer correct solutions without needing an explicit reward model.

### GRPO (Group Relative Policy Optimization)
A more direct RL approach. For each problem, generate K rollouts (K=4), score
them with the reward function, normalize the rewards within the group (hence
"relative"), and update the policy to increase probability of high-reward
rollouts. This is what the code uses for actual RL training.

### GSM8K
A benchmark of ~8,000 grade-school math word problems. Each problem has a
human-written step-by-step solution ending with `#### <number>`. Problems vary
from 1-step arithmetic to 8+ step reasoning. This is the primary evaluation
dataset.

### Self-Consistency
A baseline method: generate K independent chain-of-thought answers, then take
the majority vote. More compute = better accuracy, but it's a fixed strategy
(always uses K samples regardless of difficulty).

## What Each Person Is Doing

### Ivan's Approach: Distillation + DPO/GRPO
**Code branch:** `ivan`

**Pipeline:**
1. **Teacher Generation** (`src/data/teacher.py`): Use Gemini 2.5 Flash (or
   GSM8K gold solutions) to generate high-quality reasoning traces
2. **SFT** (`src/train/sft.py`): Fine-tune Qwen 0.5B on these traces with LoRA
3. **DPO** (`src/train/dpo.py`): Generate preference pairs from the SFT model,
   train to prefer correct answers
4. **GRPO** (`src/train/grpo.py`): Direct RL training with rollouts and
   group-relative advantages
5. **Evaluation** (`scripts/eval.py`): Compare adaptive policy vs baselines

**Current Results (200 problems):**
| Method | Accuracy | Avg Tokens | Cost/Correct |
|---|---|---|---|
| CoT Baseline | 34.0% | 293 | 862 |
| Self-Consistency (k=5) | 43.0% | 1437 | 3342 |
| SFT+DPO CoT | 36.0% | 284 | 788 |
| Adaptive SFT+DPO | 2.5% | 252 | 10080 |

The adaptive policy is struggling (2.5% accuracy) -- likely due to the small
model having difficulty learning the action selection on top of reasoning.
The SFT+DPO CoT mode slightly outperforms the vanilla baseline (36% vs 34%)
with fewer tokens (284 vs 293), which is a promising signal.

### Hao's Approach: Classical RL
Hao is formulating the problem mathematically and exploring a more traditional
RL training approach. His focus is on the formal MDP formulation and potentially
different RL algorithms.

### Ammon's Contribution: Difficulty-Aware Evaluation & Analysis
**Code branch:** `ammon`

**Goal:** Build the analysis that proves (or disproves) the paper's core thesis:
"the model learns to think more on hard problems and less on easy ones."

**Script:** `scripts/difficulty_analysis.py`

**What it does:**
1. **Difficulty classifier**: Parse GSM8K ground-truth solutions to count
   reasoning steps --> bin problems as easy/medium/hard
2. **Per-difficulty analysis**: Run each method on each difficulty bin and measure
   tokens used, accuracy, and action distribution
3. **Visualization**: Create the key plots for the paper:
   - Token usage vs. difficulty (adaptive should slope up, baselines flat)
   - Accuracy vs. difficulty per method
   - Efficiency scatter (accuracy vs. tokens by difficulty)
   - Combined 2x2 summary figure

**Why this matters:** Without this analysis, we just have overall accuracy numbers.
The difficulty-aware breakdown is what makes the paper's argument convincing.

**How to run:**
```bash
# Just see the difficulty distribution (no model results needed):
python scripts/difficulty_analysis.py --difficulty-only

# Analyze results from the eval pipeline:
python scripts/difficulty_analysis.py --results-dir results/eval_distill

# Analyze specific result files:
python scripts/difficulty_analysis.py \
    --result-files results/eval_distill/cot_results.json \
                   results/eval_distill/adaptive_results.json
```

## Repository Structure

```
learning-when-to-think/
  configs/
    default.yaml          # Model, training, and eval hyperparameters
    chpc_grpo.yaml        # CHPC cluster config for GRPO training
  src/
    data/
      gsm8k.py            # Load GSM8K, extract/grade answers
      teacher.py           # Generate teacher traces (Gemini API or GSM8K gold)
    eval/
      evaluate.py          # Run baselines + adaptive, compute metrics
    policy/
      adaptive.py          # Core adaptive loop (continue/terminate prompting)
    train/
      sft.py               # Supervised fine-tuning on teacher traces
      dpo.py               # Direct Preference Optimization
      grpo.py              # Group Relative Policy Optimization (main RL)
  scripts/
    difficulty_analysis.py # Difficulty-aware evaluation and plots
    eval.py                # Full evaluation pipeline
    run_baselines.py       # Run CoT and self-consistency baselines
    plot_results.py        # Generate accuracy-vs-tokens plots
    train.py               # Main training script
  docs/
    summary.md             # This file
  data/
    teacher_traces.jsonl   # Teacher-generated reasoning traces
    dpo_pairs.jsonl        # Preference pairs for DPO
  chpc/
    train_grpo.slurm       # SLURM job script for university cluster
    eval.slurm             # SLURM job script for evaluation
```

## Related Papers

1. **Thinkless: LLM Learns When to Think** (arXiv: 2505.13417) - Very similar
   idea; the model learns when to invoke extended thinking vs. giving a direct
   answer. Our differentiator: we have a richer action space (4 actions vs 2).

2. **arXiv: 2506.05256** - Another related work on adaptive compute.

3. **ReAct: Synergizing Reasoning and Acting** (arXiv: 2210.03629) - Framework
   for interleaving reasoning and actions. Conceptual ancestor of our approach.

## Key Links

- **Shared GitHub repo:** https://github.com/marieluofu25/learning-when-to-think
- **Overleaf (paper):** https://www.overleaf.com/4325339351tzbhkgfdctzj#9eb5f6
- **Dataset:** GSM8K (`openai/gsm8k` on HuggingFace)
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`

## Timeline

- **April 1**: Proposal due
- **End of semester**: Final paper + presentation
- **Paper format**: Using class-provided template on Overleaf
