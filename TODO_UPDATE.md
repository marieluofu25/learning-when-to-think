# TODO_UPDATE — Learning When to Think (Post-Pivot)

This file supersedes the narrative in `TODO.md` for the **Apr 2026 pivot**. Keep `TODO.md` as historical checklist if needed; use this doc for **current priorities**.

## Project framing (pivot)

**Primary research question:** Do **efficiency gains** come mainly from **ALP reward shaping**, not from **action tokens**?

**Secondary (comparison / diagnostic):** Under realistic decoding, do **`<refine>` / `<terminate>`** (with/without **hard masking**) change accuracy/length materially, or do they behave like a **non-emergent control interface**?

**Non-goal (for the final story):** Proving action tokens “make the model smarter.” If that fails, treat it as a **valid negative result** with careful analysis.

### Captured technical decisions

- **`<continue>` is a no-op** for now: autoregressive continuation already defaults to “keep writing.” Prioritize **`<refine>` + `<terminate>`** in training/eval storytelling.
- **Hard masking** (MVP: **first generated token** constrained to an action token) is an **intervention**; label results as **forced-choice**, not “natural emergence.” Enable via `constrain_action_first_token: true` in YAML (`scripts/train.py`, `scripts/eval.py`).
- **LoRA only** on **Qwen2.5-Math-7B-Instruct** (attention + MLP projections, **r=16, α=32**): full fine-tune is out of scope for compute + forgetting risk.
- **DPO** (Direct Preference Optimization) is the preference-style track in this repo ([`src/train/dpo.py`](src/train/dpo.py)) — **not** “DAPO.”

### Two evaluation / training tracks

| Track | Role | Entrypoints |
|------|------|-------------|
| **Main (Transformers)** | GRPO + ALP + DeGRPO, LoRA, adaptive rollout, CoT/direct baselines | [`scripts/train.py`](scripts/train.py), [`scripts/eval.py`](scripts/eval.py), [`src/train/grpo.py`](src/train/grpo.py), [`src/policy/adaptive.py`](src/policy/adaptive.py) |
| **Pivot (vLLM)** | Fast MATH-500 eval / rollouts | [`scripts/eval_pivot.py`](scripts/eval_pivot.py), [`src/pivot/`](src/pivot/) |

### CHPC (University of Utah)

- **Single operator script:** [`chpc/run_chpc.sh`](chpc/run_chpc.sh) — subcommands `train-grpo`, `eval`, `train-sft`, `train-dpo`.
- **Artifacts:** all batch outputs under **`chpc/results/`** (`logs/`, `checkpoints/`, `eval/`, `sft/`, `dpo/`) for easy `rsync`.
- Details: [`chpc/README.md`](chpc/README.md).

---

## Roles (time-crunch split)

- **Hao:** training runs (SFT / **DPO** / GRPO), checkpoints, run logs.
- **Ammon:** eval harness execution, results packaging, plots/tables for slides.
- **Ivan:** final report + presentation narrative (especially negative results + limitations).

---

## Stage 1 — Foundations

### Pivot track (vLLM + shared types)

- [x] `src/pivot/eval_types.py` — shared types (`RolloutResult`, `EvalMetrics`)
- [x] `src/pivot/reward.py` — ALP reward primitives (`compute_alp_rewards*`)
- [x] `src/pivot/eval_harness.py` — accuracy, tokens, cost-per-correct, breakdowns, action usage diagnostics
- [x] `scripts/eval_pivot.py` — MATH-500 eval driver (vLLM) + YAML config
- [x] `configs/eval_qwen_math_7b.yaml`
- [x] `src/data/math.py` — MATH loaders + grading (pivot / legacy helpers)
- [x] Tests for pivot reward + eval (`tests/test_pivot_*.py`)

### Main track (Transformers)

- [x] `src/data/math_500.py` — MATH-500 load + extraction + grading
- [x] `src/eval/evaluate.py` — baselines + adaptive metrics (`action_counts` per rollout)
- [x] `scripts/eval.py` — CoT + **direct-answer** + adaptive (with/without refine) on MATH-500
- [x] `scripts/run_baselines.py` — CoT + self-consistency

### SFT dataset / token plumbing (3-action JSONL)

- [x] `src/pivot/tokens.py` — `<continue>`, `<refine>`, `<terminate>` + embedding hooks
- [x] `scripts/generate_rollouts_pivot.py` — grouped rollouts on MATH train
- [x] `scripts/generate_sft_3action.py` — LLM rewrite dataset builder
- [x] Produced SFT JSONL + quality checks (per prior run notes in `TODO.md`)
- [x] `scripts/train_sft_3action.py` + `configs/chpc_sft_3action.yaml` — SFT on JSONL `messages` (LoRA / optional QLoRA)

### Baselines (pivot story)

- [x] CoT-style eval: `scripts/eval_pivot.py` **and** `scripts/eval.py` (`run_cot_baseline`)
- [x] **Direct-answer baseline** — implemented in `scripts/eval.py` (`run_direct_baseline`). For **full MATH-500:** set `eval_subset_size: 500` in config (e.g. [`configs/default.yaml`](configs/default.yaml)).
- [x] **Decoding / eval knobs** — documented in root [`README.md`](README.md) (key YAML fields + note on hardcoded CoT/direct temperatures in `cot_generate` / `direct_generate` unless extended)

---

## Stage 2 — Training + integration (ALP-first)

### LoRA + GRPO (main track)

- [x] LoRA on **Qwen2.5-Math-7B-Instruct**: attention + MLP projections (**r=16, α=32**) — [`src/train/grpo.py`](src/train/grpo.py) `setup_lora`
- [x] **GRPO + ALP + DeGRPO** — [`src/train/grpo.py`](src/train/grpo.py), [`scripts/train.py`](scripts/train.py)
- [x] **SFT warmup** on 3-action JSONL — `scripts/train_sft_3action.py` (separate from teacher-trace [`src/train/sft.py`](src/train/sft.py))
- [ ] **Checkpoint naming + run manifest** — optional: log git hash, config path, seed in `chpc/results/` (manual or small logging script)
- [ ] **CHPC production run** — operator submits via `chpc/run_chpc.sh` (execution, not code)

### DPO (preference optimization)

- [x] Library + CLI: [`src/train/dpo.py`](src/train/dpo.py), [`scripts/train_dpo.py`](scripts/train_dpo.py), [`configs/chpc_dpo.yaml`](configs/chpc_dpo.yaml)
- [ ] **7B + QLoRA alignment** — optional stretch; default CHPC config may still use smaller model until extended

### Controlled generation / masking

- [x] **Hard masking MVP** — first-token constraint to action vocabulary in [`src/policy/adaptive.py`](src/policy/adaptive.py); flag `constrain_action_first_token` in train/eval YAML
- [x] Eval logs **`action_counts`** per problem via adaptive rollout (main track); pivot `RolloutResult.actions` remains separate if you unify later

### End-to-end

- [x] Smoke: train artifact → `scripts/eval.py` → metrics under `chpc/results/eval/` when using CHPC wrapper
- [x] Repro: documented in [`README.md`](README.md) + [`chpc/README.md`](chpc/README.md)

---

## Stage 3 — Experiments (comparison table is the deliverable)

### Core comparisons (must-have)

- [ ] **Base model** (CoT prompt) — MATH-500
- [ ] **+ LoRA SFT** (action tokens) — compare **with vs without** `constrain_action_first_token`
- [ ] **+ GRPO (ALP)** checkpoint vs baselines
- [ ] **+ DPO** (if time) vs same baselines
- [ ] **Direct-answer** baseline (already in `scripts/eval.py`)

### Ablations / diagnostics (should-have)

- [ ] **`allow_refine=false`** (eval already runs adaptive with and without refine)
- [ ] **β sweep** for ALP
- [ ] **DeGRPO** — `degrpo: false` in YAML for vanilla GRPO

### Hypotheses, plots, negative results

- [ ] **H1 / H2 / H3** — as in prior TODO_UPDATE
- [ ] Plots / tables — Ammon
- [ ] Negative-results writeup — Ivan

---

## Stage 4 — Presentation (Apr 19–20)

- [ ] Slides: pivot narrative + comparison table + limitations
- [ ] Rehearsal
- [ ] Present Apr 20

---
