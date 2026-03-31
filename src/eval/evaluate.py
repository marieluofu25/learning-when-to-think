"""Evaluation pipeline: run baselines and adaptive policy, collect metrics."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

from src.data.math_500 import (
    extract_hash_answer,
    extract_predicted_answer,
    grade_answer,
    has_valid_format,
)
from src.data.humaneval import extract_humaneval_completion, passes_humaneval


def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n_total == 0:
        return (0.0, 0.0)
    p_hat = n_correct / n_total
    denom = 1 + z * z / n_total
    center = (p_hat + z * z / (2 * n_total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n_total)) / n_total) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def _merge_action_counts(results: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        ac = r.get("action_counts")
        if not ac:
            continue
        for k, v in ac.items():
            out[k] = out.get(k, 0) + int(v)
    return out


def evaluate_results(results: list[dict], *, dataset_kind: str = "gsm8k") -> dict:
    """Compute aggregate metrics from a list of per-problem results."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    total_tokens = sum(r["total_tokens"] for r in results)
    avg_tokens = total_tokens / max(total, 1)
    accuracy = correct / max(total, 1)
    cost_per_correct = total_tokens / max(correct, 1)
    avg_steps = sum(r.get("num_steps", 1) for r in results) / max(total, 1)
    total_tool_calls = sum(int(r.get("n_tool_calls", 0)) for r in results)

    texts = [r.get("answer_text", "") for r in results]
    if dataset_kind in {"gsm8k", "math_500"}:
        format_ok = sum(1 for t in texts if has_valid_format(t))
        parse_fail = sum(
            1 for t in texts
            if extract_hash_answer(t) is None and extract_predicted_answer(t) is None
        )
        multi_hash = sum(1 for t in texts if t.count("####") > 1)
    else:
        format_ok = parse_fail = multi_hash = 0

    ci_lo, ci_hi = wilson_ci(correct, total)
    action_counts_total = _merge_action_counts(results)

    out = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "total_tokens": total_tokens,
        "avg_tokens_per_problem": avg_tokens,
        "cost_per_correct_answer": cost_per_correct,
        "avg_steps": avg_steps,
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls_per_problem": total_tool_calls / max(total, 1),
        "action_counts_total": action_counts_total,
    }
    if dataset_kind in {"gsm8k", "math_500"}:
        out["format_success_rate"] = format_ok / max(total, 1)
        out["parse_fail_rate"] = parse_fail / max(total, 1)
        out["multi_answer_rate"] = multi_hash / max(total, 1)
    return out


def run_cot_baseline(
    model,
    tokenizer,
    dataset: list[dict],
    stage_name: str = "CoT",
    pulse_every: int = 1,
    **gen_kwargs,
) -> list[dict]:
    from src.policy.adaptive import cot_generate

    results = []
    total = len(dataset)
    for idx, item in enumerate(dataset, start=1):
        out = cot_generate(model, tokenizer, item["question"], **gen_kwargs)
        predicted = extract_predicted_answer(out["answer_text"])
        correct = grade_answer(predicted, item["answer_number"])
        results.append({
            "question": item["question"],
            "gold": item["answer_number"],
            "predicted": predicted,
            "correct": correct,
            "answer_text": out["answer_text"],
            "total_tokens": out["total_tokens"],
            "num_steps": out["num_steps"],
        })
        if idx % max(pulse_every, 1) == 0 or idx == total:
            print(f"[progress] {stage_name}: {idx}/{total}", flush=True)
    return results


def run_direct_baseline(
    model,
    tokenizer,
    dataset: list[dict],
    stage_name: str = "Direct",
    pulse_every: int = 1,
    **gen_kwargs,
) -> list[dict]:
    from src.policy.adaptive import direct_generate

    results = []
    total = len(dataset)
    for idx, item in enumerate(dataset, start=1):
        out = direct_generate(model, tokenizer, item["question"], **gen_kwargs)
        predicted = extract_predicted_answer(out["answer_text"])
        correct = grade_answer(predicted, item["answer_number"])
        results.append({
            "question": item["question"],
            "gold": item["answer_number"],
            "predicted": predicted,
            "correct": correct,
            "answer_text": out["answer_text"],
            "total_tokens": out["total_tokens"],
            "num_steps": out["num_steps"],
        })
        if idx % max(pulse_every, 1) == 0 or idx == total:
            print(f"[progress] {stage_name}: {idx}/{total}", flush=True)
    return results


def run_self_consistency(
    model,
    tokenizer,
    dataset: list[dict],
    k: int = 5,
    stage_name: str = "Self-Consistency",
    pulse_every: int = 1,
    **gen_kwargs,
) -> list[dict]:
    """Fixed self-consistency: sample k CoT answers, majority vote."""
    from src.policy.adaptive import cot_generate

    results = []
    total = len(dataset)
    for idx, item in enumerate(dataset, start=1):
        predictions: list[str | None] = []
        token_counts: list[int] = []
        for _ in range(k):
            out = cot_generate(model, tokenizer, item["question"], **gen_kwargs)
            pred = extract_predicted_answer(out["answer_text"])
            predictions.append(pred)
            token_counts.append(out["total_tokens"])

        valid = [p for p in predictions if p is not None]
        if valid:
            counter = Counter(valid)
            majority = counter.most_common(1)[0][0]
        else:
            majority = None

        correct = grade_answer(majority, item["answer_number"])
        results.append({
            "question": item["question"],
            "gold": item["answer_number"],
            "predicted": majority,
            "correct": correct,
            "total_tokens": sum(token_counts),
            "num_steps": k,
            "all_predictions": predictions,
        })
        if idx % max(pulse_every, 1) == 0 or idx == total:
            print(f"[progress] {stage_name}: {idx}/{total}", flush=True)
    return results


def run_adaptive_policy(
    model,
    tokenizer,
    dataset: list[dict],
    stage_name: str = "Adaptive",
    pulse_every: int = 1,
    allow_refine: bool = True,
    allow_verify: bool | None = None,
    system_prompt: str | None = None,
    **gen_kwargs,
) -> list[dict]:
    from src.policy.adaptive import adaptive_generate

    if allow_verify is not None:
        allow_refine = allow_verify

    results = []
    total = len(dataset)
    for idx, item in enumerate(dataset, start=1):
        out = adaptive_generate(
            model,
            tokenizer,
            item["question"],
            allow_refine=allow_refine,
            system_prompt=system_prompt,
            **gen_kwargs,
        )
        predicted = extract_predicted_answer(out["answer_text"])
        correct = grade_answer(predicted, item["answer_number"])
        results.append({
            "question": item["question"],
            "gold": item["answer_number"],
            "predicted": predicted,
            "correct": correct,
            "answer_text": out["answer_text"],
            "total_tokens": out["total_tokens"],
            "num_steps": out["num_steps"],
            "terminated": out.get("terminated", False),
            "action_counts": out.get("action_counts", {}),
        })
        if idx % max(pulse_every, 1) == 0 or idx == total:
            print(f"[progress] {stage_name}: {idx}/{total}", flush=True)
    return results


def run_cot_humaneval(
    model,
    tokenizer,
    dataset: list[dict],
    stage_name: str = "CoT-HE",
    pulse_every: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> list[dict]:
    from src.policy.adaptive import cot_generate

    results = []
    n = len(dataset)
    for idx, item in enumerate(dataset, start=1):
        out = cot_generate(model, tokenizer, item["prompt"], max_tokens=max_tokens, temperature=temperature)
        completion = extract_humaneval_completion(out["answer_text"])
        ok = passes_humaneval(item["prompt"], completion, item["test"], item["entry_point"])
        results.append({
            "task_id": item["task_id"],
            "correct": ok,
            "answer_text": completion,
            "total_tokens": out["total_tokens"],
            "num_steps": out["num_steps"],
        })
        if idx % max(pulse_every, 1) == 0 or idx == n:
            print(f"[progress] {stage_name}: {idx}/{n}", flush=True)
    return results


def run_adaptive_humaneval(
    model,
    tokenizer,
    dataset: list[dict],
    stage_name: str = "Adaptive-HE",
    pulse_every: int = 1,
    allow_refine: bool = True,
    **gen_kwargs,
) -> list[dict]:
    from src.policy.adaptive import (
        CONTINUE_TOKEN,
        REFINE_TOKEN,
        TERMINATE_TOKEN,
        adaptive_generate,
    )

    he_system = (
        "You are an expert Python programmer completing HumanEval-style tasks.\n"
        "Each assistant message must start by choosing exactly one action on its own line:\n"
        f"- {CONTINUE_TOKEN} — more reasoning or partial code\n"
        f"- {REFINE_TOKEN} — re-read the spec and your last step\n"
        f"- {TERMINATE_TOKEN} — final answer: output the **complete function body** that fits "
        "the given signature/docstring (only the indented body lines, or a full ```python block).\n"
        "The grader concatenates your completion after the task prompt."
    )

    results = []
    n = len(dataset)
    for idx, item in enumerate(dataset, start=1):
        out = adaptive_generate(
            model,
            tokenizer,
            item["prompt"],
            allow_refine=allow_refine,
            system_prompt=he_system,
            **gen_kwargs,
        )
        completion = extract_humaneval_completion(out["answer_text"])
        ok = passes_humaneval(item["prompt"], completion, item["test"], item["entry_point"])
        results.append({
            "task_id": item["task_id"],
            "correct": ok,
            "answer_text": completion,
            "total_tokens": out["total_tokens"],
            "num_steps": out["num_steps"],
            "terminated": out.get("terminated", False),
            "action_counts": out.get("action_counts", {}),
        })
        if idx % max(pulse_every, 1) == 0 or idx == n:
            print(f"[progress] {stage_name}: {idx}/{n}", flush=True)
    return results


def save_results(results: list[dict], metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"metrics": metrics, "results": results, "timestamp": time.time()}
    path.write_text(json.dumps(data, indent=2, default=str))
