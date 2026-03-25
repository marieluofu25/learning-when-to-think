"""Evaluation pipeline: run baselines and adaptive policy, collect metrics."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from src.data.gsm8k import extract_predicted_number, grade_answer


def evaluate_results(results: list[dict]) -> dict:
    """Compute aggregate metrics from a list of per-problem results."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    total_tokens = sum(r["total_tokens"] for r in results)
    avg_tokens = total_tokens / max(total, 1)
    accuracy = correct / max(total, 1)
    cost_per_correct = total_tokens / max(correct, 1)
    avg_steps = sum(r.get("num_steps", 1) for r in results) / max(total, 1)

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "avg_tokens_per_problem": avg_tokens,
        "cost_per_correct_answer": cost_per_correct,
        "avg_steps": avg_steps,
    }


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
        predicted = extract_predicted_number(out["answer_text"])
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
        predictions: list[float | None] = []
        token_counts: list[int] = []
        for _ in range(k):
            out = cot_generate(model, tokenizer, item["question"], **gen_kwargs)
            pred = extract_predicted_number(out["answer_text"])
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
    **gen_kwargs,
) -> list[dict]:
    from src.policy.adaptive import adaptive_generate

    results = []
    total = len(dataset)
    for idx, item in enumerate(dataset, start=1):
        out = adaptive_generate(model, tokenizer, item["question"], **gen_kwargs)
        predicted = extract_predicted_number(out["answer_text"])
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
        })
        if idx % max(pulse_every, 1) == 0 or idx == total:
            print(f"[progress] {stage_name}: {idx}/{total}", flush=True)
    return results


def save_results(results: list[dict], metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"metrics": metrics, "results": results, "timestamp": time.time()}
    path.write_text(json.dumps(data, indent=2, default=str))
