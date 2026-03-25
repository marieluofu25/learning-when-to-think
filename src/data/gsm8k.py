"""GSM8K dataset loading and answer extraction."""

from __future__ import annotations

import re
from datasets import load_dataset


def load_gsm8k(split: str = "test", subset_size: int | None = None) -> list[dict]:
    """Load GSM8K and return list of {question, answer_number, full_answer}."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    items = []
    for row in ds:
        number = extract_answer_number(row["answer"])
        items.append({
            "question": row["question"],
            "answer_number": number,
            "full_answer": row["answer"],
        })
    if subset_size is not None:
        items = items[:subset_size]
    return items


_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")


def extract_answer_number(answer_text: str) -> float:
    """Extract the numerical answer after #### in GSM8K format."""
    m = _ANSWER_RE.search(answer_text)
    if m:
        return float(m.group(1).replace(",", ""))
    raise ValueError(f"Cannot extract answer from: {answer_text!r}")


_NUMBER_RE = re.compile(r"-?[\d,]+\.?\d*")


def extract_predicted_number(text: str) -> float | None:
    """Extract the last number from model output as the predicted answer."""
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    cleaned = matches[-1].replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def grade_answer(predicted: float | None, gold: float, tol: float = 1e-3) -> bool:
    if predicted is None:
        return False
    return abs(predicted - gold) < tol
