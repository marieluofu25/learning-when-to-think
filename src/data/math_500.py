"""MATH-500 dataset loading + answer extraction/normalization + grading.

MATH-500 answers are LaTeX/math expressions (not always plain numeric floats).
We grade by normalized exact string match on the final answer extracted from
the model output.
"""

from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset


_HASH_LINE_RE = re.compile(r"^\s*####\s*(?P<ans>.+?)\s*$", re.MULTILINE)
_HASH_INLINE_RE = re.compile(r"####\s*(?P<ans>[^\n\r]+)")


def load_math_500(split: str = "test", subset_size: int | None = None) -> list[dict[str, Any]]:
    """Load HuggingFace `HuggingFaceH4/MATH-500` and return training/eval items.

    The underlying dataset only has a `test` split. We still accept `split`
    for API compatibility; any other split value will map to `test`.
    """

    # The dataset server exposes only `test`; keep this mapping explicit.
    hf_split = "test" if split not in {"test", "train"} else "test"

    ds = load_dataset("HuggingFaceH4/MATH-500", "default", split=hf_split)
    items: list[dict[str, Any]] = []
    for row in ds:
        gold = row["answer"]
        items.append(
            {
                "question": row["problem"],
                # Keep the historical key name used across the repo, but it now
                # contains an answer expression, not a numeric float.
                "answer_number": normalize_math_answer(gold),
                "full_answer": gold,
                "solution": row.get("solution"),
                "subject": row.get("subject"),
                "level": row.get("level"),
                "unique_id": row.get("unique_id"),
            }
        )

    if subset_size is not None:
        items = items[:subset_size]
    return items


def normalize_math_answer(s: str) -> str:
    """Normalize a LaTeX/math answer string for exact-match grading."""
    s = (s or "").strip()

    # Strip surrounding $...$ if present.
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()

    # Common LaTeX macro variants.
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # Remove size delimiters that often vary between generations.
    s = s.replace("\\left", "").replace("\\right", "")

    # Remove whitespace and newlines.
    s = re.sub(r"\s+", "", s)
    return s


def extract_hash_answer(text: str) -> str | None:
    """Extract the final `#### <answer>` expression from a model output."""
    if not text:
        return None
    matches = list(_HASH_LINE_RE.finditer(text))
    if not matches:
        inline = list(_HASH_INLINE_RE.finditer(text))
        if not inline:
            return None
        return normalize_math_answer(inline[-1].group("ans").strip())
    return normalize_math_answer(matches[-1].group("ans"))


def _extract_last_boxed(text: str) -> str | None:
    """Extract the last `\\boxed{...}` content, using brace matching."""
    if not text:
        return None

    needle = r"\boxed{"
    idx = text.rfind(needle)
    if idx == -1:
        return None

    i = idx + len(needle)
    depth = 1
    start = i
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None
    return normalize_math_answer(text[start : i - 1])


def extract_predicted_answer(text: str) -> str | None:
    """Extract the model's final answer expression.

    Priority:
    1) `#### ...` on its own line (repo standard for adaptive/COT prompts)
    2) last `\\boxed{...}` occurrence
    3) last non-empty line (best-effort fallback)
    """
    h = extract_hash_answer(text)
    if h is not None:
        return h

    b = _extract_last_boxed(text)
    if b is not None:
        return b

    # Fallback: last non-empty line.
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    return normalize_math_answer(lines[-1])


def has_valid_format(text: str) -> bool:
    """Competition-style format: one `####` answer line and/or a final `\\boxed{...}`."""
    if not text:
        return False
    if text.count("####") == 1 and extract_hash_answer(text) is not None:
        return True
    if "####" not in text and _extract_last_boxed(text) is not None:
        return True
    return False


def grade_answer(predicted: str | None, gold: str) -> bool:
    """Exact-match grading after normalization."""
    if predicted is None:
        return False
    return normalize_math_answer(predicted) == normalize_math_answer(gold)

