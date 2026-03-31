"""Competition-style math data + grading utilities.

This is a thin wrapper around the MATH-500 implementation, kept to match the
proposal/tex naming used in the project plan.
"""

from __future__ import annotations

from typing import Any

from src.data.math_500 import (
    extract_hash_answer,
    extract_predicted_answer,
    grade_answer as grade_math_answer,
    load_math_500,
    normalize_math_answer,
)


def load_math500(split: str = "test", subset_size: int | None = None) -> list[dict[str, Any]]:
    """Alias for MATH-500 loader."""
    return load_math_500(split=split, subset_size=subset_size)


__all__ = [
    "extract_hash_answer",
    "extract_predicted_answer",
    "grade_math_answer",
    "load_math500",
    "normalize_math_answer",
]

