"""HumanEval loading and pass@1 grading via isolated execution."""

from __future__ import annotations

import multiprocessing
import re
from typing import Any

from datasets import load_dataset

from src.policy.tool_exec import extract_python_from_response

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_TERMINATE = "<terminate>"


def extract_humaneval_completion(model_output: str) -> str:
    """Prefer last ```python``` block; else text after <terminate>; else full stripped output."""
    m = extract_python_from_response(model_output)
    if m:
        return m.strip()
    if _TERMINATE in model_output:
        tail = model_output.split(_TERMINATE, 1)[-1].strip()
        m2 = _FENCE_RE.search(tail)
        if m2:
            return m2.group(1).strip()
        return tail
    return model_output.strip()


def load_humaneval(subset_size: int | None = None) -> list[dict]:
    """Rows: task_id, prompt, test, entry_point, canonical_solution (reference)."""
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        ds = load_dataset("openai_humaneval", split="test")
    items = []
    for row in ds:
        items.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "test": row["test"],
            "entry_point": row["entry_point"],
            "canonical_solution": row.get("canonical_solution", ""),
        })
    if subset_size is not None:
        items = items[:subset_size]
    return items


def _grade_worker(code: str, test: str, entry: str, queue: multiprocessing.Queue) -> None:
    try:
        ns: dict[str, Any] = {}
        exec(compile(code, "<humaneval>", "exec"), ns, ns)
        exec(compile(test, "<humaneval_test>", "exec"), ns, ns)
        cand = ns[entry]
        ns["check"](cand)
        queue.put(("ok", True))
    except Exception:
        queue.put(("ok", False))


def passes_humaneval(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    timeout_sec: float = 10.0,
) -> bool:
    """Execute prompt+completion against HumanEval `test` checker in a subprocess."""
    code = f"{prompt}{completion}"
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(target=_grade_worker, args=(code, test, entry_point, q))
    p.start()
    p.join(timeout=timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join(timeout=2.0)
        if p.is_alive():
            p.kill()
        return False
    try:
        _kind, ok = q.get_nowait()
        return bool(ok)
    except Exception:
        return False
