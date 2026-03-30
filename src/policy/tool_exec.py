"""Sandboxed Python execution for CALL_TOOL (timeout, no network, restricted builtins)."""

from __future__ import annotations

import ast
import io
import multiprocessing
import re
import textwrap
from typing import Any

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_from_response(text: str) -> str | None:
    m = _CODE_FENCE_RE.search(text)
    if m:
        return textwrap.dedent(m.group(1)).strip()
    return None


class _RestrictingVisitor(ast.NodeVisitor):
    """Disallow imports, attribute access on dunder, and a few risky constructs."""

    _FORBIDDEN_TYPES = frozenset(
        {
            ast.Import,
            ast.ImportFrom,
            ast.ClassDef,
            ast.AsyncFunctionDef,
            ast.Delete,
            ast.Try,
            ast.With,
            ast.Raise,
            ast.Assert,
            ast.Global,
            ast.Nonlocal,
        }
    )

    def generic_visit(self, node: ast.AST) -> Any:
        if type(node) in self._FORBIDDEN_TYPES:
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str) and node.attr.startswith("_"):
            raise ValueError("access to private attributes is not allowed")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("open", "exec", "eval", "__import__", "compile"):
                raise ValueError(f"disallowed call: {node.func.id}")
        return super().generic_visit(node)


def _validate_ast(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    _RestrictingVisitor().visit(tree)


def _safe_builtins() -> dict[str, Any]:
    import builtins

    allowed = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "pow",
        "print",
        "range",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "True",
        "False",
        "None",
    }
    b = getattr(builtins, "__dict__", {})
    return {k: b[k] for k in allowed if k in b}


def _run_worker(code: str, queue: multiprocessing.Queue) -> None:
    buf = io.StringIO()
    try:
        _validate_ast(code)
        g: dict[str, Any] = {"__builtins__": _safe_builtins()}
        import contextlib
        import sys

        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<tool>", "exec"), g, g)
        queue.put(("ok", buf.getvalue()))
    except Exception as e:
        queue.put(("err", f"{type(e).__name__}: {e}"))


def run_python_sandboxed(code: str, timeout_sec: float = 5.0) -> tuple[bool, str]:
    """Execute *code* in a subprocess with timeout. Returns (success, stdout_or_error)."""
    code = code.strip()
    if not code:
        return False, "empty code"
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(target=_run_worker, args=(code, q))
    p.start()
    p.join(timeout=timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join(timeout=2.0)
        if p.is_alive():
            p.kill()
        return False, f"timeout after {timeout_sec}s"
    try:
        kind, payload = q.get_nowait()
    except Exception:
        return False, "no result from sandbox"
    if kind == "ok":
        return True, (payload or "").strip() or "(no output)"
    return False, str(payload)
