"""Optional Hugging Face Hub token from YAML before any ``from_pretrained`` calls."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def apply_hf_hub_config(cfg: Mapping[str, Any]) -> None:
    """Set ``HF_TOKEN`` from config if not already set in the environment.

    Precedence: existing non-empty ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` wins.
    Then ``hf_token`` (literal string in YAML), then ``hf_token_file`` (one line).
    """
    if (os.environ.get("HF_TOKEN") or "").strip() or (
        os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    ).strip():
        return
    tok = cfg.get("hf_token")
    if isinstance(tok, str) and tok.strip():
        os.environ["HF_TOKEN"] = tok.strip()
        return
    path = cfg.get("hf_token_file")
    if path is None or (isinstance(path, str) and not path.strip()):
        return
    if not isinstance(path, str):
        return
    p = Path(os.path.expandvars(path)).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"hf_token_file not found: {p}")
    token = p.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"hf_token_file is empty: {p}")
    os.environ["HF_TOKEN"] = token
