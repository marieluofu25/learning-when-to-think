"""Optional Hugging Face Hub token from YAML before any ``from_pretrained`` calls."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def _read_token_from_file(path: str) -> str | None:
    p = Path(os.path.expandvars(path)).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"hf_token_file not found: {p}")
    token = p.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"hf_token_file is empty: {p}")
    return token


def _apply_token_from_mapping(cfg: Mapping[str, Any]) -> str | None:
    tok = cfg.get("hf_token")
    if isinstance(tok, str) and tok.strip():
        return tok.strip()

    path = cfg.get("hf_token_file")
    if path is None or (isinstance(path, str) and not path.strip()):
        return None
    if not isinstance(path, str):
        return None
    return _read_token_from_file(path)


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML: {path}")
    return data


def apply_hf_hub_config(cfg: Mapping[str, Any]) -> None:
    """Set ``HF_TOKEN`` from config if not already set in the environment.

    Precedence: existing non-empty ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` wins.
    Then active config values (``hf_token`` / ``hf_token_file``), then optional
    ``configs/hf_secrets.yaml`` (same keys) if present.
    """
    if (os.environ.get("HF_TOKEN") or "").strip() or (
        os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    ).strip():
        return

    token = _apply_token_from_mapping(cfg)
    if token:
        os.environ["HF_TOKEN"] = token
        return

    repo_root = Path(__file__).resolve().parents[2]
    secrets_path = repo_root / "configs" / "hf_secrets.yaml"
    if not secrets_path.is_file():
        return

    secrets_cfg = _load_yaml_mapping(secrets_path)
    token = _apply_token_from_mapping(secrets_cfg)
    if token:
        os.environ["HF_TOKEN"] = token
