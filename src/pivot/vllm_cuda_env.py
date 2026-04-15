"""CUDA env tweaks for vLLM on clusters that expose MIG UUIDs."""

from __future__ import annotations

import os


def normalize_cuda_visible_devices_for_vllm() -> None:
    """Remap Slurm MIG UUIDs so vLLM does not call ``int()`` on ``CUDA_VISIBLE_DEVICES``.

    Some vLLM releases crash with::
        ValueError: invalid literal for int() with base 10: 'MIG-...'

    For typical single-GPU Slurm steps, the allocated device is visible as index ``0``
    after this remap.
    """
    cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if cvd.startswith("MIG-"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
