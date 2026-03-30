"""Shared HuggingFace model loading for training and evaluation."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _parse_dtype(name: str) -> torch.dtype:
    m = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return m.get(name.lower().replace("torch.", ""), torch.bfloat16)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_causal_lm(
    model_name: str,
    *,
    use_qlora: bool = False,
    torch_dtype_name: str = "bfloat16",
    device_map: str | dict = "auto",
) -> AutoModelForCausalLM:
    """Load base causal LM; optional 4-bit QLoRA preparation."""
    dtype = _parse_dtype(torch_dtype_name)
    if use_qlora:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map=device_map,
            dtype=dtype,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
    )
