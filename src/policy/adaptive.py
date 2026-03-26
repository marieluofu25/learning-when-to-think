"""Adaptive reasoning loop: continue / terminate via prompt-based action tokens."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

SYSTEM_PROMPT = (
    "You are a math problem solver. You solve problems step by step. "
    "After each reasoning step, you must decide: output <continue> to keep "
    "reasoning, or <terminate> to give your final numerical answer.\n"
    "When you output <terminate>, end your solution with #### <number> "
    "on its own line, where <number> is your final numerical answer."
)

DECISION_PROMPT = "\nShould I continue reasoning or give my final answer? Output <continue> or <terminate>: "

CONTINUE_TOKEN = "<continue>"
TERMINATE_TOKEN = "<terminate>"


def build_initial_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


@torch.inference_mode()
def adaptive_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_steps: int = 5,
    max_tokens_per_step: int = 256,
    temperature: float = 0.7,
    return_full_trace: bool = False,
) -> dict:
    """Run the adaptive continue/terminate loop for a single question.

    Returns dict with keys:
        answer_text: the final answer text after <terminate>
        total_tokens: total tokens generated across all steps
        num_steps: number of reasoning steps taken
        trace: (optional) list of step outputs
    """
    messages = build_initial_messages(question)
    trace: list[str] = []
    total_tokens = 0

    for step in range(max_steps):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens_per_step,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        new_tokens = outputs[0][input_len:]
        total_tokens += len(new_tokens)
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        trace.append(response)

        if TERMINATE_TOKEN in response:
            after_terminate = response.split(TERMINATE_TOKEN, 1)[1].strip()
            answer_text = after_terminate if after_terminate else response
            result = {
                "answer_text": answer_text,
                "total_tokens": total_tokens,
                "num_steps": step + 1,
                "terminated": True,
            }
            if return_full_trace:
                result["trace"] = trace
            return result

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": DECISION_PROMPT})

    combined = " ".join(trace)
    return {
        "answer_text": combined,
        "total_tokens": total_tokens,
        "num_steps": max_steps,
        "terminated": False,
        **({"trace": trace} if return_full_trace else {}),
    }


@torch.inference_mode()
def cot_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> dict:
    """Single-pass chain-of-thought baseline."""
    messages = [
        {"role": "system", "content": "You are a math problem solver. Solve step by step. End your solution with #### <number> on its own line, where <number> is your final numerical answer."},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "answer_text": response,
        "total_tokens": len(new_tokens),
        "num_steps": 1,
    }
