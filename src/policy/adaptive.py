"""Adaptive reasoning: five meta-actions (continue, verify, sample_alt, call_tool, terminate)."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.policy.tool_exec import extract_python_from_response, run_python_sandboxed

CONTINUE_TOKEN = "<continue>"
VERIFY_TOKEN = "<verify>"
SAMPLE_ALT_TOKEN = "<sample_alt>"
CALL_TOOL_TOKEN = "<call_tool>"
TERMINATE_TOKEN = "<terminate>"

SYSTEM_PROMPT = (
    "You are a math problem solver. Work step by step.\n"
    "Each assistant message must start by choosing exactly one action on its own line:\n"
    f"- {CONTINUE_TOKEN} — add more chain-of-thought\n"
    f"- {VERIFY_TOKEN} — double-check the last step\n"
    f"- {SAMPLE_ALT_TOKEN} — discard this approach and try a different one\n"
    f"- {CALL_TOOL_TOKEN} — run Python (put code in a ```python fenced block) for arithmetic\n"
    f"- {TERMINATE_TOKEN} — final answer: end with #### <number> on its own line\n"
    "After the action line, write your reasoning (and code block if using call_tool)."
)

DECISION_PROMPT = (
    "\nChoose the next action. Reply starting with one of: "
    f"{CONTINUE_TOKEN}, {VERIFY_TOKEN}, {SAMPLE_ALT_TOKEN}, {CALL_TOOL_TOKEN}, or {TERMINATE_TOKEN}."
)

VERIFY_USER_PROMPT = (
    "Double-check your previous reasoning step. Fix any mistake, then continue "
    f"(still starting your reply with an action token: {CONTINUE_TOKEN}, {VERIFY_TOKEN}, etc.)."
)

TOOL_FAILURE_PROMPT = "Tool execution failed or was empty. Fix the code or continue without tools."

DISABLED_TOOL_USER_PROMPT = (
    "Tool use is disabled for this run. Continue reasoning without CALL_TOOL "
    f"(start with {CONTINUE_TOKEN} or another allowed action)."
)

CODING_SYSTEM_PROMPT = (
    "You are an expert Python programmer completing HumanEval-style tasks.\n"
    "Each assistant message must start by choosing exactly one action on its own line:\n"
    f"- {CONTINUE_TOKEN} — more reasoning or partial code\n"
    f"- {VERIFY_TOKEN} — re-read the spec and your last step\n"
    f"- {SAMPLE_ALT_TOKEN} — discard and try a different approach\n"
    f"- {CALL_TOOL_TOKEN} — run Python in a ```python fenced block (e.g. quick tests)\n"
    f"- {TERMINATE_TOKEN} — final answer: output the **complete function body** that fits "
    "the given signature/docstring (only the indented body lines, or a full ```python block).\n"
    "The grader concatenates your completion after the task prompt."
)


def build_initial_messages(question: str, *, system_prompt: str | None = None) -> list[dict]:
    sys_p = system_prompt or SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": question},
    ]


def detect_action(response: str, *, disable_tools: bool) -> str:
    if TERMINATE_TOKEN in response:
        return "terminate"
    if (not disable_tools) and CALL_TOOL_TOKEN in response:
        return "call_tool"
    if SAMPLE_ALT_TOKEN in response:
        return "sample_alt"
    if VERIFY_TOKEN in response:
        return "verify"
    if CONTINUE_TOKEN in response:
        return "continue"
    return "continue"


@torch.inference_mode()
def adaptive_rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_steps: int = 5,
    max_tokens_per_step: int = 256,
    temperature: float = 1.0,
    disable_tools: bool = False,
    tool_timeout_sec: float = 5.0,
    system_prompt: str | None = None,
) -> dict:
    """Single rollout: shared by GRPO training and evaluation.

    Returns:
        text: joined assistant outputs (for reward / #### parsing)
        total_tokens, n_tool_calls, action_counts (dict str->int)
        messages_history, generated_ids, prompt_lengths (for GRPO log-prob replay)
        terminated: bool
    """
    model.eval()
    messages = build_initial_messages(question, system_prompt=system_prompt)
    all_generated_ids: list[list[int]] = []
    all_prompt_lengths: list[int] = []
    final_text_parts: list[str] = []
    total_tokens = 0
    n_tool_calls = 0
    action_counts: dict[str, int] = {
        "continue": 0,
        "verify": 0,
        "sample_alt": 0,
        "call_tool": 0,
        "terminate": 0,
    }
    terminated = False

    for _step in range(max_steps):
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

        new_ids = outputs[0][input_len:].tolist()
        if not new_ids:
            break
        total_tokens += len(new_ids)
        all_generated_ids.append(new_ids)
        all_prompt_lengths.append(input_len)

        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        final_text_parts.append(response)

        action = detect_action(response, disable_tools=disable_tools)
        if disable_tools and action == "call_tool":
            action = "continue"

        action_counts[action] = action_counts.get(action, 0) + 1

        if action == "terminate":
            messages.append({"role": "assistant", "content": response})
            terminated = True
            break

        if action == "call_tool":
            messages.append({"role": "assistant", "content": response})
            if disable_tools:
                messages.append({"role": "user", "content": DISABLED_TOOL_USER_PROMPT})
                continue
            code = extract_python_from_response(response)
            if code:
                n_tool_calls += 1
                ok, out = run_python_sandboxed(code, timeout_sec=tool_timeout_sec)
                tool_msg = f"Python tool output:\n{out}" if ok else f"Python tool error:\n{out}"
            else:
                tool_msg = TOOL_FAILURE_PROMPT
            messages.append({"role": "user", "content": tool_msg})
            continue

        if action == "sample_alt":
            messages = build_initial_messages(question, system_prompt=system_prompt)
            continue

        if action == "verify":
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": VERIFY_USER_PROMPT})
            continue

        # continue (default)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": DECISION_PROMPT})

    full_text = " ".join(final_text_parts)
    return {
        "text": full_text,
        "total_tokens": total_tokens,
        "n_tool_calls": n_tool_calls,
        "action_counts": action_counts,
        "messages_history": messages,
        "generated_ids": all_generated_ids,
        "prompt_lengths": all_prompt_lengths,
        "terminated": terminated,
    }


@torch.inference_mode()
def adaptive_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_steps: int = 5,
    max_tokens_per_step: int = 256,
    temperature: float = 0.7,
    return_full_trace: bool = False,
    disable_tools: bool = False,
    system_prompt: str | None = None,
) -> dict:
    """Eval-friendly wrapper around adaptive_rollout (slightly lower temperature default)."""
    out = adaptive_rollout(
        model,
        tokenizer,
        question,
        max_steps=max_steps,
        max_tokens_per_step=max_tokens_per_step,
        temperature=temperature,
        disable_tools=disable_tools,
        system_prompt=system_prompt,
    )
    answer_text = out["text"]
    if TERMINATE_TOKEN in answer_text:
        after = answer_text.split(TERMINATE_TOKEN, 1)[-1].strip()
        if after:
            answer_text = after
    result = {
        "answer_text": answer_text,
        "total_tokens": out["total_tokens"],
        "num_steps": len(out["generated_ids"]),
        "terminated": out["terminated"],
        "n_tool_calls": out["n_tool_calls"],
        "action_counts": dict(out["action_counts"]),
    }
    if return_full_trace:
        result["trace"] = out["text"]
    return result


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
        {
            "role": "system",
            "content": (
                "You are a math problem solver. Solve step by step. "
                "End your solution with #### <number> on its own line, "
                "where <number> is your final numerical answer."
            ),
        },
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


@torch.inference_mode()
def direct_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_tokens: int = 64,
    temperature: float = 0.3,
) -> dict:
    """Direct answer baseline: minimal reasoning."""
    messages = [
        {
            "role": "system",
            "content": (
                "You solve grade-school math. Reply with only the final number. "
                "End with #### <number> on its own line."
            ),
        },
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
        do_sample=temperature > 0,
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
