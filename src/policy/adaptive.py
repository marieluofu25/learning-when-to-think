"""Adaptive reasoning policy: three meta-actions (continue, refine, terminate)."""

from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedModel, PreTrainedTokenizerBase

CONTINUE_TOKEN = "<continue>"
REFINE_TOKEN = "<refine>"
TERMINATE_TOKEN = "<terminate>"

# Back-compat aliases (deprecated)
VERIFY_TOKEN = REFINE_TOKEN

SYSTEM_PROMPT = (
    "You are a math problem solver. Work step by step.\n"
    "Each assistant message must start by choosing exactly one action on its own line:\n"
    f"- {CONTINUE_TOKEN} — add more chain-of-thought\n"
    f"- {REFINE_TOKEN} — double-check and correct the last step\n"
    f"- {TERMINATE_TOKEN} — final answer: put the result in \\boxed{{...}} "
    "and/or end with #### <final answer> on its own line\n"
    "After the action line, write your reasoning."
)

SYSTEM_PROMPT_NO_REFINE = (
    "You are a math problem solver. Work step by step.\n"
    "Each assistant message must start by choosing exactly one action on its own line:\n"
    f"- {CONTINUE_TOKEN} — add more chain-of-thought\n"
    f"- {TERMINATE_TOKEN} — final answer: put the result in \\boxed{{...}} "
    "and/or end with #### <final answer> on its own line\n"
    "After the action line, write your reasoning."
)

DECISION_PROMPT = (
    "\nChoose the next action. Reply starting with one of: "
    f"{CONTINUE_TOKEN}, {REFINE_TOKEN}, or {TERMINATE_TOKEN}."
)

DECISION_PROMPT_NO_REFINE = (
    "\nChoose the next action. Reply starting with one of: "
    f"{CONTINUE_TOKEN} or {TERMINATE_TOKEN}."
)

REFINE_USER_PROMPT = (
    "Double-check your previous reasoning step. Fix any mistake, then continue "
    f"(still starting your reply with an action token: {CONTINUE_TOKEN}, {REFINE_TOKEN}, {TERMINATE_TOKEN})."
)


def build_initial_messages(question: str, *, system_prompt: str | None = None) -> list[dict]:
    sys_p = SYSTEM_PROMPT if system_prompt is None else system_prompt
    return [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": question},
    ]


def detect_action(response: str) -> str:
    if TERMINATE_TOKEN in response:
        return "terminate"
    # Legacy checkpoints / prompts used `<verify>` for the same meta-action.
    if REFINE_TOKEN in response or "<verify>" in response:
        return "refine"
    if CONTINUE_TOKEN in response:
        return "continue"
    return "continue"


def _control_token_len(
    tokenizer: PreTrainedTokenizerBase,
    step_ids: list[int],
    action_token: str,
) -> int:
    """
    Return number of *prefix* tokens whose decoded text (after leading whitespace
    stripping) exactly equals the given action token.
    """
    action_ids = tokenizer.encode(action_token, add_special_tokens=False)
    if len(step_ids) >= len(action_ids) and step_ids[: len(action_ids)] == action_ids:
        return len(action_ids)

    max_scan = min(len(step_ids), len(action_ids) + 4)
    for k in range(1, max_scan + 1):
        decoded = tokenizer.decode(step_ids[:k], skip_special_tokens=True)
        if decoded.lstrip() == action_token:
            return k

    return len(action_ids)


def _action_control_token_len(
    tokenizer: PreTrainedTokenizerBase,
    step_ids: list[int],
    action: str,
) -> int:
    """Control-prefix length for the chosen meta-action (handles legacy `<verify>`)."""
    if action != "refine":
        tok = {
            "continue": CONTINUE_TOKEN,
            "terminate": TERMINATE_TOKEN,
        }[action]
        return _control_token_len(tokenizer, step_ids, tok)
    for cand in (REFINE_TOKEN, "<verify>"):
        n = _control_token_len(tokenizer, step_ids, cand)
        dec = tokenizer.decode(step_ids[:n], skip_special_tokens=True).lstrip()
        if dec == cand:
            return n
    return _control_token_len(tokenizer, step_ids, REFINE_TOKEN)


def _first_action_token_id_candidates(
    tokenizer: PreTrainedTokenizerBase,
    *,
    allow_refine: bool,
) -> list[int]:
    """Union of first sub-token ids for each allowed action string (MVP hard mask)."""
    tokens: list[str] = [CONTINUE_TOKEN, TERMINATE_TOKEN]
    if allow_refine:
        tokens.append(REFINE_TOKEN)
    seen: dict[int, None] = {}
    for t in tokens:
        enc = tokenizer.encode(t, add_special_tokens=False)
        if enc:
            seen.setdefault(enc[0], None)
    return list(seen.keys())


class _FirstTokenRestrictLogitsProcessor(LogitsProcessor):
    """Force the first generated token to be one of ``allowed_ids``."""

    def __init__(self, allowed_ids: list[int]):
        if not allowed_ids:
            raise ValueError("allowed_ids must be non-empty")
        self.allowed_ids = list(allowed_ids)
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._step != 0:
            self._step += 1
            return scores
        self._step += 1
        mask = torch.full_like(scores, float("-inf"))
        for tid in self.allowed_ids:
            mask[:, tid] = scores[:, tid]
        return mask


@torch.inference_mode()
def adaptive_rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_steps: int = 5,
    max_tokens_per_step: int = 256,
    temperature: float = 1.0,
    allow_refine: bool = True,
    allow_verify: bool | None = None,
    system_prompt: str | None = None,
    constrain_action_first_token: bool = False,
) -> dict:
    """Single rollout: shared by GRPO training and evaluation.

    If ``allow_verify`` is set, it overrides ``allow_refine`` (deprecated alias).
    If ``constrain_action_first_token`` is True, the first token of each generation
    step is restricted to prefixes of the allowed action strings (forced interface).

    Returns:
        text: joined assistant outputs (for reward / #### parsing)
        total_tokens, action_counts (dict str->int)
        messages_history, generated_ids, prompt_lengths (for GRPO log-prob replay)
        control_token_lens: per-step lengths of the first tokens for action token
        terminated: bool
    """
    if allow_verify is not None:
        allow_refine = allow_verify

    model.eval()
    resolved_system_prompt = (
        system_prompt
        if system_prompt is not None
        else (SYSTEM_PROMPT if allow_refine else SYSTEM_PROMPT_NO_REFINE)
    )
    messages = build_initial_messages(question, system_prompt=resolved_system_prompt)
    all_generated_ids: list[list[int]] = []
    all_prompt_lengths: list[int] = []
    control_token_lens: list[int] = []
    final_text_parts: list[str] = []
    total_tokens = 0
    action_counts: dict[str, int] = {
        "continue": 0,
        "refine": 0,
        "terminate": 0,
    }
    terminated = False

    for _step in range(max_steps):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = dict(
            max_new_tokens=max_tokens_per_step,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if constrain_action_first_token:
            try:
                allowed = _first_action_token_id_candidates(
                    tokenizer, allow_refine=allow_refine
                )
            except Exception:
                allowed = []
            if allowed:
                gen_kwargs["logits_processor"] = LogitsProcessorList(
                    [_FirstTokenRestrictLogitsProcessor(allowed)]
                )

        outputs = model.generate(**inputs, **gen_kwargs)

        new_ids = outputs[0][input_len:].tolist()
        if not new_ids:
            break
        total_tokens += len(new_ids)
        all_generated_ids.append(new_ids)
        all_prompt_lengths.append(input_len)

        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        final_text_parts.append(response)

        detected_action = detect_action(response)
        control_token_lens.append(
            _action_control_token_len(tokenizer, new_ids, detected_action)
        )

        action = detected_action
        if not allow_refine and action == "refine":
            action = "continue"

        action_counts[action] = action_counts.get(action, 0) + 1

        if action == "terminate":
            messages.append({"role": "assistant", "content": response})
            terminated = True
            break

        if action == "refine":
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": REFINE_USER_PROMPT})
            continue

        messages.append({"role": "assistant", "content": response})
        next_decision_prompt = DECISION_PROMPT if allow_refine else DECISION_PROMPT_NO_REFINE
        messages.append({"role": "user", "content": next_decision_prompt})

    full_text = " ".join(final_text_parts)
    return {
        "text": full_text,
        "total_tokens": total_tokens,
        "action_counts": action_counts,
        "messages_history": messages,
        "generated_ids": all_generated_ids,
        "prompt_lengths": all_prompt_lengths,
        "control_token_lens": control_token_lens,
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
    allow_refine: bool = True,
    allow_verify: bool | None = None,
    system_prompt: str | None = None,
    constrain_action_first_token: bool = False,
) -> dict:
    """Eval-friendly wrapper around adaptive_rollout (slightly lower temperature default)."""
    if allow_verify is not None:
        allow_refine = allow_verify
    out = adaptive_rollout(
        model,
        tokenizer,
        question,
        max_steps=max_steps,
        max_tokens_per_step=max_tokens_per_step,
        temperature=temperature,
        allow_refine=allow_refine,
        system_prompt=system_prompt,
        constrain_action_first_token=constrain_action_first_token,
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
                "Put the final answer in \\boxed{...} and/or end with #### <final answer> on its own line."
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
                "You solve grade-school math. Reply with only the final answer. "
                "Use \\boxed{...} and/or end with #### <final answer> on its own line."
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
