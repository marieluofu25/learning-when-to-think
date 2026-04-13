"""Generation infrastructure for the 3-action MDP.

Provides:
- adaptive_generate(): Multi-step generation with continue/refine/terminate actions
- cot_baseline(): Single-pass chain-of-thought (no action tokens)
- direct_baseline(): Minimal reasoning, direct answer
- generate_k_rollouts(): K-rollout sampling for a single prompt (used by GRPO and eval)

All functions return RolloutResult objects compatible with the eval harness.
"""

from __future__ import annotations

import re
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.data.math import extract_boxed_answer, grade_math_answer
from src.pivot.eval_types import Action, RolloutResult
from src.pivot.tokens import ACTION_TOKENS, get_action_token_ids


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ADAPTIVE_SYSTEM_PROMPT = (
    "You are a math problem solver. Solve the problem step by step.\n"
    "After each reasoning step, output exactly one action token:\n"
    "- <continue> to keep reasoning\n"
    "- <refine> if you spot an error in your recent reasoning and want to correct it\n"
    "- <terminate> when you are ready to give the final answer\n\n"
    "After <terminate>, immediately write your final answer in \\boxed{}."
)

COT_SYSTEM_PROMPT = (
    "You are a math problem solver. Solve the problem step by step, "
    "showing your work clearly. Write your final answer in \\boxed{}."
)

DIRECT_SYSTEM_PROMPT = (
    "You are a math problem solver. Give the answer directly with minimal "
    "reasoning. Write your final answer in \\boxed{}."
)


# ---------------------------------------------------------------------------
# Adaptive generation (3-action policy)
# ---------------------------------------------------------------------------

def adaptive_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_steps: int = 10,
    max_tokens_per_step: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> dict:
    """Generate a reasoning trace using the 3-action policy.

    The model generates reasoning in steps. After each step it emits one of
    <continue>, <refine>, or <terminate>. Generation stops when <terminate>
    is emitted or max_steps is reached.

    Returns a dict with:
        text: Full generated text
        actions: List of actions taken
        step_token_counts: Tokens per step
        total_tokens: Total generated tokens
        generated_ids: List of token ID lists per step (for GRPO log-prob replay)
        messages: The chat messages used
    """
    action_ids = get_action_token_ids(tokenizer)
    terminate_id = action_ids["<terminate>"]
    action_id_set = set(action_ids.values())

    messages = [
        {"role": "system", "content": ADAPTIVE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    full_text = ""
    actions: list[Action] = []
    step_token_counts: list[int] = []
    all_generated_ids: list[list[int]] = []
    total_tokens = 0

    for step in range(max_steps):
        prompt = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": full_text}],
            tokenize=False,
            add_generation_prompt=False if full_text else True,
        )
        # If we already have partial assistant text, strip the trailing end token
        if full_text:
            # Remove trailing <|im_end|> or similar that chat template adds
            for end_tok in ["<|im_end|>", "</s>", "<|endoftext|>"]:
                if prompt.endswith(end_tok):
                    prompt = prompt[: -len(end_tok)]
                    break

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_tokens_per_step,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            output = model.generate(input_ids, **gen_kwargs)

        new_ids = output[0, input_ids.shape[1] :].tolist()
        all_generated_ids.append(new_ids)
        step_token_counts.append(len(new_ids))
        total_tokens += len(new_ids)

        step_text = tokenizer.decode(new_ids, skip_special_tokens=False)
        full_text += step_text

        # Detect which action token was emitted (if any)
        detected_action = _detect_action(step_text, action_ids)
        if detected_action:
            actions.append(detected_action)
        else:
            # No explicit action token — treat as implicit continue
            actions.append("continue")

        if detected_action == "terminate":
            break

        # Check if model emitted EOS
        if new_ids and new_ids[-1] == tokenizer.eos_token_id:
            break

    return {
        "text": full_text,
        "actions": actions,
        "step_token_counts": step_token_counts,
        "total_tokens": total_tokens,
        "generated_ids": all_generated_ids,
        "messages": messages,
    }


def _detect_action(text: str, action_ids: dict[str, int]) -> Action | None:
    """Detect which action token appears in the generated text."""
    # Check for action tokens in order of priority
    if "<terminate>" in text:
        return "terminate"
    if "<refine>" in text:
        return "refine"
    if "<continue>" in text:
        return "continue"
    return None


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def cot_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> dict:
    """Single-pass chain-of-thought generation (no action tokens).

    Returns dict with: text, total_tokens, generated_ids, messages
    """
    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        output = model.generate(input_ids, **gen_kwargs)

    new_ids = output[0, input_ids.shape[1] :].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=False)

    return {
        "text": text,
        "total_tokens": len(new_ids),
        "generated_ids": [new_ids],
        "messages": messages,
    }


def direct_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> dict:
    """Direct-answer baseline — minimal reasoning."""
    messages = [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        output = model.generate(input_ids, **gen_kwargs)

    new_ids = output[0, input_ids.shape[1] :].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=False)

    return {
        "text": text,
        "total_tokens": len(new_ids),
        "generated_ids": [new_ids],
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# K-rollout sampling (shared by GRPO training and evaluation)
# ---------------------------------------------------------------------------

def generate_k_rollouts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problem: dict,
    k: int = 8,
    method: Literal["adaptive", "cot", "direct"] = "adaptive",
    **gen_kwargs,
) -> list[RolloutResult]:
    """Generate K rollouts for a single problem.

    Args:
        model: The language model (base or LoRA-adapted).
        tokenizer: Tokenizer (must have action tokens if method="adaptive").
        problem: Dict with keys: question, answer, level, subject, unique_id.
        k: Number of rollouts to sample.
        method: Generation method.
        **gen_kwargs: Passed to the generation function (temperature, max_tokens, etc.).

    Returns:
        List of K RolloutResult objects.
    """
    question = problem["question"]
    gold = problem["answer"]
    level = problem.get("level", 0)
    subject = problem.get("subject", "unknown")
    prompt_id = problem.get("unique_id", "")

    # Parse level from string like "Level 3" if needed
    if isinstance(level, str):
        m = re.match(r"Level\s+(\d+)", level)
        level = int(m.group(1)) if m else 0

    # Select generation function
    if method == "adaptive":
        gen_fn = adaptive_generate
        default_kwargs = {"temperature": 0.7, "max_steps": 10, "max_tokens_per_step": 256}
    elif method == "cot":
        gen_fn = cot_baseline
        default_kwargs = {"temperature": 0.7, "max_tokens": 2048}
    elif method == "direct":
        gen_fn = direct_baseline
        default_kwargs = {"temperature": 0.0, "max_tokens": 128}
    else:
        raise ValueError(f"Unknown method: {method}")

    # Merge defaults with overrides
    merged_kwargs = {**default_kwargs, **gen_kwargs}

    results = []
    for _ in range(k):
        output = gen_fn(model, tokenizer, question, **merged_kwargs)

        predicted = extract_boxed_answer(output["text"])
        correct = grade_math_answer(predicted, gold)

        result = RolloutResult(
            prompt_id=str(prompt_id),
            question=question,
            gold_answer=gold,
            level=level,
            subject=subject,
            generated_text=output["text"],
            predicted_answer=predicted,
            correct=correct,
            num_tokens=output["total_tokens"],
            actions=output.get("actions", []),
            step_token_counts=output.get("step_token_counts", []),
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Self-consistency (majority vote over K CoT rollouts)
# ---------------------------------------------------------------------------

def self_consistency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    problem: dict,
    k: int = 5,
    **gen_kwargs,
) -> RolloutResult:
    """Self-consistency baseline: majority vote over K CoT samples.

    Returns a single RolloutResult with the majority-voted answer and
    total tokens across all K samples.
    """
    rollouts = generate_k_rollouts(
        model, tokenizer, problem, k=k, method="cot",
        temperature=0.7, **gen_kwargs,
    )

    # Majority vote
    from collections import Counter
    answers = [r.predicted_answer for r in rollouts if r.predicted_answer is not None]
    total_tokens = sum(r.num_tokens for r in rollouts)

    if answers:
        # Normalize for counting
        from src.data.math import normalize_math_answer
        normalized = [normalize_math_answer(a) for a in answers]
        most_common = Counter(normalized).most_common(1)[0][0]
        # Find the original (un-normalized) answer that matches
        voted_answer = next(
            a for a, n in zip(answers, normalized) if n == most_common
        )
    else:
        voted_answer = None

    gold = problem["answer"]
    correct = grade_math_answer(voted_answer, gold)

    level = problem.get("level", 0)
    if isinstance(level, str):
        m = re.match(r"Level\s+(\d+)", level)
        level = int(m.group(1)) if m else 0

    return RolloutResult(
        prompt_id=str(problem.get("unique_id", "")),
        question=problem["question"],
        gold_answer=gold,
        level=level,
        subject=problem.get("subject", "unknown"),
        generated_text=f"[Self-consistency over {k} samples]",
        predicted_answer=voted_answer,
        correct=correct,
        num_tokens=total_tokens,
        actions=[],
        step_token_counts=[],
    )
