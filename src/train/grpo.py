"""GRPO (Group Relative Policy Optimization) training with LoRA.

Simplified implementation:
- Generate N rollouts per problem using the current policy (no grad)
- Score each rollout: reward = correctness - lambda * (tokens / max_tokens)
- Recompute log probs with gradients, compute GRPO advantages, update LoRA weights
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, TaskType

from src.data.gsm8k import extract_predicted_number, grade_answer
from src.policy.adaptive import (
    SYSTEM_PROMPT,
    DECISION_PROMPT,
    TERMINATE_TOKEN,
    build_initial_messages,
)


def setup_lora(model: PreTrainedModel, rank: int = 16, alpha: int = 32) -> PreTrainedModel:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


@torch.inference_mode()
def generate_rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    max_steps: int = 5,
    max_tokens_per_step: int = 256,
    temperature: float = 1.0,
) -> dict:
    """Generate a single rollout, collecting the full token sequence (no grad)."""
    model.eval()
    messages = build_initial_messages(question)
    all_generated_ids: list[list[int]] = []
    all_prompt_lengths: list[int] = []
    total_tokens = 0
    final_text_parts: list[str] = []

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

        new_ids = outputs[0][input_len:].tolist()
        if not new_ids:
            break
        total_tokens += len(new_ids)
        all_generated_ids.append(new_ids)
        all_prompt_lengths.append(input_len)

        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        final_text_parts.append(response)

        if TERMINATE_TOKEN in response:
            break

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": DECISION_PROMPT})

    full_text = " ".join(final_text_parts)
    return {
        "text": full_text,
        "total_tokens": total_tokens,
        "messages_history": messages,
        "generated_ids": all_generated_ids,
        "prompt_lengths": all_prompt_lengths,
    }


def recompute_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    generated_ids: list[list[int]],
    prompt_lengths: list[int],
) -> torch.Tensor:
    """Recompute log probs for the generated tokens WITH gradients.

    We reconstruct the full sequence (prompt + generated) for the final step
    and compute log P(generated | prompt) to keep things tractable.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_ids_list: list[int] = []
    for step_ids in generated_ids:
        full_ids_list.extend(step_ids)

    if not full_ids_list:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    gen_ids = torch.tensor([full_ids_list], device=model.device)
    input_ids = torch.cat([prompt_ids.to(model.device), gen_ids], dim=1)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    prompt_len = prompt_ids.shape[1]
    shift_logits = logits[:, prompt_len - 1 : -1, :]
    shift_labels = gen_ids

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum()


def compute_reward(
    text: str,
    gold_answer: float,
    total_tokens: int,
    max_tokens: int,
    lambda_cost: float = 0.1,
) -> float:
    predicted = extract_predicted_number(text)

    if predicted is None:
        correctness = 0.0
    elif grade_answer(predicted, gold_answer):
        correctness = 1.0
    else:
        # Partial credit: log-ratio closeness (capped)
        ratio = abs(predicted - gold_answer) / max(abs(gold_answer), 1e-6)
        if ratio < 0.1:
            correctness = 0.5
        elif ratio < 0.5:
            correctness = 0.2
        else:
            correctness = 0.05  # at least parsed a number

    cost_penalty = lambda_cost * (total_tokens / max(max_tokens, 1))
    return correctness - cost_penalty


def grpo_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[dict],
    optimizer: torch.optim.Optimizer,
    num_rollouts: int = 4,
    max_steps: int = 5,
    max_tokens_per_step: int = 256,
    lambda_cost: float = 0.1,
) -> dict:
    """One GRPO update step over a batch of questions."""
    max_total_tokens = max_steps * max_tokens_per_step
    total_loss = 0.0
    total_reward = 0.0
    total_correct = 0
    total_items = 0

    for item in questions:
        rollouts = []
        for _ in range(num_rollouts):
            rollout = generate_rollout(
                model, tokenizer, item["question"],
                max_steps=max_steps,
                max_tokens_per_step=max_tokens_per_step,
            )
            reward = compute_reward(
                rollout["text"], item["answer_number"],
                rollout["total_tokens"], max_total_tokens, lambda_cost,
            )
            rollout["reward"] = reward
            rollouts.append(rollout)

        rewards = torch.tensor([r["reward"] for r in rollouts])
        total_reward += rewards.sum().item()

        for r in rollouts:
            predicted = extract_predicted_number(r["text"])
            if grade_answer(predicted, item["answer_number"]):
                total_correct += 1
        total_items += num_rollouts

        if rewards.std() > 1e-8:
            advantages = (rewards - rewards.mean()) / rewards.std()
        else:
            advantages = torch.zeros_like(rewards)

        model.train()
        for rollout, advantage in zip(rollouts, advantages):
            if advantage.abs() < 1e-8:
                continue
            if not rollout["generated_ids"]:
                continue
            log_prob_sum = recompute_log_probs(
                model, tokenizer,
                rollout["messages_history"],
                rollout["generated_ids"],
                rollout["prompt_lengths"],
            )
            loss = -advantage * log_prob_sum
            loss.backward()
            total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    return {
        "loss": total_loss / max(total_items, 1),
        "avg_reward": total_reward / max(total_items, 1),
        "accuracy": total_correct / max(total_items, 1),
    }


def train_grpo(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_data: list[dict],
    num_epochs: int = 3,
    batch_size: int = 2,
    num_rollouts: int = 4,
    max_steps: int = 5,
    max_tokens_per_step: int = 256,
    lambda_cost: float = 0.1,
    learning_rate: float = 1e-4,
    save_path: str | None = None,
    log_callback=None,
) -> list[dict]:
    """Full GRPO training loop."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    history: list[dict] = []

    for epoch in range(num_epochs):
        epoch_metrics = {"loss": 0, "avg_reward": 0, "accuracy": 0, "steps": 0}

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            step_metrics = grpo_step(
                model, tokenizer, batch, optimizer,
                num_rollouts=num_rollouts,
                max_steps=max_steps,
                max_tokens_per_step=max_tokens_per_step,
                lambda_cost=lambda_cost,
            )

            epoch_metrics["loss"] += step_metrics["loss"]
            epoch_metrics["avg_reward"] += step_metrics["avg_reward"]
            epoch_metrics["accuracy"] += step_metrics["accuracy"]
            epoch_metrics["steps"] += 1

            if log_callback:
                log_callback(epoch, i // batch_size, step_metrics)

        n_steps = max(epoch_metrics["steps"], 1)
        epoch_summary = {
            "epoch": epoch,
            "avg_loss": epoch_metrics["loss"] / n_steps,
            "avg_reward": epoch_metrics["avg_reward"] / n_steps,
            "avg_accuracy": epoch_metrics["accuracy"] / n_steps,
        }
        history.append(epoch_summary)
        print(f"Epoch {epoch}: loss={epoch_summary['avg_loss']:.4f} "
              f"reward={epoch_summary['avg_reward']:.4f} "
              f"accuracy={epoch_summary['avg_accuracy']:.4f}")

    if save_path:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    return history
