"""GRPO (Group Relative Policy Optimization) training with LoRA + KL regularization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, TaskType

from src.data.gsm8k import extract_hash_answer, extract_predicted_number, grade_answer, has_valid_format
from src.policy.adaptive import adaptive_rollout


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


def recompute_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    generated_ids: list[list[int]],
    prompt_lengths: list[int],
) -> torch.Tensor:
    """Recompute log probs for the generated tokens WITH gradients."""
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
    n_tool_calls: int,
    lambda_cost: float = 1e-5,
    mu_tool: float = 0.05,
    format_bonus: float = 0.05,
) -> float:
    """R ≈ r_correct + format_bonus − λ·n_tokens − μ·n_tool_calls (proposal-style)."""
    predicted = extract_hash_answer(text)
    if predicted is None:
        predicted = extract_predicted_number(text)

    r_acc = 1.0 if grade_answer(predicted, gold_answer) else 0.0
    r_fmt = format_bonus if has_valid_format(text) else 0.0
    return r_acc + r_fmt - lambda_cost * float(total_tokens) - mu_tool * float(n_tool_calls)


@torch.inference_mode()
def compute_ref_log_probs(
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    generated_ids: list[list[int]],
) -> torch.Tensor:
    """Compute log probs under the frozen reference model (no grad)."""
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_ids_list: list[int] = []
    for step_ids in generated_ids:
        full_ids_list.extend(step_ids)

    if not full_ids_list:
        return torch.tensor(0.0, device=ref_model.device)

    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    gen_ids = torch.tensor([full_ids_list], device=ref_model.device)
    input_ids = torch.cat([prompt_ids.to(ref_model.device), gen_ids], dim=1)

    outputs = ref_model(input_ids=input_ids)
    logits = outputs.logits
    prompt_len = prompt_ids.shape[1]
    shift_logits = logits[:, prompt_len - 1 : -1, :]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum()


def grpo_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[dict],
    optimizer: torch.optim.Optimizer,
    num_rollouts: int = 4,
    max_steps: int = 5,
    max_tokens_per_step: int = 256,
    lambda_cost: float = 1e-5,
    mu_tool: float = 0.05,
    ref_model: PreTrainedModel | None = None,
    kl_coef: float = 0.0,
    disable_tools: bool = False,
    do_optimizer_step: bool = True,
) -> dict:
    """One GRPO update step over a batch of questions."""
    total_loss = 0.0
    total_reward = 0.0
    total_correct = 0
    total_items = 0
    total_kl = 0.0
    grad_steps = 0

    for item in questions:
        rollouts = []
        for _ in range(num_rollouts):
            rollout = adaptive_rollout(
                model,
                tokenizer,
                item["question"],
                max_steps=max_steps,
                max_tokens_per_step=max_tokens_per_step,
                temperature=1.0,
                disable_tools=disable_tools,
            )
            reward = compute_reward(
                rollout["text"],
                item["answer_number"],
                rollout["total_tokens"],
                rollout["n_tool_calls"],
                lambda_cost=lambda_cost,
                mu_tool=mu_tool,
            )
            rollout["reward"] = reward
            rollouts.append(rollout)

        rewards = torch.tensor([r["reward"] for r in rollouts])
        total_reward += rewards.sum().item()

        for r in rollouts:
            predicted = extract_hash_answer(r["text"]) or extract_predicted_number(r["text"])
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
                model,
                tokenizer,
                rollout["messages_history"],
                rollout["generated_ids"],
                rollout["prompt_lengths"],
            )

            kl_term = torch.tensor(0.0, device=model.device)
            if ref_model is not None and kl_coef > 0:
                ref_log_prob = compute_ref_log_probs(
                    ref_model,
                    tokenizer,
                    rollout["messages_history"],
                    rollout["generated_ids"],
                )
                kl_term = log_prob_sum.detach() - ref_log_prob
                total_kl += kl_term.item()

            loss = -advantage * log_prob_sum + kl_coef * kl_term
            loss.backward()
            total_loss += loss.item()
            grad_steps += 1

    if do_optimizer_step and grad_steps > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return {
        "loss": total_loss / max(grad_steps, 1),
        "avg_reward": total_reward / max(total_items, 1),
        "accuracy": total_correct / max(total_items, 1),
        "kl_mean": total_kl / max(total_items, 1),
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
    lambda_cost: float = 1e-5,
    mu_tool: float = 0.05,
    learning_rate: float = 1e-4,
    save_path: str | None = None,
    log_callback=None,
    ref_model: PreTrainedModel | None = None,
    kl_coef: float = 0.0,
    gradient_accumulation_steps: int = 1,
    disable_tools: bool = False,
) -> list[dict]:
    """Full GRPO training loop with optional KL regularization and grad accumulation."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    history: list[dict] = []
    accum = max(1, int(gradient_accumulation_steps))

    for epoch in range(num_epochs):
        epoch_metrics = {"loss": 0, "avg_reward": 0, "accuracy": 0, "kl_mean": 0, "steps": 0}

        micro = 0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            micro += 1
            do_step = micro % accum == 0
            step_metrics = grpo_step(
                model,
                tokenizer,
                batch,
                optimizer,
                num_rollouts=num_rollouts,
                max_steps=max_steps,
                max_tokens_per_step=max_tokens_per_step,
                lambda_cost=lambda_cost,
                mu_tool=mu_tool,
                ref_model=ref_model,
                kl_coef=kl_coef,
                disable_tools=disable_tools,
                do_optimizer_step=do_step,
            )

            epoch_metrics["loss"] += step_metrics["loss"]
            epoch_metrics["avg_reward"] += step_metrics["avg_reward"]
            epoch_metrics["accuracy"] += step_metrics["accuracy"]
            epoch_metrics["kl_mean"] += step_metrics["kl_mean"]
            epoch_metrics["steps"] += 1

            if log_callback:
                log_callback(epoch, i // batch_size, step_metrics)

        if micro > 0 and micro % accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        n_steps = max(epoch_metrics["steps"], 1)
        epoch_summary = {
            "epoch": epoch,
            "avg_loss": epoch_metrics["loss"] / n_steps,
            "avg_reward": epoch_metrics["avg_reward"] / n_steps,
            "avg_accuracy": epoch_metrics["accuracy"] / n_steps,
            "avg_kl": epoch_metrics["kl_mean"] / n_steps,
        }
        history.append(epoch_summary)
        print(
            f"Epoch {epoch}: loss={epoch_summary['avg_loss']:.4f} "
            f"reward={epoch_summary['avg_reward']:.4f} "
            f"accuracy={epoch_summary['avg_accuracy']:.4f} "
            f"kl={epoch_summary['avg_kl']:.4f}"
        )

    if save_path:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    return history
