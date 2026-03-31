"""GRPO (Group Relative Policy Optimization) training with LoRA + KL regularization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, TaskType

from src.data.math_500 import (
    extract_hash_answer,
    extract_predicted_answer,
    grade_answer,
    has_valid_format,
)
from src.policy.adaptive import adaptive_rollout


def setup_lora(model: PreTrainedModel, rank: int = 16, alpha: int = 32) -> PreTrainedModel:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        # Include both attention and MLP projection layers for stronger adaptation.
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def recompute_log_probs_weighted(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    generated_ids: list[list[int]],
    n_control_tokens: int = 16,
    prompt_lengths: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute log probs for the generated tokens WITH gradients.

    DeGRPO-style split: for each step, the first `n_control_tokens` are treated as
    control tokens, and the remainder are response tokens.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_ids_list: list[int] = []
    for step_ids in generated_ids:
        full_ids_list.extend(step_ids)

    if not full_ids_list:
        z = torch.tensor(0.0, device=model.device, requires_grad=True)
        return z, z, z

    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(model.device)
    gen_ids = torch.tensor([full_ids_list], device=model.device)
    input_ids = torch.cat([prompt_ids.to(model.device), gen_ids], dim=1)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    prompt_len = prompt_ids.shape[1]
    shift_logits = logits[:, prompt_len - 1 : -1, :]
    shift_labels = gen_ids

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    # token_log_probs shape: [1, total_generated_tokens]
    control_logp_sum = token_log_probs.new_zeros(())
    resp_logp_sum = token_log_probs.new_zeros(())
    total_logp_sum = token_log_probs.new_zeros(())

    n_ctrl = max(0, int(n_control_tokens))
    idx = 0
    for step_ids in generated_ids:
        step_len = len(step_ids)
        ctl_len = max(0, min(n_ctrl, step_len))
        control_logp_sum = control_logp_sum + token_log_probs[:, idx : idx + ctl_len].sum()
        resp_logp_sum = resp_logp_sum + token_log_probs[:, idx + ctl_len : idx + step_len].sum()
        total_logp_sum = total_logp_sum + token_log_probs[:, idx : idx + step_len].sum()
        idx += step_len

    return control_logp_sum, resp_logp_sum, total_logp_sum


# Back-compat alias (older internal name).
recompute_log_probs_split = recompute_log_probs_weighted


def compute_r_acc(
    text: str,
    gold_answer: str,
) -> float:
    """Verifiable accuracy reward component (r_acc in ALP)."""
    predicted = extract_hash_answer(text)
    if predicted is None:
        predicted = extract_predicted_answer(text)

    return 1.0 if grade_answer(predicted, gold_answer) else 0.0


def compute_r_fmt(text: str, format_bonus: float) -> float:
    """Optional format score for logging only (not used in ALP reward)."""
    return float(format_bonus) if has_valid_format(text) else 0.0


def compute_alp_reward(
    r_acc: float,
    n_tokens: int,
    *,
    group_solve_rate: float,
    beta: float,
    l_max: float,
) -> float:
    """ALP: r_acc - beta * max(0, SR) * (n_tokens / L_max)."""
    if l_max <= 0:
        return float(r_acc)
    sr = max(0.0, float(group_solve_rate))
    length_term = float(n_tokens) / float(l_max)
    return float(r_acc) - float(beta) * sr * length_term


@torch.inference_mode()
def compute_ref_total_log_probs(
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    generated_ids: list[list[int]],
) -> torch.Tensor:
    """Compute total log prob under the frozen reference model (no grad)."""
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
    beta: float = 0.02,
    allow_refine: bool = True,
    allow_verify: bool | None = None,
    ref_model: PreTrainedModel | None = None,
    kl_coef: float = 0.0,
    L_max: float | None = None,
    format_bonus: float = 0.0,
    n_control_tokens: int = 16,
    w_ctrl: float = 2.0,
    w_resp: float = 1.0,
    degrpo: bool = True,
    do_optimizer_step: bool = True,
    **_ignored: object,
) -> dict:
    """One GRPO update step over a batch of questions."""
    if allow_verify is not None:
        allow_refine = allow_verify

    l_max_eff = float(L_max) if L_max is not None else float(max_steps * max_tokens_per_step)
    if l_max_eff <= 0:
        l_max_eff = float(max_steps * max_tokens_per_step)

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
                allow_refine=allow_refine,
            )
            rollouts.append(rollout)

        for rollout in rollouts:
            r_acc = compute_r_acc(rollout["text"], item["answer_number"])
            r_fmt = compute_r_fmt(rollout["text"], format_bonus=format_bonus)
            rollout["r_acc"] = r_acc
            rollout["r_fmt"] = r_fmt

            if r_acc >= 0.5:
                total_correct += 1

        acc_tensor = torch.tensor([float(r["r_acc"]) for r in rollouts], dtype=torch.float32)
        group_solve_rate = float(acc_tensor.mean().item())

        for rollout in rollouts:
            rollout["reward"] = compute_alp_reward(
                float(rollout["r_acc"]),
                int(rollout["total_tokens"]),
                group_solve_rate=group_solve_rate,
                beta=beta,
                l_max=l_max_eff,
            )

        rewards = torch.tensor([r["reward"] for r in rollouts], device=model.device)
        total_reward += rewards.sum().item()
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

            control_logp_sum, resp_logp_sum, total_logp_sum = recompute_log_probs_weighted(
                model,
                tokenizer,
                rollout["messages_history"],
                rollout["generated_ids"],
                n_control_tokens=n_control_tokens,
                prompt_lengths=rollout.get("prompt_lengths"),
            )

            kl_term = torch.tensor(0.0, device=model.device)
            if ref_model is not None and kl_coef > 0:
                ref_log_prob = compute_ref_total_log_probs(
                    ref_model,
                    tokenizer,
                    rollout["messages_history"],
                    rollout["generated_ids"],
                )
                kl_term = total_logp_sum - ref_log_prob.to(model.device)
                total_kl += float(kl_term.detach().item())

            if degrpo:
                weighted_logp = (w_ctrl * control_logp_sum) + (w_resp * resp_logp_sum)
            else:
                weighted_logp = total_logp_sum
            loss = -advantage * weighted_logp + kl_coef * kl_term
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
    beta: float = 0.02,
    learning_rate: float = 1e-4,
    save_path: str | None = None,
    log_callback=None,
    ref_model: PreTrainedModel | None = None,
    kl_coef: float = 0.0,
    gradient_accumulation_steps: int = 1,
    allow_refine: bool = True,
    allow_verify: bool | None = None,
    L_max: float | None = None,
    format_bonus: float = 0.0,
    n_control_tokens: int = 16,
    w_ctrl: float = 2.0,
    w_resp: float = 1.0,
    degrpo: bool = True,
    **_ignored: object,
) -> list[dict]:
    """Full GRPO training loop with optional KL regularization and grad accumulation."""
    if allow_verify is not None:
        allow_refine = allow_verify

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
                beta=beta,
                allow_refine=allow_refine,
                allow_verify=allow_verify,
                ref_model=ref_model,
                kl_coef=kl_coef,
                L_max=L_max,
                format_bonus=format_bonus,
                n_control_tokens=n_control_tokens,
                w_ctrl=w_ctrl,
                w_resp=w_resp,
                degrpo=degrpo,
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
