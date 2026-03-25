"""Run a small but real GRPO training session suitable for MacBook Air M4 24GB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.gsm8k import load_gsm8k
from src.train.grpo import setup_lora, train_grpo


def main():
    output_dir = Path("checkpoints/grpo/run2")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-0.5B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32, device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Setting up LoRA (rank=16)...")
    model = setup_lora(model, rank=16, alpha=32)

    train_data = load_gsm8k("train", subset_size=20)
    print(f"Training on {len(train_data)} problems")

    step_log: list[dict] = []

    def log_fn(epoch, step, metrics):
        entry = {"epoch": epoch, "step": step, **metrics}
        step_log.append(entry)
        print(f"  [E{epoch} S{step}] loss={metrics['loss']:.4f} "
              f"reward={metrics['avg_reward']:.4f} acc={metrics['accuracy']:.3f}")

    history = train_grpo(
        model, tokenizer, train_data,
        num_epochs=2,
        batch_size=1,
        num_rollouts=4,
        max_steps=3,
        max_tokens_per_step=128,
        lambda_cost=0.1,
        learning_rate=1e-4,
        save_path=str(output_dir / "final"),
        log_callback=log_fn,
    )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "step_log.json").write_text(json.dumps(step_log, indent=2))
    print(f"\nDone. Checkpoint at {output_dir / 'final'}")


if __name__ == "__main__":
    main()
