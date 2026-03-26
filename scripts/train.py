"""Train adaptive policy with GRPO + LoRA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k import load_gsm8k
from src.train.grpo import setup_lora, train_grpo


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kl_coef = cfg.get("kl_coef", 0.0)

    print(f"Loading model: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        dtype=torch.float32,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model = None
    if kl_coef > 0:
        print(f"Loading frozen reference model (kl_coef={kl_coef})...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            dtype=torch.float32,
            device_map="auto",
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    print("Setting up LoRA...")
    model = setup_lora(
        model,
        rank=cfg.get("lora_rank", 16),
        alpha=cfg.get("lora_alpha", 32),
    )

    print(f"Loading GSM8K (train, subset={cfg['train_subset_size']})")
    train_data = load_gsm8k("train", subset_size=cfg["train_subset_size"])

    step_log: list[dict] = []

    def log_callback(epoch, step, metrics):
        entry = {"epoch": epoch, "step": step, **metrics}
        step_log.append(entry)
        print(f"  [E{epoch} S{step}] loss={metrics['loss']:.4f} "
              f"reward={metrics['avg_reward']:.4f} acc={metrics['accuracy']:.3f}")

    print("\n=== Starting GRPO Training ===")
    if kl_coef > 0:
        print(f"  KL regularization enabled (coef={kl_coef})")
    history = train_grpo(
        model, tokenizer, train_data,
        num_epochs=cfg.get("num_epochs", 3),
        batch_size=cfg.get("batch_size", 2),
        num_rollouts=cfg.get("num_rollouts_per_problem", 4),
        max_steps=cfg.get("max_steps", 5),
        max_tokens_per_step=cfg.get("max_tokens_per_step", 256),
        lambda_cost=cfg.get("lambda_cost", 0.1),
        learning_rate=cfg.get("learning_rate", 1e-4),
        save_path=str(output_dir / "final"),
        log_callback=log_callback,
        ref_model=ref_model,
        kl_coef=kl_coef,
    )

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2)
    )
    (output_dir / "step_log.json").write_text(
        json.dumps(step_log, indent=2)
    )
    print(f"\nTraining complete. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
