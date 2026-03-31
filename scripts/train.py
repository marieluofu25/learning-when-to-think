"""Train adaptive policy with GRPO + LoRA (optional 4-bit QLoRA)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.math_500 import load_math_500
from peft import prepare_model_for_kbit_training

from src.train.grpo import setup_lora, train_grpo
from src.train.model_loading import load_base_causal_lm, load_tokenizer


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

    kl_coef = float(cfg.get("kl_coef", 0.0))
    use_qlora = bool(cfg.get("use_qlora", False))
    torch_dtype_name = str(cfg.get("torch_dtype", "bfloat16"))

    print(f"Loading model: {cfg['model_name']} (use_qlora={use_qlora})")
    tokenizer = load_tokenizer(cfg["model_name"])
    model = load_base_causal_lm(
        cfg["model_name"],
        use_qlora=use_qlora,
        torch_dtype_name=torch_dtype_name,
    )
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    ref_model = None
    if kl_coef > 0:
        print(f"Loading frozen reference model (kl_coef={kl_coef})...")
        ref_model = load_base_causal_lm(
            cfg["model_name"],
            use_qlora=use_qlora,
            torch_dtype_name=torch_dtype_name,
        )
        if use_qlora:
            ref_model = prepare_model_for_kbit_training(ref_model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    print("Setting up LoRA...")
    model = setup_lora(
        model,
        rank=cfg.get("lora_rank", 16),
        alpha=cfg.get("lora_alpha", 32),
    )

    dataset_name = str(cfg.get("dataset", "math_500")).lower()
    if dataset_name not in {"math_500", "math-500", "math500"}:
        raise ValueError(
            f"MATH-500-only training expected dataset=math_500, got dataset={dataset_name!r}"
        )

    print(f"Loading MATH-500 (train slice, subset={cfg['train_subset_size']})")
    train_data = load_math_500("test", subset_size=cfg["train_subset_size"])

    step_log: list[dict] = []

    def log_callback(epoch, step, metrics):
        entry = {"epoch": epoch, "step": step, **metrics}
        step_log.append(entry)
        print(
            f"  [E{epoch} S{step}] loss={metrics['loss']:.4f} "
            f"reward={metrics['avg_reward']:.4f} acc={metrics['accuracy']:.3f}"
        )

    print("\n=== Starting GRPO Training ===")
    if kl_coef > 0:
        print(f"  KL regularization enabled (coef={kl_coef})")
    allow_refine = bool(cfg.get("allow_refine", cfg.get("allow_verify", True)))
    l_max = cfg.get("L_max")
    history = train_grpo(
        model,
        tokenizer,
        train_data,
        num_epochs=cfg.get("num_epochs", 3),
        batch_size=cfg.get("batch_size", 2),
        num_rollouts=cfg.get("num_rollouts_per_problem", 4),
        max_steps=cfg.get("max_steps", 5),
        max_tokens_per_step=cfg.get("max_tokens_per_step", 256),
        beta=float(cfg.get("beta", 0.02)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        save_path=str(output_dir / "final"),
        log_callback=log_callback,
        ref_model=ref_model,
        kl_coef=kl_coef,
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        allow_refine=allow_refine,
        L_max=float(l_max) if l_max is not None else None,
        format_bonus=float(cfg.get("format_bonus", 0.0)),
        n_control_tokens=int(cfg.get("n_control_tokens", 16)),
        w_ctrl=float(cfg.get("w_ctrl", 2.0)),
        w_resp=float(cfg.get("w_resp", 1.0)),
        degrpo=bool(cfg.get("degrpo", True)),
    )

    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "step_log.json").write_text(json.dumps(step_log, indent=2))
    print(f"\nTraining complete. Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
