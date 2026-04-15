"""MATH-500 ablations: with-refine vs no-refine (GRPO + LoRA)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.math_500 import load_math_500
from src.eval.evaluate import evaluate_results, run_adaptive_policy, save_results
from src.train.grpo import setup_lora, train_grpo
from src.train.hf_hub_config import apply_hf_hub_config
from src.train.model_loading import load_base_causal_lm, load_tokenizer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/ablations")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller subsets for faster iteration",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_hf_hub_config(cfg)
    output_dir = Path(args.output_dir)

    if args.quick:
        cfg["train_subset_size"] = 50
        cfg["eval_subset_size"] = 30
        cfg["num_epochs"] = 1

    use_qlora = bool(cfg.get("use_qlora", False))
    torch_dtype_name = str(cfg.get("torch_dtype", "float16"))

    print(f"Loading model: {cfg['model_name']} (use_qlora={use_qlora})")
    tokenizer = load_tokenizer(cfg["model_name"])

    train_data = load_math_500("test", subset_size=cfg["train_subset_size"])
    eval_data = load_math_500("test", subset_size=cfg["eval_subset_size"])

    common_train_kwargs = dict(
        num_epochs=int(cfg.get("num_epochs", 3)),
        batch_size=int(cfg.get("batch_size", 2)),
        num_rollouts=int(cfg.get("num_rollouts_per_problem", 4)),
        max_steps=int(cfg.get("max_steps", 5)),
        max_tokens_per_step=int(cfg.get("max_tokens_per_step", 256)),
        beta=float(cfg.get("beta", 0.02)),
        L_max=cfg.get("L_max"),
        degrpo=bool(cfg.get("degrpo", True)),
        w_ctrl=float(cfg.get("w_ctrl", 2.0)),
        w_resp=float(cfg.get("w_resp", 1.0)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
    )

    common_eval_kwargs = dict(
        max_steps=int(cfg.get("max_steps", 5)),
        max_tokens_per_step=int(cfg.get("max_tokens_per_step", 256)),
    )

    def train_and_evaluate(*, allow_refine: bool, out_path: Path, label: str) -> None:
        print(f"\n=== Condition: {label} (allow_refine={allow_refine}) ===")

        # Train a fresh model for each condition.
        model = load_base_causal_lm(
            cfg["model_name"],
            use_qlora=use_qlora,
            torch_dtype_name=torch_dtype_name,
        )
        model = setup_lora(
            model,
            rank=int(cfg.get("lora_rank", 16)),
            alpha=int(cfg.get("lora_alpha", 32)),
        )

        train_grpo(
            model,
            tokenizer,
            train_data,
            **common_train_kwargs,
            allow_refine=allow_refine,
        )

        results = run_adaptive_policy(
            model,
            tokenizer,
            eval_data,
            allow_refine=allow_refine,
            **common_eval_kwargs,
        )
        metrics = evaluate_results(results, dataset_kind="math_500")
        save_results(results, metrics, out_path)

        print(
            f"{label}: acc={metrics['accuracy']:.3f} "
            f"tokens={metrics['avg_tokens_per_problem']:.1f} "
            f"format_ok={metrics.get('format_success_rate', 0):.2f}"
        )

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_and_evaluate(
        allow_refine=True,
        out_path=output_dir / "ablation_with_refine.json",
        label="with-refine",
    )
    train_and_evaluate(
        allow_refine=False,
        out_path=output_dir / "ablation_no_refine.json",
        label="no-refine",
    )


if __name__ == "__main__":
    main()
