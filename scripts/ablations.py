"""Run ablations over lambda_cost and max_steps (GRPO + LoRA)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k import load_gsm8k
from src.eval.evaluate import evaluate_results, run_adaptive_policy, save_results
from src.train.grpo import setup_lora, train_grpo
from src.train.model_loading import load_base_causal_lm, load_tokenizer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/ablations")
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[1e-6, 5e-6, 1e-5, 5e-5],
        help="lambda_cost values (token penalty scale; paired with proposal-style reward)",
    )
    parser.add_argument("--max-steps-list", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller subsets for faster iteration",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Ablation: disable CALL_TOOL during training rollouts",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)

    if args.quick:
        cfg["train_subset_size"] = 50
        cfg["eval_subset_size"] = 30
        cfg["num_epochs"] = 1

    use_qlora = bool(cfg.get("use_qlora", False))
    torch_dtype_name = str(cfg.get("torch_dtype", "float16"))

    print(f"Loading model: {cfg['model_name']} (use_qlora={use_qlora})")
    tokenizer = load_tokenizer(cfg["model_name"])

    train_data = load_gsm8k("train", subset_size=cfg["train_subset_size"])
    eval_data = load_gsm8k("test", subset_size=cfg["eval_subset_size"])

    all_results = {}
    mu_tool = float(cfg.get("mu_tool", 0.05))
    disable_tools = bool(cfg.get("disable_tools", False)) or args.no_tools

    print("\n=== Lambda ablation (max_steps fixed) ===")
    for lam in args.lambdas:
        print(f"\n--- lambda_cost={lam} ---")
        model = load_base_causal_lm(
            cfg["model_name"],
            use_qlora=use_qlora,
            torch_dtype_name=torch_dtype_name,
        )
        model = setup_lora(model, rank=cfg.get("lora_rank", 16), alpha=cfg.get("lora_alpha", 32))

        train_grpo(
            model,
            tokenizer,
            train_data,
            num_epochs=cfg.get("num_epochs", 3),
            batch_size=cfg.get("batch_size", 2),
            num_rollouts=cfg.get("num_rollouts_per_problem", 4),
            max_steps=cfg.get("max_steps", 5),
            max_tokens_per_step=cfg.get("max_tokens_per_step", 256),
            lambda_cost=lam,
            mu_tool=mu_tool,
            learning_rate=float(cfg.get("learning_rate", 1e-4)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
            disable_tools=disable_tools,
        )

        results = run_adaptive_policy(
            model,
            tokenizer,
            eval_data,
            max_steps=cfg.get("max_steps", 5),
            max_tokens_per_step=cfg.get("max_tokens_per_step", 256),
            disable_tools=disable_tools,
        )
        metrics = evaluate_results(results, dataset_kind="gsm8k")
        key = f"lambda_{lam}"
        all_results[key] = metrics
        save_results(results, metrics, output_dir / f"{key}.json")
        print(
            f"lambda_cost={lam}: acc={metrics['accuracy']:.3f} "
            f"tokens={metrics['avg_tokens_per_problem']:.1f} "
            f"tools={metrics.get('total_tool_calls', 0)}"
        )

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n=== Max-steps ablation (lambda_cost fixed) ===")
    fixed_lambda = float(cfg.get("lambda_cost", 1e-5))
    for ms in args.max_steps_list:
        print(f"\n--- max_steps={ms} ---")
        model = load_base_causal_lm(
            cfg["model_name"],
            use_qlora=use_qlora,
            torch_dtype_name=torch_dtype_name,
        )
        model = setup_lora(model, rank=cfg.get("lora_rank", 16), alpha=cfg.get("lora_alpha", 32))

        train_grpo(
            model,
            tokenizer,
            train_data,
            num_epochs=cfg.get("num_epochs", 3),
            batch_size=cfg.get("batch_size", 2),
            num_rollouts=cfg.get("num_rollouts_per_problem", 4),
            max_steps=ms,
            max_tokens_per_step=cfg.get("max_tokens_per_step", 256),
            lambda_cost=fixed_lambda,
            mu_tool=mu_tool,
            learning_rate=float(cfg.get("learning_rate", 1e-4)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
            disable_tools=disable_tools,
        )

        results = run_adaptive_policy(
            model,
            tokenizer,
            eval_data,
            max_steps=ms,
            max_tokens_per_step=cfg.get("max_tokens_per_step", 256),
            disable_tools=disable_tools,
        )
        metrics = evaluate_results(results, dataset_kind="gsm8k")
        key = f"max_steps_{ms}"
        all_results[key] = metrics
        save_results(results, metrics, output_dir / f"{key}.json")
        print(
            f"max_steps={ms}: acc={metrics['accuracy']:.3f} "
            f"tokens={metrics['avg_tokens_per_problem']:.1f}"
        )

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    summary_path = output_dir / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2))
    print("\n=== Ablation summary ===")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
