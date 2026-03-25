"""Run ablations over lambda and max_steps hyperparameters."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k import load_gsm8k
from src.eval.evaluate import evaluate_results, run_adaptive_policy, save_results
from src.train.grpo import setup_lora, train_grpo


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/ablations")
    parser.add_argument(
        "--lambdas", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2]
    )
    parser.add_argument("--max-steps-list", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument(
        "--quick", action="store_true",
        help="Use smaller subsets for faster iteration",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)

    if args.quick:
        cfg["train_subset_size"] = 50
        cfg["eval_subset_size"] = 30
        cfg["num_epochs"] = 1

    print(f"Loading model: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = load_gsm8k("train", subset_size=cfg["train_subset_size"])
    eval_data = load_gsm8k("test", subset_size=cfg["eval_subset_size"])

    all_results = {}

    # --- Lambda ablation (fix max_steps) ---
    print("\n=== Lambda Ablation ===")
    for lam in args.lambdas:
        print(f"\n--- lambda={lam} ---")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], dtype=torch.float32, device_map="auto",
        )
        model = setup_lora(model, rank=cfg.get("lora_rank", 16))

        train_grpo(
            model, tokenizer, train_data,
            num_epochs=cfg.get("num_epochs", 3),
            batch_size=cfg.get("batch_size", 2),
            num_rollouts=cfg.get("num_rollouts_per_problem", 4),
            max_steps=cfg.get("max_steps", 5),
            max_tokens_per_step=cfg.get("max_tokens_per_step", 256),
            lambda_cost=lam,
            learning_rate=cfg.get("learning_rate", 1e-4),
        )

        results = run_adaptive_policy(
            model, tokenizer, eval_data,
            max_steps=cfg.get("max_steps", 5),
        )
        metrics = evaluate_results(results)
        key = f"lambda_{lam}"
        all_results[key] = metrics
        save_results(results, metrics, output_dir / f"{key}.json")
        print(f"lambda={lam}: acc={metrics['accuracy']:.3f} "
              f"tokens={metrics['avg_tokens_per_problem']:.1f}")

        del model
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # --- Max-steps ablation (fix lambda) ---
    print("\n=== Max Steps Ablation ===")
    fixed_lambda = cfg.get("lambda_cost", 0.1)
    for ms in args.max_steps_list:
        print(f"\n--- max_steps={ms} ---")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], dtype=torch.float32, device_map="auto",
        )
        model = setup_lora(model, rank=cfg.get("lora_rank", 16))

        train_grpo(
            model, tokenizer, train_data,
            num_epochs=cfg.get("num_epochs", 3),
            batch_size=cfg.get("batch_size", 2),
            num_rollouts=cfg.get("num_rollouts_per_problem", 4),
            max_steps=ms,
            max_tokens_per_step=cfg.get("max_tokens_per_step", 256),
            lambda_cost=fixed_lambda,
            learning_rate=cfg.get("learning_rate", 1e-4),
        )

        results = run_adaptive_policy(model, tokenizer, eval_data, max_steps=ms)
        metrics = evaluate_results(results)
        key = f"max_steps_{ms}"
        all_results[key] = metrics
        save_results(results, metrics, output_dir / f"{key}.json")
        print(f"max_steps={ms}: acc={metrics['accuracy']:.3f} "
              f"tokens={metrics['avg_tokens_per_problem']:.1f}")

        del model
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # --- Summary ---
    summary_path = output_dir / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n=== Ablation Summary ===")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
