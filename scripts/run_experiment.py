"""Run evaluation experiments on MATH-500.

Usage:
    python -m scripts.run_experiment --config configs/experiment.yaml
    python -m scripts.run_experiment --method cot --checkpoint checkpoints/sft/final
    python -m scripts.run_experiment --method adaptive --checkpoint checkpoints/grpo/final

Supports: adaptive (3-action), CoT single-pass, direct-answer, self-consistency.
All methods output RolloutResult objects evaluated by Hao's eval harness.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.data.math import load_math500, load_math_train
from src.pivot.eval_harness import evaluate, print_eval_report, metrics_to_dict
from src.pivot.eval_types import RolloutResult
from src.pivot.generate import (
    adaptive_generate,
    cot_baseline,
    direct_baseline,
    generate_k_rollouts,
    self_consistency,
)
from src.pivot.tokens import setup_tokenizer_and_model, ACTION_TOKENS


def load_config(config_path: str | None, cli_args: argparse.Namespace) -> dict:
    """Load YAML config and override with CLI args."""
    defaults = {
        "base_model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "checkpoint": None,
        "eval_dataset": "math500",
        "eval_subset": None,
        "method": "adaptive",
        "k_rollouts": 1,
        "temperature": 0.7,
        "max_steps": 10,
        "max_tokens_per_step": 256,
        "max_tokens": 2048,
        "sc_k": 5,
        "output_dir": "results",
        "experiment_name": "default",
        "save_rollouts": True,
    }

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        defaults.update(file_cfg)

    # CLI overrides (only non-None values)
    for key in ["base_model", "checkpoint", "method", "eval_subset",
                 "temperature", "max_steps", "experiment_name", "output_dir"]:
        val = getattr(cli_args, key, None)
        if val is not None:
            defaults[key] = val

    return defaults


def load_model_and_tokenizer(
    base_model: str,
    checkpoint: str | None = None,
    device: str = "auto",
) -> tuple:
    """Load base model + optional LoRA checkpoint.

    If checkpoint is provided, also checks if action tokens need to be added.
    """
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    # Check if action tokens already exist
    has_action_tokens = all(
        tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id
        for t in ACTION_TOKENS
    )

    if checkpoint:
        if not has_action_tokens:
            # Add action tokens before loading LoRA (LoRA was trained with them)
            print("Adding action tokens to tokenizer/model...")
            setup_tokenizer_and_model(tokenizer, model)

        print(f"Loading LoRA checkpoint: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        print("LoRA merged.")

    model.eval()
    return model, tokenizer


def run_single_method(
    model, tokenizer, problems: list[dict], cfg: dict,
) -> list[RolloutResult]:
    """Run a single evaluation method over all problems."""
    method = cfg["method"]
    results: list[RolloutResult] = []

    print(f"\nRunning method: {method} on {len(problems)} problems")
    start = time.time()

    for i, problem in enumerate(problems):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(problems)}] elapsed={elapsed:.0f}s")

        if method == "self_consistency":
            result = self_consistency(
                model, tokenizer, problem,
                k=cfg["sc_k"],
                max_tokens=cfg["max_tokens"],
            )
            results.append(result)
        else:
            rollouts = generate_k_rollouts(
                model, tokenizer, problem,
                k=cfg["k_rollouts"],
                method=method,
                temperature=cfg["temperature"],
                max_steps=cfg.get("max_steps", 10),
                max_tokens_per_step=cfg.get("max_tokens_per_step", 256),
                max_tokens=cfg.get("max_tokens", 2048),
            )
            # For eval, take the first rollout (or best if k>1)
            if cfg["k_rollouts"] == 1:
                results.append(rollouts[0])
            else:
                # Take best (correct + fewest tokens)
                correct = [r for r in rollouts if r.correct]
                if correct:
                    results.append(min(correct, key=lambda r: r.num_tokens))
                else:
                    results.append(rollouts[0])

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({elapsed/len(problems):.2f}s/problem)")
    return results


def save_results(
    results: list[RolloutResult],
    metrics: dict,
    cfg: dict,
) -> str:
    """Save metrics and optionally individual rollouts."""
    out_dir = Path(cfg["output_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    method = cfg["method"]
    ckpt_name = Path(cfg["checkpoint"]).stem if cfg["checkpoint"] else "base"

    # Save metrics
    metrics_path = out_dir / f"metrics_{method}_{ckpt_name}.json"
    with open(metrics_path, "w") as f:
        json.dump({"config": cfg, "metrics": metrics}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save individual rollouts
    if cfg.get("save_rollouts"):
        rollouts_path = out_dir / f"rollouts_{method}_{ckpt_name}.jsonl"
        with open(rollouts_path, "w") as f:
            for r in results:
                entry = {
                    "prompt_id": r.prompt_id,
                    "question": r.question,
                    "gold_answer": r.gold_answer,
                    "level": r.level,
                    "subject": r.subject,
                    "predicted_answer": r.predicted_answer,
                    "correct": r.correct,
                    "num_tokens": r.num_tokens,
                    "actions": r.actions,
                    "step_token_counts": r.step_token_counts,
                    "generated_text": r.generated_text,
                }
                f.write(json.dumps(entry) + "\n")
        print(f"Rollouts saved to {rollouts_path}")

    return str(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Run MATH-500 evaluation experiment")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--method", choices=["adaptive", "cot", "direct", "self_consistency"])
    parser.add_argument("--eval_subset", type=int, default=None, help="Subset size for quick testing")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args)
    print(f"Config: {json.dumps(cfg, indent=2)}")

    # Load dataset
    if cfg["eval_dataset"] == "math500":
        problems = load_math500()
    else:
        problems = load_math_train(subset_size=cfg.get("eval_subset"))

    if cfg["eval_subset"]:
        import random
        random.seed(42)
        problems = random.sample(problems, min(cfg["eval_subset"], len(problems)))

    print(f"Loaded {len(problems)} problems")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        cfg["base_model"], cfg["checkpoint"]
    )

    # Run experiment
    results = run_single_method(model, tokenizer, problems, cfg)

    # Evaluate
    metrics = evaluate(results)
    print_eval_report(metrics)

    # Save
    metrics_dict = metrics_to_dict(metrics)
    save_results(results, metrics_dict, cfg)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
