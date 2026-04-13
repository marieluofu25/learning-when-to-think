"""Run ablation studies for the Learning When to Think project.

Usage:
    python -m scripts.run_ablations --config configs/ablations.yaml
    python -m scripts.run_ablations --checkpoint checkpoints/grpo/final --quick

Ablation dimensions:
  - beta_sweep: ALP length penalty coefficient
  - allow_refine: Toggle refine action
  - degrpo: Toggle DeGRPO gradient weighting

Each ablation requires a separately trained checkpoint. This script evaluates
pre-trained checkpoints and collects results for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from src.data.math import load_math500
from src.pivot.eval_harness import evaluate, print_eval_report, metrics_to_dict
from scripts.run_experiment import load_model_and_tokenizer, run_single_method, save_results


def load_ablation_config(config_path: str | None) -> dict:
    """Load ablation config."""
    defaults = {
        "base_model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "eval_dataset": "math500",
        "eval_subset": None,
        "temperature": 0.7,
        "max_steps": 10,
        "max_tokens_per_step": 256,
        "k_rollouts": 1,
        "output_dir": "results/ablations",
    }
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        defaults.update(file_cfg)
    return defaults


def run_checkpoint_ablation(
    checkpoint_dir: str,
    base_model: str,
    problems: list[dict],
    cfg: dict,
    label: str,
) -> dict:
    """Evaluate a single checkpoint and return metrics."""
    print(f"\n{'='*60}")
    print(f"  Ablation: {label}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(base_model, checkpoint_dir)

    eval_cfg = {
        **cfg,
        "method": "adaptive",
        "checkpoint": checkpoint_dir,
        "experiment_name": label,
    }

    results = run_single_method(model, tokenizer, problems, eval_cfg)
    metrics = evaluate(results)
    print_eval_report(metrics)

    metrics_dict = metrics_to_dict(metrics)
    save_results(results, metrics_dict, eval_cfg)

    # Free GPU memory
    del model
    import torch
    torch.cuda.empty_cache()

    return {"label": label, "checkpoint": checkpoint_dir, "metrics": metrics_dict}


def run_beta_sweep(
    checkpoints: dict[float, str],
    base_model: str,
    problems: list[dict],
    cfg: dict,
) -> list[dict]:
    """Evaluate checkpoints trained with different beta values."""
    all_results = []
    for beta, ckpt in sorted(checkpoints.items()):
        result = run_checkpoint_ablation(
            ckpt, base_model, problems, cfg, label=f"beta={beta}"
        )
        result["beta"] = beta
        all_results.append(result)
    return all_results


def run_toggle_ablation(
    checkpoints: dict[str, str],
    base_model: str,
    problems: list[dict],
    cfg: dict,
    ablation_name: str,
) -> list[dict]:
    """Evaluate checkpoints with a toggled feature (refine, degrpo)."""
    all_results = []
    for setting, ckpt in checkpoints.items():
        result = run_checkpoint_ablation(
            ckpt, base_model, problems, cfg, label=f"{ablation_name}={setting}"
        )
        result[ablation_name] = setting
        all_results.append(result)
    return all_results


def compare_baselines(
    base_model: str,
    problems: list[dict],
    cfg: dict,
    checkpoint: str | None = None,
) -> list[dict]:
    """Run all baseline methods for comparison.

    This can run without any trained checkpoints — it evaluates the base model
    with different generation strategies.
    """
    model, tokenizer = load_model_and_tokenizer(base_model, checkpoint)
    all_results = []

    for method in ["cot", "direct"]:
        print(f"\n{'='*60}")
        print(f"  Baseline: {method}")
        print(f"{'='*60}")

        eval_cfg = {
            **cfg,
            "method": method,
            "checkpoint": checkpoint,
            "experiment_name": f"baseline_{method}",
            "max_tokens": 2048 if method == "cot" else 128,
        }

        results = run_single_method(model, tokenizer, problems, eval_cfg)
        metrics = evaluate(results)
        print_eval_report(metrics)

        metrics_dict = metrics_to_dict(metrics)
        save_results(results, metrics_dict, eval_cfg)
        all_results.append({"label": f"baseline_{method}", "metrics": metrics_dict})

    # Self-consistency (separate because it needs k samples)
    print(f"\n{'='*60}")
    print(f"  Baseline: self_consistency (k=5)")
    print(f"{'='*60}")

    eval_cfg = {
        **cfg,
        "method": "self_consistency",
        "checkpoint": checkpoint,
        "experiment_name": "baseline_self_consistency",
        "sc_k": 5,
        "max_tokens": 2048,
    }

    from src.pivot.generate import self_consistency as sc_fn
    sc_results = []
    for i, problem in enumerate(problems):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(problems)}]")
        sc_results.append(sc_fn(model, tokenizer, problem, k=5, max_tokens=2048))

    metrics = evaluate(sc_results)
    print_eval_report(metrics)
    metrics_dict = metrics_to_dict(metrics)
    all_results.append({"label": "baseline_self_consistency_k5", "metrics": metrics_dict})

    # Free GPU memory
    del model
    import torch
    torch.cuda.empty_cache()

    return all_results


def save_ablation_summary(all_results: list[dict], output_dir: str):
    """Save combined ablation results summary."""
    out_path = Path(output_dir) / "ablation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAblation summary saved to {out_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  {'Label':<35} {'Accuracy':>10} {'Avg Tokens':>12} {'Cost/Correct':>14}")
    print(f"  {'-'*73}")
    for r in all_results:
        m = r["metrics"]
        print(
            f"  {r['label']:<35} {m['accuracy']:>10.4f} "
            f"{m['avg_tokens']:>12.1f} {m['cost_per_correct']:>14.1f}"
        )
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", type=str, default="configs/ablations.yaml")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Only run baselines (no trained checkpoints needed)")
    parser.add_argument("--quick", action="store_true",
                        help="Use subset of 50 problems for quick testing")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint to evaluate")
    parser.add_argument("--beta-checkpoints", type=str, default=None,
                        help="JSON mapping beta->checkpoint_path for beta sweep")
    parser.add_argument("--refine-checkpoints", type=str, default=None,
                        help="JSON mapping true/false->checkpoint_path for refine ablation")
    parser.add_argument("--degrpo-checkpoints", type=str, default=None,
                        help="JSON mapping true/false->checkpoint_path for degrpo ablation")
    args = parser.parse_args()

    cfg = load_ablation_config(args.config)

    if args.quick:
        cfg["eval_subset"] = 50

    # Load dataset
    problems = load_math500()
    if cfg.get("eval_subset"):
        import random
        random.seed(42)
        problems = random.sample(problems, min(cfg["eval_subset"], len(problems)))
    print(f"Loaded {len(problems)} problems")

    all_results = []

    # Always run baselines
    baseline_results = compare_baselines(
        cfg["base_model"], problems, cfg, checkpoint=args.checkpoint
    )
    all_results.extend(baseline_results)

    if args.baselines_only:
        save_ablation_summary(all_results, cfg["output_dir"])
        return

    # Beta sweep
    if args.beta_checkpoints:
        beta_map = json.loads(args.beta_checkpoints)
        beta_results = run_beta_sweep(
            {float(k): v for k, v in beta_map.items()},
            cfg["base_model"], problems, cfg,
        )
        all_results.extend(beta_results)

    # Refine ablation
    if args.refine_checkpoints:
        refine_map = json.loads(args.refine_checkpoints)
        refine_results = run_toggle_ablation(
            refine_map, cfg["base_model"], problems, cfg, "allow_refine"
        )
        all_results.extend(refine_results)

    # DeGRPO ablation
    if args.degrpo_checkpoints:
        degrpo_map = json.loads(args.degrpo_checkpoints)
        degrpo_results = run_toggle_ablation(
            degrpo_map, cfg["base_model"], problems, cfg, "degrpo"
        )
        all_results.extend(degrpo_results)

    # Single checkpoint evaluation
    if args.checkpoint and not args.baselines_only:
        result = run_checkpoint_ablation(
            args.checkpoint, cfg["base_model"], problems, cfg, label="trained_adaptive"
        )
        all_results.append(result)

    save_ablation_summary(all_results, cfg["output_dir"])
    print("\nAll ablations complete.")


if __name__ == "__main__":
    main()
