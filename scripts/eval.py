"""Evaluate trained adaptive policy vs baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k import load_gsm8k
from src.eval.evaluate import (
    evaluate_results,
    run_adaptive_policy,
    run_cot_baseline,
    run_self_consistency,
    save_results,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/grpo/final")
    parser.add_argument("--output-dir", default="results/eval")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--pulse-every", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)

    print(f"Loading base model: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        dtype=torch.float16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_gsm8k("test", subset_size=cfg["eval_subset_size"])

    all_metrics = {}

    # --- Baselines ---
    if not args.skip_baselines:
        print("\n=== CoT Baseline ===")
        cot_results = run_cot_baseline(
            base_model,
            tokenizer,
            dataset,
            stage_name="CoT",
            pulse_every=args.pulse_every,
        )
        cot_metrics = evaluate_results(cot_results)
        all_metrics["cot"] = cot_metrics
        save_results(cot_results, cot_metrics, output_dir / "cot_results.json")
        print(f"CoT: accuracy={cot_metrics['accuracy']:.3f}, "
              f"avg_tokens={cot_metrics['avg_tokens_per_problem']:.1f}")

        k = cfg.get("self_consistency_k", 5)
        print(f"\n=== Self-Consistency (k={k}) ===")
        sc_results = run_self_consistency(
            base_model,
            tokenizer,
            dataset,
            k=k,
            stage_name=f"Self-Consistency(k={k})",
            pulse_every=args.pulse_every,
        )
        sc_metrics = evaluate_results(sc_results)
        all_metrics["self_consistency"] = sc_metrics
        save_results(sc_results, sc_metrics, output_dir / "sc_results.json")
        print(f"SC: accuracy={sc_metrics['accuracy']:.3f}, "
              f"avg_tokens={sc_metrics['avg_tokens_per_problem']:.1f}")

    # --- Adaptive policy ---
    print("\n=== Adaptive Policy (trained) ===")
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"Loading LoRA checkpoint from {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
    else:
        print("WARNING: No checkpoint found, using base model (untrained policy)")
        model = base_model

    adaptive_results = run_adaptive_policy(
        model, tokenizer, dataset,
        stage_name="Adaptive",
        pulse_every=args.pulse_every,
        max_steps=cfg.get("max_steps", 5),
        max_tokens_per_step=cfg.get("max_tokens_per_step", 256),
    )
    adaptive_metrics = evaluate_results(adaptive_results)
    all_metrics["adaptive"] = adaptive_metrics
    save_results(adaptive_results, adaptive_metrics, output_dir / "adaptive_results.json")
    print(f"Adaptive: accuracy={adaptive_metrics['accuracy']:.3f}, "
          f"avg_tokens={adaptive_metrics['avg_tokens_per_problem']:.1f}")

    # --- Comparison summary ---
    print("\n=== Final Comparison ===")
    print(json.dumps(all_metrics, indent=2))
    (output_dir / "comparison.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
