"""Run CoT and self-consistency baselines on MATH-500."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.math_500 import load_math_500
from src.eval.evaluate import (
    evaluate_results,
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
    parser.add_argument("--output-dir", default="results/baselines")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)

    print(f"Loading model: {cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        dtype=torch.float16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading MATH-500 (test, subset={cfg['eval_subset_size']})")
    dataset = load_math_500("test", subset_size=cfg["eval_subset_size"])

    # --- CoT baseline ---
    print("\n=== Running CoT Baseline ===")
    cot_results = run_cot_baseline(model, tokenizer, dataset)
    cot_metrics = evaluate_results(cot_results, dataset_kind="math_500")
    print(f"CoT: accuracy={cot_metrics['accuracy']:.3f}, "
          f"avg_tokens={cot_metrics['avg_tokens_per_problem']:.1f}")
    save_results(cot_results, cot_metrics, output_dir / "cot_results.json")

    # --- Self-consistency baseline ---
    k = cfg.get("self_consistency_k", 5)
    print(f"\n=== Running Self-Consistency (k={k}) ===")
    sc_results = run_self_consistency(model, tokenizer, dataset, k=k)
    sc_metrics = evaluate_results(sc_results, dataset_kind="math_500")
    print(f"Self-Consistency: accuracy={sc_metrics['accuracy']:.3f}, "
          f"avg_tokens={sc_metrics['avg_tokens_per_problem']:.1f}")
    save_results(sc_results, sc_metrics, output_dir / "self_consistency_results.json")

    # --- Summary ---
    summary = {"cot": cot_metrics, "self_consistency": sc_metrics}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
