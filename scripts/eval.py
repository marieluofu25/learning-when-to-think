"""Evaluate trained adaptive policy vs baselines (MATH-500 only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from peft import PeftModel, prepare_model_for_kbit_training

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.math_500 import load_math_500
from src.eval.evaluate import (
    evaluate_results,
    run_adaptive_policy,
    run_cot_baseline,
    run_direct_baseline,
    save_results,
)
from src.train.hf_hub_config import apply_hf_hub_config
from src.train.model_loading import load_base_causal_lm, load_tokenizer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_trained_model(base_model, checkpoint_path: str):
    """Load trained adaptive policy LoRA checkpoint on top of base model."""
    cp = Path(checkpoint_path)
    if cp.exists() and cp.is_dir():
        print(f"Loading LoRA checkpoint from {cp}", flush=True)
        return PeftModel.from_pretrained(base_model, str(cp))

    print("WARNING: No checkpoint found, using base model (untrained policy)", flush=True)
    return base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Override config checkpoint_path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--pulse-every", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_hf_hub_config(cfg)
    output_dir = Path(args.output_dir or cfg.get("eval_output_dir", "results/eval"))
    checkpoint = args.checkpoint or cfg.get("checkpoint_path", "checkpoints/grpo/final")

    # MATH-500 only (no HumanEval/GSM8K branches).
    use_qlora = bool(cfg.get("use_qlora", False))
    torch_dtype_name = str(cfg.get("torch_dtype", "float16"))

    print(f"Loading base model: {cfg['model_name']}", flush=True)
    tokenizer = load_tokenizer(cfg["model_name"])
    base_model = load_base_causal_lm(
        cfg["model_name"],
        use_qlora=use_qlora,
        torch_dtype_name=torch_dtype_name,
    )
    if use_qlora:
        base_model = prepare_model_for_kbit_training(base_model)

    gen_kw = {
        "max_steps": cfg.get("max_steps", 5),
        "max_tokens_per_step": cfg.get("max_tokens_per_step", 256),
        "constrain_action_first_token": bool(
            cfg.get("constrain_action_first_token", False)
        ),
    }

    all_metrics: dict = {}

    dataset = load_math_500("test", subset_size=cfg.get("eval_subset_size", 100))
    kind = "math_500"
    print(f"Dataset: MATH-500 test subset (n={len(dataset)})", flush=True)

    if not args.skip_baselines:
        print("\n=== CoT Baseline (base model) ===", flush=True)
        cot_results = run_cot_baseline(
            base_model,
            tokenizer,
            dataset,
            stage_name="CoT",
            pulse_every=args.pulse_every,
            partial_path=output_dir / "cot_results.partial.jsonl",
        )
        cot_metrics = evaluate_results(cot_results, dataset_kind=kind)
        all_metrics["cot_baseline"] = cot_metrics
        save_results(cot_results, cot_metrics, output_dir / "cot_results.json")
        print(
            f"CoT: accuracy={cot_metrics['accuracy']:.3f}, "
            f"avg_tokens={cot_metrics['avg_tokens_per_problem']:.1f}",
            flush=True,
        )

        print("\n=== Direct answer baseline (base model) ===", flush=True)
        direct_results = run_direct_baseline(
            base_model,
            tokenizer,
            dataset,
            stage_name="Direct",
            pulse_every=args.pulse_every,
            max_tokens=cfg.get("direct_max_tokens", 64),
            partial_path=output_dir / "direct_results.partial.jsonl",
        )
        direct_metrics = evaluate_results(direct_results, dataset_kind=kind)
        all_metrics["direct_baseline"] = direct_metrics
        save_results(direct_results, direct_metrics, output_dir / "direct_results.json")
        print(
            f"Direct: accuracy={direct_metrics['accuracy']:.3f}, "
            f"avg_tokens={direct_metrics['avg_tokens_per_problem']:.1f}",
            flush=True,
        )

    print("\n=== Adaptive policy (GRPO LoRA) ===", flush=True)
    trained_model = load_trained_model(base_model, checkpoint)

    adaptive_runs = [
        (True, "adaptive_with_refine_results.json", "adaptive_grpo_with_refine"),
        (False, "adaptive_no_refine_results.json", "adaptive_grpo_no_refine"),
    ]
    for allow_refine, out_name, metric_key in adaptive_runs:
        stage_name = "Adaptive-GRPO-with-refine" if allow_refine else "Adaptive-GRPO-no-refine"
        partial_name = out_name.replace(".json", ".partial.jsonl")
        adaptive_results = run_adaptive_policy(
            trained_model,
            tokenizer,
            dataset,
            stage_name=stage_name,
            pulse_every=args.pulse_every,
            allow_refine=allow_refine,
            partial_path=output_dir / partial_name,
            **gen_kw,
        )
        adaptive_metrics = evaluate_results(adaptive_results, dataset_kind=kind)
        all_metrics[metric_key] = adaptive_metrics
        save_results(adaptive_results, adaptive_metrics, output_dir / out_name)
        print(
            f"{stage_name}: accuracy={adaptive_metrics['accuracy']:.3f}, "
            f"avg_tokens={adaptive_metrics['avg_tokens_per_problem']:.1f}",
            flush=True,
        )

    print("\n=== Final Comparison ===", flush=True)
    print(json.dumps(all_metrics, indent=2), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"\nAll results saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
