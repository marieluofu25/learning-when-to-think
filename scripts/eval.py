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


def load_trained_model(base_model, checkpoint_path: str, sft_checkpoint: str | None = None):
    """Load trained model, handling SFT+DPO stacking."""
    cp = Path(checkpoint_path)
    sft_cp = Path(sft_checkpoint) if sft_checkpoint else None

    if sft_cp and sft_cp.exists():
        print(f"Loading SFT LoRA from {sft_cp} and merging...", flush=True)
        sft_model = PeftModel.from_pretrained(base_model, str(sft_cp))
        merged = sft_model.merge_and_unload()
        if cp.exists():
            print(f"Loading DPO LoRA from {cp} on top...", flush=True)
            return PeftModel.from_pretrained(merged, str(cp))
        return merged

    if cp.exists():
        print(f"Loading LoRA checkpoint from {cp}", flush=True)
        return PeftModel.from_pretrained(base_model, str(cp))

    print("WARNING: No checkpoint found, using base model (untrained policy)", flush=True)
    return base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/dpo/final")
    parser.add_argument("--sft-checkpoint", default="checkpoints/sft/final")
    parser.add_argument("--output-dir", default="results/eval_distill")
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

    if not args.skip_baselines:
        print("\n=== CoT Baseline (base model) ===")
        cot_results = run_cot_baseline(
            base_model,
            tokenizer,
            dataset,
            stage_name="CoT",
            pulse_every=args.pulse_every,
        )
        cot_metrics = evaluate_results(cot_results)
        all_metrics["cot_baseline"] = cot_metrics
        save_results(cot_results, cot_metrics, output_dir / "cot_results.json")
        print(f"CoT: accuracy={cot_metrics['accuracy']:.3f}, "
              f"avg_tokens={cot_metrics['avg_tokens_per_problem']:.1f}")

        k = cfg.get("self_consistency_k", 5)
        print(f"\n=== Self-Consistency (k={k}, base model) ===")
        sc_results = run_self_consistency(
            base_model,
            tokenizer,
            dataset,
            k=k,
            stage_name=f"SC(k={k})",
            pulse_every=args.pulse_every,
        )
        sc_metrics = evaluate_results(sc_results)
        all_metrics["self_consistency_baseline"] = sc_metrics
        save_results(sc_results, sc_metrics, output_dir / "sc_results.json")
        print(f"SC: accuracy={sc_metrics['accuracy']:.3f}, "
              f"avg_tokens={sc_metrics['avg_tokens_per_problem']:.1f}")

    # --- SFT-only evaluation ---
    sft_path = Path(args.sft_checkpoint)
    if sft_path.exists():
        print("\n=== CoT with SFT model ===")
        sft_model = PeftModel.from_pretrained(base_model, str(sft_path))
        sft_cot_results = run_cot_baseline(
            sft_model, tokenizer, dataset,
            stage_name="SFT-CoT",
            pulse_every=args.pulse_every,
        )
        sft_cot_metrics = evaluate_results(sft_cot_results)
        all_metrics["sft_cot"] = sft_cot_metrics
        save_results(sft_cot_results, sft_cot_metrics, output_dir / "sft_cot_results.json")
        print(f"SFT-CoT: accuracy={sft_cot_metrics['accuracy']:.3f}, "
              f"avg_tokens={sft_cot_metrics['avg_tokens_per_problem']:.1f}")
        del sft_model

    # --- SFT+DPO Adaptive evaluation ---
    print("\n=== Adaptive Policy (SFT+DPO) ===")
    trained_model = load_trained_model(base_model, args.checkpoint, args.sft_checkpoint)

    adaptive_results = run_adaptive_policy(
        trained_model, tokenizer, dataset,
        stage_name="Adaptive(SFT+DPO)",
        pulse_every=args.pulse_every,
        max_steps=cfg.get("max_steps", 5),
        max_tokens_per_step=cfg.get("max_tokens_per_step", 256),
    )
    adaptive_metrics = evaluate_results(adaptive_results)
    all_metrics["adaptive_sft_dpo"] = adaptive_metrics
    save_results(adaptive_results, adaptive_metrics, output_dir / "adaptive_results.json")
    print(f"Adaptive(SFT+DPO): accuracy={adaptive_metrics['accuracy']:.3f}, "
          f"avg_tokens={adaptive_metrics['avg_tokens_per_problem']:.1f}")

    # --- Also run CoT on the trained model for fair comparison ---
    print("\n=== CoT with SFT+DPO model ===")
    trained_cot_results = run_cot_baseline(
        trained_model, tokenizer, dataset,
        stage_name="SFT+DPO-CoT",
        pulse_every=args.pulse_every,
    )
    trained_cot_metrics = evaluate_results(trained_cot_results)
    all_metrics["sft_dpo_cot"] = trained_cot_metrics
    save_results(trained_cot_results, trained_cot_metrics, output_dir / "sft_dpo_cot_results.json")
    print(f"SFT+DPO-CoT: accuracy={trained_cot_metrics['accuracy']:.3f}, "
          f"avg_tokens={trained_cot_metrics['avg_tokens_per_problem']:.1f}")

    # --- Comparison summary ---
    print("\n=== Final Comparison ===")
    print(json.dumps(all_metrics, indent=2))
    (output_dir / "comparison.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
