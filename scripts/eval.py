"""Evaluate trained adaptive policy vs baselines (GSM8K or HumanEval)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from peft import PeftModel, prepare_model_for_kbit_training

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gsm8k import load_gsm8k
from src.data.humaneval import load_humaneval
from src.eval.evaluate import (
    evaluate_results,
    run_adaptive_humaneval,
    run_adaptive_policy,
    run_cot_baseline,
    run_cot_humaneval,
    run_direct_baseline,
    run_self_consistency,
    save_results,
)
from src.train.model_loading import load_base_causal_lm, load_tokenizer


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
    parser.add_argument("--checkpoint", default=None, help="Override config checkpoint_path")
    parser.add_argument("--sft-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--pulse-every", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir or cfg.get("eval_output_dir", "results/eval"))
    checkpoint = args.checkpoint or cfg.get("checkpoint_path", "checkpoints/grpo/final")
    sft_checkpoint = args.sft_checkpoint if args.sft_checkpoint is not None else cfg.get(
        "sft_checkpoint", "checkpoints/sft/final"
    )

    dataset_name = str(cfg.get("dataset", "gsm8k")).lower()
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
    }
    disable_tools = bool(cfg.get("disable_tools", False))

    all_metrics: dict = {}

    if dataset_name == "humaneval":
        dataset = load_humaneval(subset_size=cfg.get("eval_subset_size"))
        kind = "humaneval"
        print(f"Dataset: HumanEval (n={len(dataset)})", flush=True)

        if not args.skip_baselines:
            print("\n=== CoT Baseline HumanEval (base model) ===", flush=True)
            cot_he = run_cot_humaneval(
                base_model,
                tokenizer,
                dataset,
                stage_name="CoT-HE",
                pulse_every=args.pulse_every,
                max_tokens=cfg.get("humaneval_max_tokens", 512),
            )
            m = evaluate_results(cot_he, dataset_kind=kind)
            all_metrics["cot_baseline_humaneval"] = m
            save_results(cot_he, m, output_dir / "cot_humaneval_results.json")
            print(f"CoT-HE: pass@1={m['accuracy']:.3f}", flush=True)

        print("\n=== Adaptive (GRPO) HumanEval ===", flush=True)
        trained = load_trained_model(base_model, checkpoint, sft_checkpoint)
        adapt_he = run_adaptive_humaneval(
            trained,
            tokenizer,
            dataset,
            stage_name="Adaptive-HE",
            pulse_every=args.pulse_every,
            disable_tools=disable_tools,
            **gen_kw,
        )
        m_ad = evaluate_results(adapt_he, dataset_kind=kind)
        all_metrics["adaptive_humaneval"] = m_ad
        save_results(adapt_he, m_ad, output_dir / "adaptive_humaneval_results.json")
        print(f"Adaptive-HE: pass@1={m_ad['accuracy']:.3f}", flush=True)

    else:
        dataset = load_gsm8k("test", subset_size=cfg.get("eval_subset_size", 100))
        kind = "gsm8k"
        print(f"Dataset: GSM8K test subset (n={len(dataset)})", flush=True)

        if not args.skip_baselines:
            print("\n=== CoT Baseline (base model) ===", flush=True)
            cot_results = run_cot_baseline(
                base_model,
                tokenizer,
                dataset,
                stage_name="CoT",
                pulse_every=args.pulse_every,
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
            )
            direct_metrics = evaluate_results(direct_results, dataset_kind=kind)
            all_metrics["direct_baseline"] = direct_metrics
            save_results(direct_results, direct_metrics, output_dir / "direct_results.json")
            print(
                f"Direct: accuracy={direct_metrics['accuracy']:.3f}, "
                f"avg_tokens={direct_metrics['avg_tokens_per_problem']:.1f}",
                flush=True,
            )

            k = cfg.get("self_consistency_k", 5)
            print(f"\n=== Self-Consistency (k={k}, base model) ===", flush=True)
            sc_results = run_self_consistency(
                base_model,
                tokenizer,
                dataset,
                k=k,
                stage_name=f"SC(k={k})",
                pulse_every=args.pulse_every,
            )
            sc_metrics = evaluate_results(sc_results, dataset_kind=kind)
            all_metrics["self_consistency_baseline"] = sc_metrics
            save_results(sc_results, sc_metrics, output_dir / "sc_results.json")
            print(
                f"SC: accuracy={sc_metrics['accuracy']:.3f}, "
                f"avg_tokens={sc_metrics['avg_tokens_per_problem']:.1f}",
                flush=True,
            )

        sft_path = Path(sft_checkpoint)
        if sft_path.exists():
            print("\n=== CoT with SFT model ===", flush=True)
            sft_model = PeftModel.from_pretrained(base_model, str(sft_path))
            sft_cot_results = run_cot_baseline(
                sft_model,
                tokenizer,
                dataset,
                stage_name="SFT-CoT",
                pulse_every=args.pulse_every,
            )
            sft_cot_metrics = evaluate_results(sft_cot_results, dataset_kind=kind)
            all_metrics["sft_cot"] = sft_cot_metrics
            save_results(sft_cot_results, sft_cot_metrics, output_dir / "sft_cot_results.json")
            print(
                f"SFT-CoT: accuracy={sft_cot_metrics['accuracy']:.3f}, "
                f"avg_tokens={sft_cot_metrics['avg_tokens_per_problem']:.1f}",
                flush=True,
            )
            del sft_model

        print("\n=== Adaptive policy (GRPO LoRA) ===", flush=True)
        trained_model = load_trained_model(base_model, checkpoint, sft_checkpoint)

        adaptive_results = run_adaptive_policy(
            trained_model,
            tokenizer,
            dataset,
            stage_name="Adaptive-GRPO",
            pulse_every=args.pulse_every,
            disable_tools=disable_tools,
            **gen_kw,
        )
        adaptive_metrics = evaluate_results(adaptive_results, dataset_kind=kind)
        all_metrics["adaptive_grpo"] = adaptive_metrics
        save_results(adaptive_results, adaptive_metrics, output_dir / "adaptive_results.json")
        print(
            f"Adaptive: accuracy={adaptive_metrics['accuracy']:.3f}, "
            f"avg_tokens={adaptive_metrics['avg_tokens_per_problem']:.1f}",
            flush=True,
        )

        print("\n=== CoT with same trained adapter stack ===", flush=True)
        trained_cot_results = run_cot_baseline(
            trained_model,
            tokenizer,
            dataset,
            stage_name="Trained-CoT",
            pulse_every=args.pulse_every,
        )
        trained_cot_metrics = evaluate_results(trained_cot_results, dataset_kind=kind)
        all_metrics["trained_cot"] = trained_cot_metrics
        save_results(trained_cot_results, trained_cot_metrics, output_dir / "trained_cot_results.json")
        print(
            f"Trained-CoT: accuracy={trained_cot_metrics['accuracy']:.3f}, "
            f"avg_tokens={trained_cot_metrics['avg_tokens_per_problem']:.1f}",
            flush=True,
        )

    print("\n=== Final Comparison ===", flush=True)
    print(json.dumps(all_metrics, indent=2), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"\nAll results saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
