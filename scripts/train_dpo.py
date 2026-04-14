"""CLI for DPO training (wraps src.train.dpo.run_dpo)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train.dpo import run_dpo


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training (preference optimization)")
    parser.add_argument("--config", default=None, help="YAML config (optional)")
    parser.add_argument("--sft-checkpoint", default="checkpoints/sft/final")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="checkpoints/dpo")
    parser.add_argument("--num-problems", type=int, default=80)
    parser.add_argument("--samples-per-problem", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        args.sft_checkpoint = cfg.get("sft_checkpoint", args.sft_checkpoint)
        args.model = cfg.get("model_name", args.model)
        args.output_dir = cfg.get("output_dir", args.output_dir)
        args.num_problems = int(cfg.get("num_problems", args.num_problems))
        args.samples_per_problem = int(
            cfg.get("samples_per_problem", args.samples_per_problem)
        )
        args.epochs = int(cfg.get("epochs", args.epochs))
        args.lr = float(cfg.get("lr", args.lr))
        args.beta = float(cfg.get("beta", args.beta))

    run_dpo(
        sft_checkpoint=args.sft_checkpoint,
        base_model_name=args.model,
        output_dir=args.output_dir,
        num_problems=args.num_problems,
        samples_per_problem=args.samples_per_problem,
        num_epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
