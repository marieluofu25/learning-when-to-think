"""Generate accuracy-vs-tokens plot from evaluation comparison.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


LABELS = {
    "cot_baseline": "CoT baseline",
    "direct_baseline": "Direct answer",
    "self_consistency_baseline": "Self-consistency",
    "sft_cot": "SFT CoT",
    "adaptive_grpo": "Adaptive (GRPO)",
    "adaptive_sft_dpo": "Adaptive (legacy key)",
    "trained_cot": "CoT (trained stack)",
    "sft_dpo_cot": "SFT+DPO CoT",
    "cot_baseline_humaneval": "CoT HumanEval",
    "adaptive_humaneval": "Adaptive HumanEval",
}

COLORS = {
    "cot_baseline": "#4A90D9",
    "direct_baseline": "#1abc9c",
    "self_consistency_baseline": "#7B68EE",
    "sft_cot": "#2ECC71",
    "adaptive_grpo": "#E74C3C",
    "adaptive_sft_dpo": "#E74C3C",
    "trained_cot": "#F39C12",
    "sft_dpo_cot": "#95a5a6",
    "cot_baseline_humaneval": "#3498db",
    "adaptive_humaneval": "#c0392b",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/eval/comparison.json"),
        help="Path to comparison.json from scripts/eval.py",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    comparison_path = args.comparison
    if not comparison_path.exists():
        print(f"No comparison file: {comparison_path}")
        sys.exit(1)

    with open(comparison_path) as f:
        data = json.load(f)

    if not data:
        print("Empty comparison.json")
        sys.exit(1)

    out_dir = comparison_path.parent
    out_path = args.out or (out_dir / "accuracy_vs_tokens.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    for key, metrics in data.items():
        label = LABELS.get(key, key)
        color = COLORS.get(key, "#333333")
        ax1.scatter(
            metrics["avg_tokens_per_problem"],
            metrics["accuracy"] * 100,
            s=200,
            c=color,
            label=label,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
    ax1.set_xlabel("Avg tokens per problem", fontsize=12)
    ax1.set_ylabel("Accuracy / pass@1 (%)", fontsize=12)
    ax1.set_title("Accuracy vs compute", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ymax = max(m["accuracy"] * 100 for m in data.values()) + 10
    ax1.set_ylim(-5, max(ymax, 15))

    ax2 = axes[1]
    names = [LABELS.get(k, k) for k in data]
    accuracies = [data[k]["accuracy"] * 100 for k in data]
    colors = [COLORS.get(k, "#333333") for k in data]
    bars = ax2.barh(names, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, accuracies):
        ax2.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_xlabel("Accuracy / pass@1 (%)", fontsize=12)
    ax2.set_title("Comparison", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, max(accuracies) + 10)
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    print("\n" + "=" * 72)
    print(f"{'Method':<28} {'Acc':>8} {'Tok/prob':>12} {'Cost/corr':>12} {'Tool calls':>10}")
    print("-" * 72)
    for key, metrics in data.items():
        label = LABELS.get(key, key)[:28]
        tc = metrics.get("total_tool_calls", 0)
        print(
            f"{label:<28} {metrics['accuracy']*100:>7.1f}% "
            f"{metrics['avg_tokens_per_problem']:>12.1f} "
            f"{metrics['cost_per_correct_answer']:>12.1f} {tc:>10}"
        )
    print("=" * 72)

    for key, metrics in data.items():
        ac = metrics.get("action_counts_total")
        if ac:
            print(f"\n{key} action_counts_total: {ac}")


if __name__ == "__main__":
    main()
