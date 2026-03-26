"""Generate accuracy-vs-tokens scatter plot from evaluation results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


LABELS = {
    "cot_baseline": "CoT Baseline",
    "self_consistency_baseline": "Self-Consistency (k=3)",
    "sft_cot": "SFT-CoT (teacher distilled)",
    "adaptive_sft_dpo": "Adaptive (SFT+DPO)",
    "sft_dpo_cot": "SFT+DPO CoT",
}

COLORS = {
    "cot_baseline": "#4A90D9",
    "self_consistency_baseline": "#7B68EE",
    "sft_cot": "#2ECC71",
    "adaptive_sft_dpo": "#E74C3C",
    "sft_dpo_cot": "#F39C12",
}


def main():
    results_dir = Path("results/eval_distill")
    comparison_path = results_dir / "comparison.json"

    if not comparison_path.exists():
        print(f"No comparison.json found at {comparison_path}")
        sys.exit(1)

    with open(comparison_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Scatter: Accuracy vs Avg Tokens ---
    ax1 = axes[0]
    for key, metrics in data.items():
        label = LABELS.get(key, key)
        color = COLORS.get(key, "#333")
        ax1.scatter(
            metrics["avg_tokens_per_problem"],
            metrics["accuracy"] * 100,
            s=200, c=color, label=label, edgecolors="black", linewidths=0.8, zorder=3,
        )
    ax1.set_xlabel("Avg Tokens per Problem", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy vs Compute Cost", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, max(m["accuracy"] * 100 for m in data.values()) + 10)

    # --- Bar: Accuracy comparison ---
    ax2 = axes[1]
    names = [LABELS.get(k, k) for k in data]
    accuracies = [data[k]["accuracy"] * 100 for k in data]
    colors = [COLORS.get(k, "#333") for k in data]
    bars = ax2.barh(names, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Accuracy (%)", fontsize=12)
    ax2.set_title("GSM8K Accuracy Comparison", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, max(accuracies) + 10)
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = results_dir / "accuracy_vs_tokens.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    # --- Print summary table ---
    print("\n" + "=" * 70)
    print(f"{'Method':<30} {'Accuracy':>10} {'Avg Tokens':>12} {'Cost/Correct':>14}")
    print("-" * 70)
    for key, metrics in data.items():
        label = LABELS.get(key, key)
        print(f"{label:<30} {metrics['accuracy']*100:>9.1f}% {metrics['avg_tokens_per_problem']:>12.1f} {metrics['cost_per_correct_answer']:>14.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
