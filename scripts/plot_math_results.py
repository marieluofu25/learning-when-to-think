"""Generate plots and tables for the Learning When to Think project.

Usage:
    python -m scripts.plot_math_results --results-dir results/default
    python -m scripts.plot_math_results --ablation-summary results/ablations/ablation_summary.json

Generates:
  1. Pareto frontier: accuracy vs avg tokens (H1)
  2. Token allocation by difficulty (H2)
  3. Action distribution by difficulty level (H3)
  4. Beta sweep curves (ablation)
  5. Combined summary figure (2x2)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# Method display names and colors
METHOD_STYLE = {
    "baseline_direct": {"label": "Direct Answer", "color": "#e74c3c", "marker": "s"},
    "baseline_cot": {"label": "CoT (single-pass)", "color": "#3498db", "marker": "^"},
    "baseline_self_consistency_k5": {"label": "Self-Consistency (k=5)", "color": "#9b59b6", "marker": "D"},
    "vanilla_grpo": {"label": "Vanilla GRPO", "color": "#f39c12", "marker": "v"},
    "trained_adaptive": {"label": "Ours (ALP+DeGRPO)", "color": "#2ecc71", "marker": "*"},
}


def load_metrics_files(results_dir: str) -> dict[str, dict]:
    """Load all metrics_*.json files from a results directory."""
    results = {}
    for p in Path(results_dir).glob("metrics_*.json"):
        with open(p) as f:
            data = json.load(f)
        label = p.stem.replace("metrics_", "")
        results[label] = data.get("metrics", data)
    return results


def load_rollouts_file(path: str) -> list[dict]:
    """Load rollouts from a JSONL file."""
    rollouts = []
    with open(path) as f:
        for line in f:
            rollouts.append(json.loads(line))
    return rollouts


def load_ablation_summary(path: str) -> list[dict]:
    """Load ablation summary JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# H1: Pareto frontier (accuracy vs efficiency)
# ---------------------------------------------------------------------------

def plot_pareto(results: list[dict], output_path: str):
    """Accuracy vs avg tokens scatter — Pareto frontier visualization."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        label = r["label"]
        m = r["metrics"]
        style = METHOD_STYLE.get(label, {"label": label, "color": "gray", "marker": "o"})

        ax.scatter(
            m["avg_tokens"], m["accuracy"],
            s=120, zorder=5,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            edgecolors="black", linewidth=0.5,
        )

    ax.set_xlabel("Average Tokens per Problem")
    ax.set_ylabel("Accuracy")
    ax.set_title("H1: Accuracy\u2013Efficiency Pareto Frontier")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved Pareto plot to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# H2: Token allocation by difficulty
# ---------------------------------------------------------------------------

def plot_tokens_by_difficulty(results: list[dict], output_path: str):
    """Average tokens per problem grouped by difficulty level."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Bar chart of tokens by level
    ax = axes[0]
    levels = [1, 2, 3, 4, 5]
    width = 0.15
    offsets = np.arange(len(levels))

    for i, r in enumerate(results):
        m = r["metrics"]
        style = METHOD_STYLE.get(r["label"], {"label": r["label"], "color": "gray"})
        tokens_by_level = m.get("avg_tokens_by_level", {})
        vals = [tokens_by_level.get(str(l), 0) for l in levels]
        ax.bar(
            offsets + i * width, vals, width,
            label=style["label"], color=style["color"], edgecolor="black", linewidth=0.3,
        )

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Avg Tokens")
    ax.set_title("H2: Token Allocation by Difficulty")
    ax.set_xticks(offsets + width * (len(results) - 1) / 2)
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: Accuracy by level
    ax = axes[1]
    for i, r in enumerate(results):
        m = r["metrics"]
        style = METHOD_STYLE.get(r["label"], {"label": r["label"], "color": "gray"})
        acc_by_level = m.get("accuracy_by_level", {})
        vals = [acc_by_level.get(str(l), 0) for l in levels]
        ax.plot(levels, vals, marker="o", label=style["label"], color=style["color"])

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Difficulty Level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(levels)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved difficulty plot to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# H3: Action distribution by difficulty
# ---------------------------------------------------------------------------

def plot_action_distribution(rollouts_path: str, output_path: str):
    """Action usage fractions by difficulty level — stacked bar chart."""
    rollouts = load_rollouts_file(rollouts_path)

    # Group by level
    import re
    by_level: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rollouts:
        level = r["level"]
        if isinstance(level, str):
            m = re.match(r"Level\s+(\d+)", level)
            level = int(m.group(1)) if m else 0
        for action in r.get("actions", []):
            by_level[level][action] += 1

    if not by_level:
        print("No action data found in rollouts.")
        return

    levels = sorted(by_level.keys())
    actions = ["continue", "refine", "terminate"]
    colors = {"continue": "#3498db", "refine": "#f39c12", "terminate": "#2ecc71"}

    # Compute fractions
    fracs = {}
    for lvl in levels:
        total = sum(by_level[lvl].values())
        fracs[lvl] = {a: by_level[lvl].get(a, 0) / total if total > 0 else 0 for a in actions}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(levels))
    width = 0.6
    bottom = np.zeros(len(levels))

    for action in actions:
        vals = [fracs[lvl].get(action, 0) for lvl in levels]
        ax.bar(x, vals, width, bottom=bottom, label=action, color=colors[action],
               edgecolor="black", linewidth=0.3)
        bottom += np.array(vals)

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Action Fraction")
    ax.set_title("H3: Action Distribution by Difficulty")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {l}" for l in levels])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved action distribution plot to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Beta sweep
# ---------------------------------------------------------------------------

def plot_beta_sweep(results: list[dict], output_path: str):
    """Accuracy and avg tokens as a function of beta."""
    beta_results = [r for r in results if "beta" in r]
    if not beta_results:
        print("No beta sweep results found.")
        return

    beta_results.sort(key=lambda r: r["beta"])
    betas = [r["beta"] for r in beta_results]
    accs = [r["metrics"]["accuracy"] for r in beta_results]
    tokens = [r["metrics"]["avg_tokens"] for r in beta_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "#2ecc71"
    ax1.set_xlabel("Beta (length penalty)")
    ax1.set_ylabel("Accuracy", color=color1)
    ax1.plot(betas, accs, "o-", color=color1, label="Accuracy", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    color2 = "#3498db"
    ax2.set_ylabel("Avg Tokens", color=color2)
    ax2.plot(betas, tokens, "s--", color=color2, label="Avg Tokens", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Beta Sweep: Accuracy\u2013Efficiency Tradeoff")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved beta sweep plot to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Combined summary figure (2x2)
# ---------------------------------------------------------------------------

def plot_summary(results: list[dict], rollouts_path: str | None, output_path: str):
    """Combined 2x2 figure: Pareto, tokens by difficulty, accuracy by difficulty, action dist."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Pareto
    ax = axes[0, 0]
    for r in results:
        m = r["metrics"]
        style = METHOD_STYLE.get(r["label"], {"label": r["label"], "color": "gray", "marker": "o"})
        ax.scatter(m["avg_tokens"], m["accuracy"], s=100, color=style["color"],
                   marker=style["marker"], label=style["label"], edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Avg Tokens")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Accuracy\u2013Efficiency Frontier")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # (0,1) Tokens by difficulty
    ax = axes[0, 1]
    levels = [1, 2, 3, 4, 5]
    for r in results:
        m = r["metrics"]
        style = METHOD_STYLE.get(r["label"], {"label": r["label"], "color": "gray"})
        tokens_by_level = m.get("avg_tokens_by_level", {})
        vals = [tokens_by_level.get(str(l), 0) for l in levels]
        ax.plot(levels, vals, marker="o", label=style["label"], color=style["color"])
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Avg Tokens")
    ax.set_title("(b) Token Allocation by Difficulty")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(levels)

    # (1,0) Accuracy by difficulty
    ax = axes[1, 0]
    for r in results:
        m = r["metrics"]
        style = METHOD_STYLE.get(r["label"], {"label": r["label"], "color": "gray"})
        acc_by_level = m.get("accuracy_by_level", {})
        vals = [acc_by_level.get(str(l), 0) for l in levels]
        ax.plot(levels, vals, marker="o", label=style["label"], color=style["color"])
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Accuracy")
    ax.set_title("(c) Accuracy by Difficulty Level")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(levels)

    # (1,1) Action distribution (if rollouts available)
    ax = axes[1, 1]
    if rollouts_path and Path(rollouts_path).exists():
        import re
        rollouts = load_rollouts_file(rollouts_path)
        by_level: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in rollouts:
            level = r["level"]
            if isinstance(level, str):
                m_match = re.match(r"Level\s+(\d+)", level)
                level = int(m_match.group(1)) if m_match else 0
            for action in r.get("actions", []):
                by_level[level][action] += 1

        plot_levels = sorted(by_level.keys())
        action_names = ["continue", "refine", "terminate"]
        colors = {"continue": "#3498db", "refine": "#f39c12", "terminate": "#2ecc71"}
        x = np.arange(len(plot_levels))
        bottom = np.zeros(len(plot_levels))

        for action in action_names:
            total_per_level = [sum(by_level[l].values()) for l in plot_levels]
            vals = [by_level[l].get(action, 0) / t if t > 0 else 0
                    for l, t in zip(plot_levels, total_per_level)]
            ax.bar(x, vals, 0.6, bottom=bottom, label=action, color=colors[action],
                   edgecolor="black", linewidth=0.3)
            bottom += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in plot_levels])
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No action data\navailable", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Action Fraction")
    ax.set_title("(d) Action Distribution by Difficulty")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Learning When to Think \u2014 Results Summary", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved summary plot to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Comparison table (printed to stdout)
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"  {'Method':<35} {'Accuracy':>10} {'Avg Tokens':>12} {'Cost/Correct':>14} {'Avg Steps':>10}")
    print(f"  {'-'*85}")
    for r in results:
        m = r["metrics"]
        style = METHOD_STYLE.get(r["label"], {"label": r["label"]})
        label = style.get("label", r["label"])
        print(
            f"  {label:<35} {m['accuracy']:>10.4f} "
            f"{m['avg_tokens']:>12.1f} {m['cost_per_correct']:>14.1f} "
            f"{m.get('avg_steps', 0):>10.2f}"
        )
    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# Hypothesis verification
# ---------------------------------------------------------------------------

def verify_hypotheses(results: list[dict], adaptive_rollouts_path: str | None):
    """Check H1/H2/H3 against collected results."""
    print(f"\n{'='*60}")
    print("  Hypothesis Verification")
    print(f"{'='*60}")

    # Find our method and baselines
    ours = next((r for r in results if r["label"] == "trained_adaptive"), None)
    cot = next((r for r in results if r["label"] == "baseline_cot"), None)

    # H1: Better Pareto frontier
    print("\n  H1: ALP+DeGRPO improves accuracy-efficiency Pareto frontier")
    if ours and cot:
        ours_m = ours["metrics"]
        cot_m = cot["metrics"]
        better_acc = ours_m["accuracy"] >= cot_m["accuracy"]
        fewer_tokens = ours_m["avg_tokens"] <= cot_m["avg_tokens"]
        better_cost = ours_m["cost_per_correct"] <= cot_m["cost_per_correct"]
        print(f"    Accuracy:  Ours={ours_m['accuracy']:.4f} vs CoT={cot_m['accuracy']:.4f} "
              f"{'PASS' if better_acc else 'FAIL'}")
        print(f"    Tokens:    Ours={ours_m['avg_tokens']:.1f} vs CoT={cot_m['avg_tokens']:.1f} "
              f"{'PASS' if fewer_tokens else 'FAIL'}")
        print(f"    Cost/Corr: Ours={ours_m['cost_per_correct']:.1f} vs CoT={cot_m['cost_per_correct']:.1f} "
              f"{'PASS' if better_cost else 'FAIL'}")
        h1 = better_acc or fewer_tokens or better_cost
        print(f"    H1 overall: {'SUPPORTED' if h1 else 'NOT SUPPORTED'}")
    else:
        print("    [Missing results -- need trained_adaptive and baseline_cot]")

    # H2: More tokens on harder problems
    print("\n  H2: Learned policy allocates more tokens to harder problems")
    if ours:
        tokens_by_level = ours["metrics"].get("avg_tokens_by_level", {})
        if tokens_by_level:
            levels_sorted = sorted(tokens_by_level.keys(), key=int)
            vals = [tokens_by_level[l] for l in levels_sorted]
            increasing = all(v1 <= v2 for v1, v2 in zip(vals, vals[1:]))
            print(f"    Tokens by level: {dict(zip(levels_sorted, [f'{v:.0f}' for v in vals]))}")
            print(f"    Monotonically increasing: {'YES' if increasing else 'NO'}")

            if adaptive_rollouts_path and Path(adaptive_rollouts_path).exists():
                import re
                rollouts = load_rollouts_file(adaptive_rollouts_path)
                levels_list = []
                tokens_list = []
                for r in rollouts:
                    lvl = r["level"]
                    if isinstance(lvl, str):
                        m = re.match(r"Level\s+(\d+)", lvl)
                        lvl = int(m.group(1)) if m else 0
                    levels_list.append(lvl)
                    tokens_list.append(r["num_tokens"])
                from scipy.stats import spearmanr
                corr, pval = spearmanr(levels_list, tokens_list)
                print(f"    Spearman correlation: r={corr:.4f}, p={pval:.4f}")
                print(f"    H2: {'SUPPORTED' if corr > 0 and pval < 0.05 else 'NOT SUPPORTED'}")
            else:
                print(f"    H2: {'LIKELY SUPPORTED' if increasing else 'NOT SUPPORTED'} (no rollout data for correlation)")
        else:
            print("    [No per-level token data available]")
    else:
        print("    [Missing trained_adaptive results]")

    # H3: Action distribution varies by difficulty
    print("\n  H3: Action usage shifts with difficulty")
    if ours:
        action_fracs = ours["metrics"].get("action_fractions_by_level", {})
        if action_fracs:
            terminate_fracs = {}
            for lvl, fracs in sorted(action_fracs.items(), key=lambda x: int(x[0])):
                terminate_fracs[lvl] = fracs.get("terminate", 0)
                print(f"    Level {lvl}: " + ", ".join(f"{a}={v:.1%}" for a, v in fracs.items()))
            t_vals = list(terminate_fracs.values())
            decreasing = all(v1 >= v2 for v1, v2 in zip(t_vals, t_vals[1:]))
            print(f"    Terminate decreases with difficulty: {'YES' if decreasing else 'NO'}")
            print(f"    H3: {'SUPPORTED' if decreasing else 'NOT SUPPORTED'}")
        else:
            print("    [No action fraction data available]")
    else:
        print("    [Missing trained_adaptive results]")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate plots and tables")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory with metrics_*.json files")
    parser.add_argument("--ablation-summary", type=str, default=None,
                        help="Path to ablation_summary.json")
    parser.add_argument("--adaptive-rollouts", type=str, default=None,
                        help="Path to adaptive method rollouts JSONL (for action distribution)")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if args.results_dir:
        metrics = load_metrics_files(args.results_dir)
        for label, m in metrics.items():
            results.append({"label": label, "metrics": m})

    if args.ablation_summary:
        results = load_ablation_summary(args.ablation_summary)

    if not results:
        print("No results found. Provide --results-dir or --ablation-summary.")
        return

    print_comparison_table(results)

    plot_pareto(results, str(out_dir / "pareto_frontier.png"))
    plot_tokens_by_difficulty(results, str(out_dir / "tokens_by_difficulty.png"))

    if args.adaptive_rollouts:
        plot_action_distribution(args.adaptive_rollouts, str(out_dir / "action_distribution.png"))

    plot_summary(results, args.adaptive_rollouts, str(out_dir / "summary_2x2.png"))

    beta_results = [r for r in results if "beta" in r]
    if beta_results:
        plot_beta_sweep(results, str(out_dir / "beta_sweep.png"))

    verify_hypotheses(results, args.adaptive_rollouts)

    print(f"All plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
