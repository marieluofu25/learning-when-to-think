"""Difficulty-Aware Evaluation & Analysis for Learning When to Think.

This script:
1. Categorizes GSM8K problems by difficulty (number of reasoning steps in
   the ground-truth solution).
2. Loads evaluation result JSON files produced by the eval pipeline.
3. Generates per-difficulty breakdowns and publication-ready plots:
   - Token usage vs. difficulty
   - Accuracy vs. difficulty
   - Action distribution vs. difficulty (for adaptive methods)
   - Efficiency scatter (accuracy vs. tokens) per difficulty bin

Usage:
    # Using result JSON files from the eval pipeline:
    python scripts/difficulty_analysis.py --results-dir results/eval_distill

    # Specify individual result files:
    python scripts/difficulty_analysis.py \\
        --result-files results/eval_distill/cot_results.json \\
                       results/eval_distill/adaptive_results.json

    # Just generate the difficulty map (no model results needed):
    python scripts/difficulty_analysis.py --difficulty-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# 1. Difficulty Classification
# ---------------------------------------------------------------------------

# GSM8K solutions annotate each arithmetic operation as <<expr=result>>.
_CALC_PATTERN = re.compile(r"<<[^>]+=[\d.,]+>>")


def count_reasoning_steps(full_answer: str) -> int:
    """Count the number of calculation steps in a GSM8K gold solution."""
    return len(_CALC_PATTERN.findall(full_answer))


def classify_difficulty(gt_steps: int) -> str:
    """Bin a step count into easy / medium / hard."""
    if gt_steps <= 2:
        return "easy"
    elif gt_steps <= 4:
        return "medium"
    else:
        return "hard"


def build_difficulty_map(split: str = "test", subset_size: int | None = None) -> list[dict]:
    """Load GSM8K and annotate each problem with step count and difficulty.

    Returns a list of dicts with keys:
        question, answer_number, full_answer, gt_steps, difficulty
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    items = []
    for row in ds:
        gt_steps = count_reasoning_steps(row["answer"])
        m = re.search(r"####\s*(-?[\d,]+\.?\d*)", row["answer"])
        answer_number = float(m.group(1).replace(",", "")) if m else None
        items.append({
            "question": row["question"],
            "answer_number": answer_number,
            "full_answer": row["answer"],
            "gt_steps": gt_steps,
            "difficulty": classify_difficulty(gt_steps),
        })
    if subset_size is not None:
        items = items[:subset_size]
    return items


def print_difficulty_distribution(items: list[dict]) -> None:
    """Print a summary of the difficulty distribution."""
    step_counts = Counter(item["gt_steps"] for item in items)
    diff_counts = Counter(item["difficulty"] for item in items)

    print("\n=== Difficulty Distribution ===")
    print(f"Total problems: {len(items)}")
    print(f"\nBy difficulty bin:")
    for level in ["easy", "medium", "hard"]:
        count = diff_counts.get(level, 0)
        pct = 100 * count / len(items)
        print(f"  {level:>6s}: {count:4d} ({pct:5.1f}%)")
    print(f"\nBy step count:")
    for steps in sorted(step_counts):
        count = step_counts[steps]
        print(f"  {steps} steps: {count:4d}")


# ---------------------------------------------------------------------------
# 2. Load and Match Results
# ---------------------------------------------------------------------------

def load_result_file(path: Path) -> dict:
    """Load a result JSON file from the eval pipeline.

    Expected format: {"metrics": {...}, "results": [...], "timestamp": ...}
    Each entry in results has: question, gold, predicted, correct, total_tokens,
    num_steps, answer_text, terminated (for adaptive).
    """
    with open(path) as f:
        return json.load(f)


def match_results_to_difficulty(
    results: list[dict],
    difficulty_map: list[dict],
) -> list[dict]:
    """Annotate each result with difficulty info by matching on question text.

    Returns a new list of dicts with added keys: gt_steps, difficulty.
    Note: uses 'gt_steps' (ground-truth steps) to avoid collision with the
    model's 'num_steps' field (number of adaptive loop iterations).
    """
    q_to_diff = {}
    for item in difficulty_map:
        key = item["question"].strip()
        q_to_diff[key] = {
            "gt_steps": item["gt_steps"],
            "difficulty": item["difficulty"],
        }

    annotated = []
    matched = 0
    for r in results:
        key = r["question"].strip()
        if key in q_to_diff:
            annotated.append({**r, **q_to_diff[key]})
            matched += 1
        else:
            annotated.append({**r, "gt_steps": -1, "difficulty": "unknown"})

    print(f"  Matched {matched}/{len(results)} results to difficulty map")
    return annotated


# ---------------------------------------------------------------------------
# 3. Per-Difficulty Metrics
# ---------------------------------------------------------------------------

def compute_per_difficulty_metrics(annotated_results: list[dict]) -> dict:
    """Compute metrics grouped by difficulty bin.

    Returns:
        {
            "easy":   {"total": N, "correct": N, "accuracy": float, ...},
            "medium": {...},
            "hard":   {...},
            "all":    {...},
        }
    """
    groups = defaultdict(list)
    for r in annotated_results:
        groups[r["difficulty"]].append(r)
        groups["all"].append(r)

    metrics = {}
    for level, items in groups.items():
        total = len(items)
        correct = sum(1 for r in items if r["correct"])
        total_tokens = sum(r["total_tokens"] for r in items)
        avg_tokens = total_tokens / max(total, 1)
        avg_model_steps = sum(r.get("num_steps", 1) for r in items) / max(total, 1)
        terminated = sum(1 for r in items if r.get("terminated", False))

        metrics[level] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
            "avg_tokens": avg_tokens,
            "total_tokens": total_tokens,
            "avg_model_steps": avg_model_steps,
            "cost_per_correct": total_tokens / max(correct, 1),
            "terminated_rate": terminated / max(total, 1),
        }
    return metrics


def compute_per_step_metrics(annotated_results: list[dict]) -> dict:
    """Compute metrics grouped by exact ground-truth step count."""
    groups = defaultdict(list)
    for r in annotated_results:
        groups[r["gt_steps"]].append(r)

    metrics = {}
    for steps, items in sorted(groups.items()):
        if steps < 0:
            continue
        total = len(items)
        correct = sum(1 for r in items if r["correct"])
        avg_tokens = sum(r["total_tokens"] for r in items) / max(total, 1)
        avg_model_steps = sum(r.get("num_steps", 1) for r in items) / max(total, 1)
        metrics[steps] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
            "avg_tokens": avg_tokens,
            "avg_model_steps": avg_model_steps,
        }
    return metrics


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "cot_results": "CoT Baseline",
    "sc_results": "Self-Consistency",
    "self_consistency_results": "Self-Consistency",
    "adaptive_results": "Adaptive (SFT+DPO)",
    "sft_cot_results": "SFT-CoT",
    "sft_dpo_cot_results": "SFT+DPO CoT",
    "grpo_results": "GRPO Adaptive",
}

METHOD_COLORS = {
    "cot_results": "#4A90D9",
    "sc_results": "#7B68EE",
    "self_consistency_results": "#7B68EE",
    "adaptive_results": "#E74C3C",
    "sft_cot_results": "#2ECC71",
    "sft_dpo_cot_results": "#F39C12",
    "grpo_results": "#9B59B6",
}


def _get_label(method_key: str) -> str:
    return METHOD_LABELS.get(method_key, method_key.replace("_", " ").title())


def _get_color(method_key: str) -> str:
    return METHOD_COLORS.get(method_key, "#333333")


def plot_tokens_vs_difficulty(
    all_method_metrics: dict[str, dict],
    output_path: Path,
) -> None:
    """Bar chart: average tokens per problem, grouped by difficulty."""
    difficulties = ["easy", "medium", "hard"]
    methods = list(all_method_metrics.keys())
    x = np.arange(len(difficulties))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        metrics = all_method_metrics[method]
        vals = [metrics.get(d, {}).get("avg_tokens", 0) for d in difficulties]
        bars = ax.bar(
            x + i * width, vals, width,
            label=_get_label(method), color=_get_color(method),
            edgecolor="black", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("Problem Difficulty", fontsize=12)
    ax.set_ylabel("Average Tokens per Problem", fontsize=12)
    ax.set_title("Token Usage vs. Problem Difficulty", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_vs_difficulty(
    all_method_metrics: dict[str, dict],
    output_path: Path,
) -> None:
    """Bar chart: accuracy grouped by difficulty."""
    difficulties = ["easy", "medium", "hard"]
    methods = list(all_method_metrics.keys())
    x = np.arange(len(difficulties))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        metrics = all_method_metrics[method]
        vals = [metrics.get(d, {}).get("accuracy", 0) * 100 for d in difficulties]
        bars = ax.bar(
            x + i * width, vals, width,
            label=_get_label(method), color=_get_color(method),
            edgecolor="black", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("Problem Difficulty", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs. Problem Difficulty", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_tokens_vs_steps(
    all_method_step_metrics: dict[str, dict],
    output_path: Path,
) -> None:
    """Line plot: average tokens vs. exact step count (fine-grained)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, step_metrics in all_method_step_metrics.items():
        steps_sorted = sorted(s for s in step_metrics if s > 0)
        if not steps_sorted:
            continue
        x_vals = steps_sorted
        y_vals = [step_metrics[s]["avg_tokens"] for s in x_vals]
        ax.plot(x_vals, y_vals, marker="o", label=_get_label(method),
                color=_get_color(method), linewidth=2, markersize=6)

    ax.set_xlabel("Number of Reasoning Steps (Ground Truth)", fontsize=12)
    ax.set_ylabel("Average Tokens Generated", fontsize=12)
    ax.set_title("Token Usage vs. Problem Complexity", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_vs_steps(
    all_method_step_metrics: dict[str, dict],
    output_path: Path,
) -> None:
    """Line plot: accuracy vs. exact step count."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, step_metrics in all_method_step_metrics.items():
        steps_sorted = sorted(s for s in step_metrics if s > 0)
        if not steps_sorted:
            continue
        x_vals = steps_sorted
        y_vals = [step_metrics[s]["accuracy"] * 100 for s in x_vals]
        ax.plot(x_vals, y_vals, marker="s", label=_get_label(method),
                color=_get_color(method), linewidth=2, markersize=6)

    ax.set_xlabel("Number of Reasoning Steps (Ground Truth)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs. Problem Complexity", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_efficiency_scatter(
    all_method_metrics: dict[str, dict],
    output_path: Path,
) -> None:
    """Scatter plot: accuracy vs. avg tokens, one point per (method, difficulty)."""
    difficulties = ["easy", "medium", "hard"]
    markers = {"easy": "o", "medium": "s", "hard": "D"}

    fig, ax = plt.subplots(figsize=(10, 7))
    for method, metrics in all_method_metrics.items():
        color = _get_color(method)
        for diff in difficulties:
            if diff not in metrics:
                continue
            m = metrics[diff]
            ax.scatter(
                m["avg_tokens"], m["accuracy"] * 100,
                s=150, c=color, marker=markers[diff],
                edgecolors="black", linewidth=0.5, zorder=3,
            )
            ax.annotate(
                f"{_get_label(method)}\n({diff})",
                (m["avg_tokens"], m["accuracy"] * 100),
                fontsize=7, ha="center", va="bottom",
                textcoords="offset points", xytext=(0, 8),
            )

    for diff, marker in markers.items():
        ax.scatter([], [], marker=marker, c="gray", s=80,
                   label=f"{diff.capitalize()} problems", edgecolors="black")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlabel("Average Tokens per Problem", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Efficiency: Accuracy vs. Compute by Difficulty",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_summary(
    all_method_metrics: dict[str, dict],
    output_path: Path,
) -> None:
    """2x2 summary figure for the paper."""
    difficulties = ["easy", "medium", "hard"]
    methods = list(all_method_metrics.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(difficulties))
    width = 0.8 / max(len(methods), 1)

    # (a) Accuracy vs difficulty
    ax = axes[0, 0]
    for i, method in enumerate(methods):
        m = all_method_metrics[method]
        vals = [m.get(d, {}).get("accuracy", 0) * 100 for d in difficulties]
        ax.bar(x + i * width, vals, width, label=_get_label(method),
               color=_get_color(method), edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Accuracy vs. Difficulty")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    # (b) Tokens vs difficulty
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        m = all_method_metrics[method]
        vals = [m.get(d, {}).get("avg_tokens", 0) for d in difficulties]
        ax.bar(x + i * width, vals, width, label=_get_label(method),
               color=_get_color(method), edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Avg Tokens")
    ax.set_title("(b) Token Usage vs. Difficulty")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    # (c) Cost per correct answer vs difficulty
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        m = all_method_metrics[method]
        vals = [m.get(d, {}).get("cost_per_correct", 0) for d in difficulties]
        vals = [min(v, 20000) for v in vals]
        ax.bar(x + i * width, vals, width, label=_get_label(method),
               color=_get_color(method), edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Tokens per Correct Answer")
    ax.set_title("(c) Compute Efficiency vs. Difficulty")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    # (d) Accuracy vs tokens scatter (overall per method)
    ax = axes[1, 1]
    for method in methods:
        m = all_method_metrics[method].get("all", {})
        if not m:
            continue
        ax.scatter(
            m["avg_tokens"], m["accuracy"] * 100,
            s=200, c=_get_color(method), label=_get_label(method),
            edgecolors="black", linewidth=0.8, zorder=3,
        )
    ax.set_xlabel("Avg Tokens per Problem")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(d) Overall Accuracy vs. Compute")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Difficulty-Aware Analysis: Learning When to Think",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Print summary tables
# ---------------------------------------------------------------------------

def print_summary_table(all_method_metrics: dict[str, dict]) -> None:
    """Print a formatted comparison table."""
    difficulties = ["easy", "medium", "hard", "all"]

    print("\n" + "=" * 90)
    print(f"{'Method':<25} {'Difficulty':<10} {'N':>5} {'Correct':>8} "
          f"{'Accuracy':>9} {'Avg Tok':>9} {'Cost/Corr':>10}")
    print("-" * 90)

    for method, metrics in all_method_metrics.items():
        label = _get_label(method)
        for diff in difficulties:
            if diff not in metrics:
                continue
            m = metrics[diff]
            cost = m["cost_per_correct"]
            cost_str = f"{cost:.0f}" if cost < 100000 else "inf"
            print(f"{label:<25} {diff:<10} {m['total']:>5} {m['correct']:>8} "
                  f"{m['accuracy']*100:>8.1f}% {m['avg_tokens']:>9.1f} {cost_str:>10}")
        print("-" * 90)
    print("=" * 90)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Difficulty-aware analysis for Learning When to Think"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory containing result JSON files from eval pipeline",
    )
    parser.add_argument(
        "--result-files", nargs="+", default=None,
        help="Individual result JSON files to analyze",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/difficulty_analysis",
        help="Where to save plots and analysis output",
    )
    parser.add_argument(
        "--subset-size", type=int, default=None,
        help="Only use first N problems from GSM8K test set",
    )
    parser.add_argument(
        "--difficulty-only", action="store_true",
        help="Just print difficulty distribution, don't analyze results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build difficulty map
    print("Loading GSM8K and computing difficulty map...")
    difficulty_map = build_difficulty_map("test", subset_size=args.subset_size)
    print_difficulty_distribution(difficulty_map)

    # Save difficulty map for reuse
    diff_map_path = output_dir / "difficulty_map.json"
    with open(diff_map_path, "w") as f:
        json.dump(
            [{"question": d["question"], "gt_steps": d["gt_steps"],
              "difficulty": d["difficulty"], "answer_number": d["answer_number"]}
             for d in difficulty_map],
            f, indent=2,
        )
    print(f"Saved difficulty map to {diff_map_path}")

    if args.difficulty_only:
        return

    # Step 2: Find and load result files
    result_files = []
    if args.result_files:
        result_files = [Path(f) for f in args.result_files]
    elif args.results_dir:
        results_dir = Path(args.results_dir)
        result_files = sorted(results_dir.glob("*_results.json"))
    else:
        for candidate in ["results/eval_distill", "results/baselines", "results"]:
            p = Path(candidate)
            if p.exists():
                result_files.extend(p.glob("*_results.json"))

    if not result_files:
        print("\nNo result files found. Run the eval pipeline first, then re-run this script.")
        print("Example: python scripts/eval.py --config configs/default.yaml")
        return

    print(f"\nFound {len(result_files)} result files:")
    for f in result_files:
        print(f"  {f}")

    # Step 3: Load results and match to difficulty
    all_method_metrics = {}
    all_method_step_metrics = {}

    for result_file in result_files:
        method_key = result_file.stem
        print(f"\nProcessing: {method_key}")

        data = load_result_file(result_file)
        results = data.get("results", [])
        if not results:
            print(f"  No results in {result_file}, skipping")
            continue

        annotated = match_results_to_difficulty(results, difficulty_map)
        per_diff = compute_per_difficulty_metrics(annotated)
        per_step = compute_per_step_metrics(annotated)

        all_method_metrics[method_key] = per_diff
        all_method_step_metrics[method_key] = per_step

    if not all_method_metrics:
        print("\nNo valid results to analyze.")
        return

    # Step 4: Print summary
    print_summary_table(all_method_metrics)

    # Step 5: Generate plots
    print("\nGenerating plots...")
    plot_tokens_vs_difficulty(all_method_metrics, output_dir / "tokens_vs_difficulty.png")
    plot_accuracy_vs_difficulty(all_method_metrics, output_dir / "accuracy_vs_difficulty.png")
    plot_tokens_vs_steps(all_method_step_metrics, output_dir / "tokens_vs_steps.png")
    plot_accuracy_vs_steps(all_method_step_metrics, output_dir / "accuracy_vs_steps.png")
    plot_efficiency_scatter(all_method_metrics, output_dir / "efficiency_scatter.png")
    plot_combined_summary(all_method_metrics, output_dir / "combined_summary.png")

    # Step 6: Save full analysis as JSON
    analysis_path = output_dir / "full_analysis.json"
    analysis = {
        "per_difficulty": {
            method: {diff: m for diff, m in metrics.items()}
            for method, metrics in all_method_metrics.items()
        },
        "per_step": {
            method: {str(s): m for s, m in metrics.items()}
            for method, metrics in all_method_step_metrics.items()
        },
    }
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFull analysis saved to {analysis_path}")
    print(f"All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
