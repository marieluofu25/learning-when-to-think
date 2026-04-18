"""H1 analysis: accuracy-efficiency Pareto frontier across GRPO variants + baselines.

Reads two `comparison.json` files (DeGRPO and vanilla GRPO eval outputs) from
`scripts/eval.py`, builds a combined table with accuracy, avg_tokens,
cost_per_correct, and Wilson 95% CI, then checks Pareto dominance.

Usage:
  python scripts/analyze_h1_pareto.py \
    --main chpc/results/eval/main/comparison.json \
    --vanilla chpc/results/eval/vanilla/comparison.json \
    --out chpc/results/eval/analysis

Outputs:
  <out>/h1_table.csv
  <out>/h1_pareto.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


LABELS = {
    "cot_baseline": "CoT",
    "direct_baseline": "Direct",
    "adaptive_grpo_with_refine": "Adaptive (+refine)",
    "adaptive_grpo_no_refine": "Adaptive (-refine)",
}


def load_comparison(path: Path, source_tag: str) -> list[dict]:
    if not path.exists():
        print(f"[warn] missing comparison file: {path}", file=sys.stderr)
        return []
    data = json.loads(path.read_text())
    rows = []
    for key, metrics in data.items():
        label_base = LABELS.get(key, key)
        if key.startswith("adaptive_"):
            label = f"{label_base} [{source_tag}]"
        else:
            label = label_base
        rows.append(
            {
                "source": source_tag,
                "method_key": key,
                "label": label,
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "accuracy_ci_lo": float(metrics.get("accuracy_ci_95", [0.0, 0.0])[0]),
                "accuracy_ci_hi": float(metrics.get("accuracy_ci_95", [0.0, 0.0])[1]),
                "avg_tokens": float(metrics.get("avg_tokens_per_problem", 0.0)),
                "cost_per_correct": float(metrics.get("cost_per_correct_answer", 0.0)),
                "total": int(metrics.get("total", 0)),
                "correct": int(metrics.get("correct", 0)),
            }
        )
    return rows


def dedupe_baselines(rows: list[dict]) -> list[dict]:
    """Baselines (CoT / direct) run on base model and are identical across runs.
    Keep first occurrence per method_key for those; keep all adaptive rows.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        key = r["method_key"]
        if key.startswith("adaptive_"):
            out.append(r)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def pareto_dominated(rows: list[dict]) -> list[str]:
    """Return labels that are strictly dominated (higher tokens AND lower accuracy)."""
    dominated: list[str] = []
    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            if i == j:
                continue
            better_acc = b["accuracy"] > a["accuracy"]
            fewer_tokens = b["avg_tokens"] < a["avg_tokens"]
            eq_acc = b["accuracy"] >= a["accuracy"]
            eq_tokens = b["avg_tokens"] <= a["avg_tokens"]
            if (better_acc and eq_tokens) or (eq_acc and fewer_tokens):
                dominated.append(a["label"])
                break
    return dominated


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "method_key",
        "label",
        "total",
        "correct",
        "accuracy",
        "accuracy_ci_lo",
        "accuracy_ci_hi",
        "avg_tokens",
        "cost_per_correct",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def plot_pareto(rows: list[dict], path: Path, dominated: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping plot", file=sys.stderr)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in rows:
        x = r["avg_tokens"]
        y = r["accuracy"] * 100
        ci_lo = r["accuracy_ci_lo"] * 100
        ci_hi = r["accuracy_ci_hi"] * 100
        yerr = [[y - ci_lo], [ci_hi - y]]
        marker = "o" if r["label"] in dominated else "D"
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker,
            markersize=9,
            capsize=4,
            label=r["label"],
            linewidth=1.2,
        )
        ax.annotate(
            r["label"],
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.set_xlabel("Avg tokens per problem")
    ax.set_ylabel("Accuracy (%) with 95% Wilson CI")
    ax.set_title("H1: accuracy-efficiency frontier (MATH-500 subset)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--main", type=Path, default=Path("chpc/results/eval/main/comparison.json"))
    p.add_argument("--vanilla", type=Path, default=Path("chpc/results/eval/vanilla/comparison.json"))
    p.add_argument("--out", type=Path, default=Path("chpc/results/eval/analysis"))
    args = p.parse_args()

    rows: list[dict] = []
    rows += load_comparison(args.main, source_tag="DeGRPO")
    rows += load_comparison(args.vanilla, source_tag="vanilla")

    if not rows:
        print("[error] no rows loaded; check --main and --vanilla paths", file=sys.stderr)
        return 1

    rows = dedupe_baselines(rows)
    dominated = pareto_dominated(rows)

    print("\n=== H1 Pareto rows ===")
    for r in rows:
        flag = "DOMINATED" if r["label"] in dominated else "non-dominated"
        print(
            f"{r['label']:<32s}  acc={r['accuracy']:.3f} "
            f"[{r['accuracy_ci_lo']:.3f},{r['accuracy_ci_hi']:.3f}]  "
            f"tokens={r['avg_tokens']:.1f}  cost={r['cost_per_correct']:.1f}  "
            f"({flag})"
        )

    csv_path = args.out / "h1_table.csv"
    plot_path = args.out / "h1_pareto.png"
    write_csv(rows, csv_path)
    plot_pareto(rows, plot_path, dominated)
    print(f"\nTable:  {csv_path}")
    print(f"Plot:   {plot_path}")
    if dominated:
        print(f"Dominated: {', '.join(sorted(set(dominated)))}")
    else:
        print("All methods are Pareto-non-dominated relative to this set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
