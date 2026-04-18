"""H2 and H3 analysis on MATH-500 adaptive eval results.

H2: learned policy allocates more compute (tokens, steps) to harder prompts.
H3: action usage shifts with difficulty (more continue/refine on hard, more
    terminate on easy).

Reads the per-problem JSON produced by scripts/eval.py (either the final
`*_results.json` or the incremental `*_results.partial.jsonl`). Each record
must include `level` (1-5), `total_tokens`, `num_steps`, and `action_counts`.

Usage:
  python scripts/analyze_h2_h3.py \
    --input chpc/results/eval/main/adaptive_with_refine_results.json \
    --out chpc/results/eval/analysis \
    --tag adaptive_with_refine

Outputs (under --out):
  h2_<tag>_per_level.csv
  h2_<tag>_tokens_vs_level.png
  h3_<tag>_action_dist_per_level.csv
  h3_<tag>_action_dist_per_level.png
  summary_<tag>.json  (Spearman correlation + per-level means)

If the policy was trained with allow_refine=false, H3 will only show
continue/terminate; the script still produces the plot and notes the
reduced action set in the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_records(path: Path) -> list[dict]:
    """Load per-problem records. Accept either final JSON or partial JSONL."""
    if not path.exists():
        print(f"[error] missing input: {path}", file=sys.stderr)
        return []
    if path.suffix == ".jsonl":
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "results" in data:
        return list(data["results"])
    if isinstance(data, list):
        return data
    print(f"[error] unexpected JSON shape in {path}", file=sys.stderr)
    return []


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation with tie-averaged ranks. Returns NaN if n<2."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return float("nan")

    def rank(values: list[float]) -> list[float]:
        indexed = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def aggregate_per_level(records: list[dict]) -> dict[int, dict]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        lvl = r.get("level")
        if lvl is None:
            continue
        try:
            lvl_int = int(lvl)
        except (TypeError, ValueError):
            continue
        buckets[lvl_int].append(r)

    per_level: dict[int, dict] = {}
    for lvl in sorted(buckets.keys()):
        items = buckets[lvl]
        tokens = [float(x.get("total_tokens", 0)) for x in items]
        steps = [float(x.get("num_steps", 0)) for x in items]
        correct = [1.0 if x.get("correct") else 0.0 for x in items]
        actions_sum: dict[str, int] = defaultdict(int)
        total_actions = 0
        for x in items:
            ac = x.get("action_counts") or {}
            for k, v in ac.items():
                actions_sum[k] += int(v)
                total_actions += int(v)
        per_level[lvl] = {
            "n": len(items),
            "mean_tokens": mean(tokens) if tokens else 0.0,
            "mean_steps": mean(steps) if steps else 0.0,
            "accuracy": mean(correct) if correct else 0.0,
            "action_counts": dict(actions_sum),
            "action_fractions": {
                k: (v / total_actions if total_actions else 0.0)
                for k, v in actions_sum.items()
            },
        }
    return per_level


def write_h2_csv(per_level: dict[int, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "n", "mean_tokens", "mean_steps", "accuracy"])
        for lvl, stats in sorted(per_level.items()):
            w.writerow(
                [lvl, stats["n"], f"{stats['mean_tokens']:.2f}",
                 f"{stats['mean_steps']:.2f}", f"{stats['accuracy']:.4f}"]
            )


def write_h3_csv(per_level: dict[int, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actions = sorted({k for s in per_level.values() for k in s["action_counts"].keys()})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "n"] + [f"count_{a}" for a in actions] + [f"frac_{a}" for a in actions])
        for lvl, stats in sorted(per_level.items()):
            counts = [stats["action_counts"].get(a, 0) for a in actions]
            fracs = [f"{stats['action_fractions'].get(a, 0.0):.4f}" for a in actions]
            w.writerow([lvl, stats["n"]] + counts + fracs)


def plot_h2(per_level: dict[int, dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping H2 plot", file=sys.stderr)
        return
    if not per_level:
        return
    levels = sorted(per_level.keys())
    tokens = [per_level[l]["mean_tokens"] for l in levels]
    steps = [per_level[l]["mean_steps"] for l in levels]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar([str(l) for l in levels], tokens, alpha=0.7, label="avg tokens")
    ax1.set_xlabel("MATH-500 level (1=easy ... 5=hard)")
    ax1.set_ylabel("Avg tokens per problem")
    ax2 = ax1.twinx()
    ax2.plot([str(l) for l in levels], steps, "o-", color="tab:red", label="avg steps")
    ax2.set_ylabel("Avg steps per problem")
    ax1.set_title("H2: compute vs difficulty")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_h3(per_level: dict[int, dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping H3 plot", file=sys.stderr)
        return
    if not per_level:
        return
    actions = sorted({k for s in per_level.values() for k in s["action_counts"].keys()})
    if not actions:
        print("[warn] no action_counts found; skipping H3 plot", file=sys.stderr)
        return
    levels = sorted(per_level.keys())
    level_labels = [str(l) for l in levels]
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = [0.0] * len(levels)
    for a in actions:
        vals = [per_level[l]["action_fractions"].get(a, 0.0) for l in levels]
        ax.bar(level_labels, vals, bottom=bottom, label=a)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylim(0, 1)
    ax.set_xlabel("MATH-500 level")
    ax.set_ylabel("Action fraction")
    ax.set_title("H3: action distribution vs difficulty")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("chpc/results/eval/main/adaptive_with_refine_results.json"),
    )
    p.add_argument("--out", type=Path, default=Path("chpc/results/eval/analysis"))
    p.add_argument("--tag", default="adaptive_with_refine")
    args = p.parse_args()

    records = load_records(args.input)
    if not records:
        return 1

    levels_present = sum(1 for r in records if r.get("level") is not None)
    if levels_present == 0:
        print(
            "[error] no records have `level` field; rerun eval after augmenting "
            "src/eval/evaluate.py with level/subject/unique_id.",
            file=sys.stderr,
        )
        return 1

    per_level = aggregate_per_level(records)

    xs = []
    ys = []
    for r in records:
        lvl = r.get("level")
        if lvl is None:
            continue
        try:
            xs.append(float(lvl))
            ys.append(float(r.get("total_tokens", 0)))
        except (TypeError, ValueError):
            continue
    rho_tokens = spearman(xs, ys)

    steps_ys = [float(r.get("num_steps", 0)) for r in records if r.get("level") is not None]
    rho_steps = spearman(xs, steps_ys)

    print("\n=== H2: compute vs level ===")
    for lvl, stats in sorted(per_level.items()):
        print(
            f"level {lvl}: n={stats['n']:3d}  "
            f"mean_tokens={stats['mean_tokens']:7.1f}  "
            f"mean_steps={stats['mean_steps']:4.2f}  "
            f"acc={stats['accuracy']:.3f}"
        )
    print(f"Spearman(level, tokens) = {rho_tokens:.3f}")
    print(f"Spearman(level, steps)  = {rho_steps:.3f}")

    print("\n=== H3: action fractions per level ===")
    actions = sorted({k for s in per_level.values() for k in s["action_counts"].keys()})
    if actions:
        header = "level  " + "  ".join(f"{a:>10s}" for a in actions)
        print(header)
        for lvl, stats in sorted(per_level.items()):
            row = "  ".join(f"{stats['action_fractions'].get(a, 0.0):10.3f}" for a in actions)
            print(f"{lvl:5d}  {row}")
    else:
        print("(no action_counts present)")

    write_h2_csv(per_level, args.out / f"h2_{args.tag}_per_level.csv")
    write_h3_csv(per_level, args.out / f"h3_{args.tag}_action_dist_per_level.csv")
    plot_h2(per_level, args.out / f"h2_{args.tag}_tokens_vs_level.png")
    plot_h3(per_level, args.out / f"h3_{args.tag}_action_dist_per_level.png")

    summary = {
        "tag": args.tag,
        "n_total": len(records),
        "n_with_level": levels_present,
        "spearman_level_tokens": rho_tokens,
        "spearman_level_steps": rho_steps,
        "actions_observed": actions,
        "reduced_action_set": len(actions) < 3,
        "per_level": {str(k): v for k, v in per_level.items()},
    }
    (args.out / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\nArtifacts written to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
