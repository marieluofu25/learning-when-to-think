import clsx from "clsx";
import type { MethodRow } from "../data/results";

type Props = { rows: MethodRow[] };

export function MetricsTable({ rows }: Props) {
  return (
    <div className="my-4 overflow-x-auto rounded-lg border border-rule">
      <table className="w-full min-w-[640px] border-collapse text-[13.5px]">
        <thead>
          <tr className="bg-bg-3 text-left text-[11px] font-semibold uppercase tracking-wide text-ink-3">
            <th className="border-b border-rule px-2.5 py-2">Method</th>
            <th className="border-b border-rule px-2.5 py-2">Correct / total</th>
            <th className="border-b border-rule px-2.5 py-2">Accuracy (95% CI)</th>
            <th className="border-b border-rule px-2.5 py-2">Avg tokens</th>
            <th className="border-b border-rule px-2.5 py-2">Cost / correct</th>
            <th className="border-b border-rule px-2.5 py-2">Pareto</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.method}
              className={clsx(
                "border-b border-rule2 last:border-b-0",
                r.highlight && "bg-[#f4faf6]",
                r.status === "dominated" && !r.highlight && "text-ink-3"
              )}
            >
              <td
                className={clsx(
                  "px-2.5 py-2 align-top text-ink-2",
                  r.highlight && "font-semibold text-ink"
                )}
              >
                {r.method}
              </td>
              <td className="px-2.5 py-2 font-mono text-[13px] text-ink">
                {r.correct} / {r.total}
              </td>
              <td className="px-2.5 py-2 font-mono text-[13px] text-ink">
                {r.acc.toFixed(3)} [{r.ciLow.toFixed(3)}, {r.ciHigh.toFixed(3)}]
              </td>
              <td className="px-2.5 py-2 font-mono text-[13px] text-ink">
                {r.avgTokens.toFixed(1)}
              </td>
              <td className="px-2.5 py-2 font-mono text-[13px] text-ink">
                {r.costPerCorrect.toFixed(1)}
              </td>
              <td className="px-2.5 py-2 align-top">
                <span
                  className={clsx(
                    "inline-block rounded px-2 py-0.5 font-mono text-[11px] font-semibold",
                    r.status === "pareto"
                      ? "bg-[#e1f5ee] text-[#085041]"
                      : "bg-bg-3 text-ink-3"
                  )}
                >
                  {r.status === "pareto" ? "non-dominated" : "dominated"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
