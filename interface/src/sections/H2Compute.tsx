import { Section } from "../components/Section";
import { Verdict } from "../components/Verdict";
import { TokensByLevelChart } from "../components/charts/TokensByLevelChart";
import { StepsByLevelChart } from "../components/charts/StepsByLevelChart";
import { h2NoRefine, h2WithRefine, spearman } from "../data/results";

export function H2Compute() {
  return (
    <Section id="h2" num="07" title="H2 — More compute on harder problems">
      <p className="mb-3 text-[16.5px] leading-relaxed">
        <strong>Hypothesis.</strong> A learned policy should give more tokens (and more
        steps) to harder problems.
      </p>
      <div className="mb-3 flex flex-wrap gap-2">
        <span className="rounded-md border border-rule bg-bg-2 px-2.5 py-1 font-mono text-xs text-ink-2">
          +refine: Spearman(level, tokens) ={" "}
          <b className="text-accent">{spearman.withRefine.tokens.toFixed(3)}</b>
        </span>
        <span className="rounded-md border border-rule bg-bg-2 px-2.5 py-1 font-mono text-xs text-ink-2">
          +refine: Spearman(level, steps) ={" "}
          <b className="text-accent">{spearman.withRefine.steps.toFixed(3)}</b>
        </span>
        <span className="rounded-md border border-rule bg-bg-2 px-2.5 py-1 font-mono text-xs text-ink-2">
          -refine: Spearman(level, tokens) ={" "}
          <b className="text-accent">{spearman.noRefine.tokens.toFixed(3)}</b>
        </span>
        <span className="rounded-md border border-rule bg-bg-2 px-2.5 py-1 font-mono text-xs text-ink-2">
          -refine: Spearman(level, steps) ={" "}
          <b className="text-accent">{spearman.noRefine.steps.toFixed(3)}</b>
        </span>
      </div>
      <div className="mb-4 space-y-4">
        <TokensByLevelChart />
        <StepsByLevelChart />
      </div>
      <h3 className="mb-2 font-serif text-lg font-normal text-ink">
        Per-level numbers (Adaptive DeGRPO)
      </h3>
      <div className="mb-4 overflow-x-auto rounded-lg border border-rule">
        <table className="w-full min-w-[720px] border-collapse text-[13.5px]">
          <thead>
            <tr className="bg-bg-3 text-left text-[11px] font-semibold uppercase tracking-wide text-ink-3">
              <th className="border-b border-rule px-2 py-2">Level</th>
              <th className="border-b border-rule px-2 py-2">n</th>
              <th className="border-b border-rule px-2 py-2">+refine: tokens</th>
              <th className="border-b border-rule px-2 py-2">+refine: steps</th>
              <th className="border-b border-rule px-2 py-2">+refine: acc</th>
              <th className="border-b border-rule px-2 py-2">-refine: tokens</th>
              <th className="border-b border-rule px-2 py-2">-refine: steps</th>
              <th className="border-b border-rule px-2 py-2">-refine: acc</th>
            </tr>
          </thead>
          <tbody>
            {h2WithRefine.map((w, i) => {
              const n = h2NoRefine[i]!;
              return (
                <tr key={w.level} className="border-b border-rule2 last:border-b-0">
                  <td className="px-2 py-2 text-ink">{w.level}</td>
                  <td className="px-2 py-2 font-mono text-ink">{w.n}</td>
                  <td className="px-2 py-2 font-mono text-ink">{w.meanTokens.toFixed(1)}</td>
                  <td className="px-2 py-2 font-mono text-ink">{w.meanSteps.toFixed(2)}</td>
                  <td className="px-2 py-2 font-mono text-ink">{w.acc.toFixed(3)}</td>
                  <td className="px-2 py-2 font-mono text-ink">{n.meanTokens.toFixed(1)}</td>
                  <td className="px-2 py-2 font-mono text-ink">{n.meanSteps.toFixed(2)}</td>
                  <td className="px-2 py-2 font-mono text-ink">{n.acc.toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Verdict tone="weak" tag="Weakly supported">
        <b>Yes for tokens, no for steps.</b> Mean tokens rise clearly with difficulty
        (594 &rarr; 1027 for +refine). But step count is almost flat. So the model spends
        more <i>inside</i> a step on hard problems, not more steps. Spearman is positive
        but only moderate (0.29&ndash;0.36).
      </Verdict>
    </Section>
  );
}
