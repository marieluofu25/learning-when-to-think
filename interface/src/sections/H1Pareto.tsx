import { Section } from "../components/Section";
import { MetricsTable } from "../components/MetricsTable";
import { Verdict } from "../components/Verdict";
import { ParetoChart } from "../components/charts/ParetoChart";
import { h1Methods } from "../data/results";

export function H1Pareto() {
  return (
    <Section id="h1" num="06" title="H1 — Better accuracy/efficiency trade-off">
      <p className="mb-3 text-[16.5px] leading-relaxed">
        <strong>Hypothesis.</strong> The adaptive policy with ALP + DeGRPO should give a
        better Pareto frontier (more accuracy per token) than CoT and vanilla GRPO.
      </p>
      <div className="mb-4">
        <ParetoChart />
      </div>
      <MetricsTable rows={h1Methods} />
      <Verdict tone="partial" tag="Partially supported">
        <b>The best adaptive run is DeGRPO without refine</b> (0.59 accuracy, 613
        tokens). It is the only adaptive point on the Pareto frontier and beats vanilla
        GRPO without refine (0.50). But its CI overlaps with CoT (0.53), so the gain
        over CoT is not statistically clean at{" "}
        <code className="rounded bg-bg-3 px-1 font-mono text-xs">n=100</code>. All{" "}
        <code className="font-mono text-xs">+refine</code> rows are dominated.
      </Verdict>
    </Section>
  );
}
