import { Section } from "../components/Section";
import { ScoreCard } from "../components/ScoreCard";

export function Conclusion() {
  return (
    <Section id="conclusion" num="09" title="What worked, what did not">
      <div className="mb-6 grid gap-2.5 sm:grid-cols-3">
        <ScoreCard
          title="H1 · Pareto"
          verdict="Partial"
          tone="partial"
          desc="DeGRPO -refine is non-dominated. CIs vs CoT overlap."
        />
        <ScoreCard
          title="H2 · Adaptive compute"
          verdict="Weak"
          tone="weak"
          desc="Tokens grow with level. Steps are flat."
        />
        <ScoreCard
          title="H3 · Action mix"
          verdict="Partial"
          tone="partial"
          desc="More terminate on easy. Refine never really learned."
        />
      </div>
      <p className="mb-3">
        <strong>What worked.</strong> DeGRPO without refine is the best adaptive setting.
        It beats vanilla GRPO without refine at almost the same token cost (0.59 vs 0.50
        accuracy, ~613 vs ~594 tokens). The model also spends more tokens on harder
        levels, which is the main behavior we wanted.
      </p>
      <p className="mb-3">
        <strong>What did not work.</strong> The{" "}
        <code className="rounded bg-bg-3 px-1 font-mono text-xs">refine</code> action did
        not help. All{" "}
        <code className="font-mono text-xs">+refine</code> runs are dominated. The main
        reason is simple: training had{" "}
        <code className="rounded bg-bg-3 px-1 font-mono text-xs">allow_refine: false</code>
        , so the model was never rewarded for using refine well. Step count also stays
        flat, so &quot;thinking longer&quot; here means &quot;longer single steps&quot;,
        not &quot;more steps&quot;.
      </p>
      <p>
        <strong>Headline answer.</strong> The pipeline does push the policy in the right
        direction (more tokens on hard, more terminate on easy, DeGRPO &gt; vanilla). But
        the size of the win is small and not statistically clean at{" "}
        <code className="rounded bg-bg-3 px-1 font-mono text-xs">n=100</code>. We see a{" "}
        <i>signal</i>, not a <i>proof</i>.
      </p>
    </Section>
  );
}
