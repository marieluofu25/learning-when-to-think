import { Section } from "../components/Section";
import { Callout } from "../components/Callout";

export function Setup() {
  return (
    <Section id="setup" num="05" title="Experimental setup">
      <p className="mb-3">
        We test on <strong>MATH-500</strong>, a hard high-school competition math
        benchmark with 5 difficulty levels (1 = easiest, 5 = hardest) and several
        subjects.
      </p>
      <ul className="mb-4 list-disc space-y-1.5 pl-5 text-[14px] leading-relaxed text-ink-2 [&_b]:text-ink">
        <li>
          <b>Eval size:</b>{" "}
          <code className="rounded bg-bg-3 px-1 font-mono text-xs">n = 100</code> per
          checkpoint (CHPC jobs are capped at 2 hours).
        </li>
        <li>
          <b>Resume:</b> per-problem JSONL, so a killed job restarts from the last
          solved item.
        </li>
        <li>
          <b>Methods compared:</b> CoT baseline, Direct (no reasoning), Adaptive DeGRPO
          with and without <code className="font-mono text-xs">refine</code>, Adaptive
          Vanilla GRPO with and without <code className="font-mono text-xs">refine</code>
          .
        </li>
        <li>
          <b>Pipeline:</b> see{" "}
          <a
            className="text-accent underline decoration-accent/30 hover:decoration-accent"
            href="../docs/results_h1_h3.md"
          >
            docs/results_h1_h3.md
          </a>{" "}
          for full commands.
        </li>
      </ul>
      <Callout variant="warn">
        <strong>Statistical caveat.</strong> At{" "}
        <code className="rounded bg-bg-3 px-1 font-mono text-xs">n=100</code> the 95%
        CI near accuracy 0.5 is about &plusmn;10 points. So differences smaller than
        that are not statistically clear.
      </Callout>
    </Section>
  );
}
