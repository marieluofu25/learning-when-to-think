import { Section } from "../components/Section";

export function Limitations() {
  return (
    <Section id="limits" num="10" title="Limitations and next steps">
      <h3 className="mb-2 font-serif text-lg font-normal text-ink">Limits in this report</h3>
      <ul className="mb-6 list-disc space-y-1.5 pl-5 text-[14px] leading-relaxed text-ink-2 [&_b]:text-ink">
        <li>
          <b>Train set is tiny (n=32):</b> training accuracy went 0.59 &rarr; 0.49 between
          epochs, which hints at instability.
        </li>
        <li>
          <b>Eval set is small (n=100):</b> 95% CI near accuracy 0.5 is about &plusmn;10
          points. Most differences are inside this gap.
        </li>
        <li>
          <b>Reduced action space at train time:</b>{" "}
          <code className="rounded bg-bg-3 px-1 font-mono text-xs">allow_refine: false</code>
          , so H3 is really a 2-action test (continue vs terminate).
        </li>
        <li>
          <b>Vanilla GRPO uses the same small data,</b> so the H1 ablation inherits the
          same noise.
        </li>
        <li>
          <b>No external PRM, no AIME, no full MATH, no Pass@k baseline.</b> All out of
          scope for this proposal.
        </li>
      </ul>
      <h3 className="mb-2 font-serif text-lg font-normal text-ink">Next steps if we had more time</h3>
      <ul className="list-disc space-y-1.5 pl-5 text-[14px] leading-relaxed text-ink-2 [&_b]:text-ink">
        <li>
          Retrain with{" "}
          <code className="rounded bg-bg-3 px-1 font-mono text-xs">allow_refine: true</code>{" "}
          and a real refine reward signal so H3 is a true 3-action test.
        </li>
        <li>
          Scale train data from{" "}
          <code className="rounded bg-bg-3 px-1 font-mono text-xs">n=32</code> to several
          hundred MATH problems and run more epochs.
        </li>
        <li>
          Push eval to{" "}
          <code className="rounded bg-bg-3 px-1 font-mono text-xs">n=500</code> (full
          MATH-500) so CIs shrink to roughly &plusmn;4 points.
        </li>
        <li>Add self-consistency Pass@k as a stronger compute baseline for H1.</li>
      </ul>
    </Section>
  );
}
