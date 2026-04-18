import { Section } from "../components/Section";
import { Verdict } from "../components/Verdict";
import { ActionStackChart } from "../components/charts/ActionStackChart";
import { h3NoRefine, h3StackData, h3WithRefine } from "../data/results";

export function H3ActionMix() {
  return (
    <Section id="h3" num="08" title="H3 — Action mix changes with difficulty">
      <p className="mb-3 text-[16.5px] leading-relaxed">
        <strong>Hypothesis.</strong> Easy problems should get more{" "}
        <code className="rounded bg-bg-3 px-1 font-mono text-xs">terminate</code>. Hard
        problems should get more{" "}
        <code className="rounded bg-bg-3 px-1 font-mono text-xs">continue</code> and{" "}
        <code className="rounded bg-bg-3 px-1 font-mono text-xs">refine</code>.
      </p>
      <p className="mb-4">
        Note: the policy was trained with{" "}
        <code className="rounded bg-bg-3 px-1 font-mono text-xs">allow_refine: false</code>
        . So the cleanest test is the{" "}
        <code className="font-mono text-xs">+refine</code> decode (refine token allowed at
        test time) and the{" "}
        <code className="font-mono text-xs">-refine</code> decode (refine forced off).
      </p>
      <div className="mb-4 space-y-4">
        <ActionStackChart
          title="Action share by level — +refine decode"
          caption="Each bar adds up to 100%. Lower terminate share on harder levels would support H3."
          data={h3StackData(h3WithRefine)}
        />
        <div className="overflow-x-auto rounded-lg border border-rule">
          <table className="w-full border-collapse text-[13.5px]">
            <thead>
              <tr className="bg-bg-3 text-left text-[11px] font-semibold uppercase tracking-wide text-ink-3">
                <th className="border-b border-rule px-2 py-2">Level</th>
                <th className="border-b border-rule px-2 py-2">n</th>
                <th className="border-b border-rule px-2 py-2">frac continue</th>
                <th className="border-b border-rule px-2 py-2">frac refine</th>
                <th className="border-b border-rule px-2 py-2">frac terminate</th>
              </tr>
            </thead>
            <tbody>
              {h3WithRefine.map((r) => (
                <tr key={r.level} className="border-b border-rule2 last:border-b-0">
                  <td className="px-2 py-2">{r.level}</td>
                  <td className="px-2 py-2 font-mono">{r.n}</td>
                  <td className="px-2 py-2 font-mono">{r.fracContinue.toFixed(3)}</td>
                  <td className="px-2 py-2 font-mono">{r.fracRefine.toFixed(3)}</td>
                  <td className="px-2 py-2 font-mono">{r.fracTerminate.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <ActionStackChart
          title="Action share by level — -refine decode (refine forced off)"
          caption="Two-action regime: continue vs terminate. Refine is always 0 here."
          data={h3StackData(h3NoRefine)}
        />
        <div className="overflow-x-auto rounded-lg border border-rule">
          <table className="w-full border-collapse text-[13.5px]">
            <thead>
              <tr className="bg-bg-3 text-left text-[11px] font-semibold uppercase tracking-wide text-ink-3">
                <th className="border-b border-rule px-2 py-2">Level</th>
                <th className="border-b border-rule px-2 py-2">n</th>
                <th className="border-b border-rule px-2 py-2">frac continue</th>
                <th className="border-b border-rule px-2 py-2">frac refine</th>
                <th className="border-b border-rule px-2 py-2">frac terminate</th>
              </tr>
            </thead>
            <tbody>
              {h3NoRefine.map((r) => (
                <tr key={r.level} className="border-b border-rule2 last:border-b-0">
                  <td className="px-2 py-2">{r.level}</td>
                  <td className="px-2 py-2 font-mono">{r.n}</td>
                  <td className="px-2 py-2 font-mono">{r.fracContinue.toFixed(3)}</td>
                  <td className="px-2 py-2 font-mono">{r.fracRefine.toFixed(3)}</td>
                  <td className="px-2 py-2 font-mono">{r.fracTerminate.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <Verdict tone="partial" tag="Partially supported">
        <b>Terminate share is higher on level 1 than on level 5</b> in both decodes
        (0.138 vs 0.091 with refine; 0.161 vs 0.088 without). That matches H3 in spirit.
        But the trend is noisy (level 4 is an outlier with very few terminates) and
        refine is rarely picked because the model was never trained to use it. A clean
        3-action H3 needs a retrain with{" "}
        <code className="rounded bg-bg-3 px-1 font-mono text-xs">allow_refine: true</code>
        .
      </Verdict>
    </Section>
  );
}
