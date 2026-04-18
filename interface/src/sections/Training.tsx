import { Section } from "../components/Section";
import { Formula } from "../components/Formula";
import { KvCard } from "../components/KvCard";
import { Callout } from "../components/Callout";

export function Training() {
  return (
    <Section id="train" num="04" title="Training: GRPO + DeGRPO + LoRA">
      <p className="mb-3">
        We use <strong>GRPO</strong> (Group Relative Policy Optimization). For each
        question we sample a small group of answers and use the group as a baseline.
        No separate critic model needed.
      </p>
      <p className="mb-3">
        <strong>DeGRPO</strong> is our small change. The control tokens (action
        choices) get a higher gradient weight than the normal text tokens. This stops
        the action signal from being washed out by the long text.
      </p>
      <Formula
        sub={
          <>
            Default weights: w<sub>ctrl</sub> = 2.0, w<sub>resp</sub> = 1.0. KL term
            keeps the policy close to the base model.
          </>
        }
      >
        L = &minus;A<sub>k</sub> ({" "}
        <span className="text-accent">w<sub>ctrl</sub></span> &middot; log p
        <sub>ctrl</sub> + <span className="text-accent">w<sub>resp</sub></span>{" "}
        &middot; log p<sub>resp</sub> ) +{" "}
        <span className="text-accent">&lambda;<sub>KL</sub></span> &middot; KL
      </Formula>
      <div className="mb-4 grid gap-2.5 sm:grid-cols-2">
        <KvCard label="Base model">Qwen2.5-Math-7B-Instruct</KvCard>
        <KvCard label="Adapter">
          LoRA, <code className="rounded bg-bg-3 px-1 font-mono text-xs">r=16</code>,{" "}
          <code className="rounded bg-bg-3 px-1 font-mono text-xs">&alpha;=32</code>{" "}
          on attention + MLP
        </KvCard>
        <KvCard label="Algorithm">GRPO + ALP reward + DeGRPO weighting</KvCard>
        <KvCard label="Train data">
          MATH-500 train subset,{" "}
          <code className="rounded bg-bg-3 px-1 font-mono text-xs">n=32</code>, 2 epochs
        </KvCard>
      </div>
      <Callout variant="note">
        <strong>Important note for H3.</strong> The training config used{" "}
        <code className="rounded bg-bg-3 px-1 font-mono text-xs">allow_refine: false</code>
        , so the policy only really learned <code className="font-mono text-xs">continue</code>{" "}
        and <code className="font-mono text-xs">terminate</code>. The{" "}
        <code className="font-mono text-xs">refine</code> action can still appear at
        decode time, but the model was not trained to use it well.
      </Callout>
    </Section>
  );
}
