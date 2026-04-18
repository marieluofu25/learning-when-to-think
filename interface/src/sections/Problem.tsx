import { Section } from "../components/Section";
import { Callout } from "../components/Callout";

export function Problem() {
  return (
    <Section id="problem" num="01" title="The problem">
      <p className="mb-3 text-[16.5px] leading-relaxed">
        Large language models think for the same amount of time on every question.
        They use the same number of tokens for &quot;What is 2 + 2?&quot; and for a
        hard olympiad problem.
      </p>
      <p className="mb-3">
        This is a waste. Easy questions get too many tokens. Hard questions sometimes
        stop too early and get the wrong answer. We want a model that{" "}
        <strong>chooses how much to think</strong> based on the question.
      </p>
      <Callout variant="info">
        <strong>Goal in one line.</strong> Train a small policy on top of an LLM so it
        can decide: keep thinking, fix a mistake, or stop and answer.
      </Callout>
    </Section>
  );
}
