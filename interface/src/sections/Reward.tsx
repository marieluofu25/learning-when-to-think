import { Section } from "../components/Section";
import { Formula } from "../components/Formula";

export function Reward() {
  return (
    <Section id="reward" num="03" title="Reward: ALP (Adaptive Length Penalty)">
      <p className="mb-3">
        The reward gives a point for a correct answer and takes points away for a long
        answer. The penalty is bigger when the question is easy.
      </p>
      <Formula
        tex={String.raw`R_k \;=\; r_{\mathrm{acc},k} \;-\; \beta \cdot \max\bigl(0,\, \mathrm{SR}(q)\bigr) \cdot \frac{n_{\mathrm{tokens},k}}{L_{\max}}`}
        sub={
          <>
            <b>
              r<sub>acc,k</sub>
            </b>
            : 1 if the answer is correct, else 0. <b>SR(q)</b>: how often the group
            solves question q (a proxy for &quot;easy&quot;).{" "}
            <b>
              n<sub>tokens,k</sub>
            </b>
            : how long the answer is.{" "}
            <b>
              &beta;, L<sub>max</sub>
            </b>
            : fixed scale factors.
          </>
        }
      />
      <p>
        In plain words: <strong>easy + long = big penalty</strong>.{" "}
        <strong>hard + long = small penalty</strong>. So the model learns to spend
        tokens where they help.
      </p>
    </Section>
  );
}
