import { Section } from "../components/Section";
import { ActionCard } from "../components/ActionCard";

export function Idea() {
  return (
    <Section id="idea" num="02" title="The idea: three actions">
      <p className="mb-3">
        We treat reasoning as a small game (an MDP). At every step the model picks one
        of three actions:
      </p>
      <div className="mb-4 grid gap-2.5 sm:grid-cols-3">
        <ActionCard action="continue">
          Keep writing the next reasoning step. Use this when the answer is not ready
          yet.
        </ActionCard>
        <ActionCard action="refine">
          Stop, look back, and fix the last steps. Use this when something feels wrong.
        </ActionCard>
        <ActionCard action="terminate">
          Stop reasoning and write the final answer. Use this when the answer is clear.
        </ActionCard>
      </div>
      <p>
        The model emits a small <em>control token</em> for the action, then writes the
        matching text. This makes the action choice trainable with normal RL.
      </p>
    </Section>
  );
}
