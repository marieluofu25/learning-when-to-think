import clsx from "clsx";

type Action = "continue" | "refine" | "terminate";

const titles: Record<Action, string> = {
  continue: "continue",
  refine: "refine",
  terminate: "terminate",
};

const titleColors: Record<Action, string> = {
  continue: "text-[#085041]",
  refine: "text-[#633806]",
  terminate: "text-[#3c3489]",
};

type Props = { action: Action; children: string };

export function ActionCard({ action, children }: Props) {
  return (
    <div className="rounded-lg border border-rule bg-bg-2 p-4">
      <div
        className={clsx(
          "mb-1.5 font-mono text-xs font-semibold uppercase tracking-wide",
          titleColors[action]
        )}
      >
        {titles[action]}
      </div>
      <p className="text-[13px] leading-relaxed text-ink-3">{children}</p>
    </div>
  );
}
