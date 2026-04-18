import clsx from "clsx";
import type { ReactNode } from "react";

type Tone = "partial" | "weak" | "win" | "fail";

const tagStyles: Record<Tone, string> = {
  partial: "bg-[#faeeda] text-[#633806]",
  weak: "bg-[#faece7] text-[#712b13]",
  win: "bg-[#e1f5ee] text-[#085041]",
  fail: "bg-bg-3 text-ink-3",
};

type Props = { tone: Tone; tag: string; children: ReactNode };

export function Verdict({ tone, tag, children }: Props) {
  return (
    <div className="my-4 flex gap-2.5 rounded-lg border border-rule bg-bg-2 p-3.5">
      <span
        className={clsx(
          "mt-0.5 shrink-0 rounded px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wide",
          tagStyles[tone]
        )}
      >
        {tag}
      </span>
      <div className="text-[13.5px] leading-relaxed text-ink-2 [&_b]:text-ink">
        {children}
      </div>
    </div>
  );
}
