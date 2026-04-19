import type { ReactNode } from "react";
import { BlockMath } from "react-katex";

type Props = {
  /** LaTeX source rendered as a display-style block equation. */
  tex?: string;
  /** Fallback for legacy inline JSX formulas. Ignored if `tex` is set. */
  children?: ReactNode;
  /** Optional explainer shown beneath the equation. */
  sub?: ReactNode;
};

export function Formula({ tex, children, sub }: Props) {
  return (
    <div className="my-4 rounded-r-lg border border-rule border-l-[3px] border-l-accent bg-bg-2 px-5 py-4 text-ink">
      {tex !== undefined ? (
        <div className="overflow-x-auto text-[15.5px] leading-normal">
          <BlockMath math={tex} />
        </div>
      ) : (
        <div className="font-mono text-[14.5px] leading-relaxed">{children}</div>
      )}
      {sub ? (
        <div className="mt-2 font-sans text-[12.5px] leading-relaxed text-ink-3">
          {sub}
        </div>
      ) : null}
    </div>
  );
}
