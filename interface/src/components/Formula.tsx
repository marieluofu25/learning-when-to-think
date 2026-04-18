import type { ReactNode } from "react";

type Props = { children: ReactNode; sub?: ReactNode };

export function Formula({ children, sub }: Props) {
  return (
    <div className="my-4 rounded-r-lg border border-rule border-l-[3px] border-l-accent bg-bg-2 px-5 py-4 font-mono text-[14.5px] leading-relaxed text-ink">
      <div>{children}</div>
      {sub ? (
        <div className="mt-2 font-sans text-[12.5px] leading-relaxed text-ink-3">
          {sub}
        </div>
      ) : null}
    </div>
  );
}
