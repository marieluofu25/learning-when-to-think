import type { ReactNode } from "react";
import clsx from "clsx";

type Props = {
  id: string;
  num: string;
  title: string;
  children: ReactNode;
  className?: string;
};

export function Section({ id, num, title, children, className }: Props) {
  return (
    <section
      id={id}
      className={clsx("mb-14 scroll-mt-[4.5rem] md:scroll-mt-20", className)}
    >
      <div className="mb-5 flex items-center gap-3 border-b border-rule pb-3">
        <span className="rounded bg-accent-3 px-2 py-0.5 font-mono text-[11px] font-medium uppercase tracking-wide text-accent">
          {num}
        </span>
        <h2 className="font-serif text-2xl font-normal text-ink md:text-[26px]">
          {title}
        </h2>
      </div>
      <div className="max-w-none text-[15.5px] leading-relaxed text-ink-2">
        {children}
      </div>
    </section>
  );
}
