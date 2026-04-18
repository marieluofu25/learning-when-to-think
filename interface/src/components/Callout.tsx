import type { ReactNode } from "react";
import clsx from "clsx";

type Variant = "info" | "warn" | "note";

const styles: Record<Variant, string> = {
  info: "border-accent bg-bg-2",
  warn: "border-coral bg-bg-2",
  note: "border-amber bg-bg-2",
};

type Props = { variant?: Variant; children: ReactNode };

export function Callout({ variant = "info", children }: Props) {
  return (
    <div
      className={clsx(
        "my-4 rounded-r-lg border-l-[3px] border border-rule px-4 py-3 text-[14.5px] leading-relaxed text-ink-2",
        styles[variant]
      )}
    >
      {children}
    </div>
  );
}
