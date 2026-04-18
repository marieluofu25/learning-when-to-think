import type { ReactNode } from "react";

type Props = { label: string; children: ReactNode };

export function KvCard({ label, children }: Props) {
  return (
    <div className="rounded-lg border border-rule bg-bg-2 p-3.5">
      <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-3">
        {label}
      </div>
      <div className="text-[13.5px] leading-relaxed text-ink">{children}</div>
    </div>
  );
}
