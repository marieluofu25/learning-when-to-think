import clsx from "clsx";

type Props = {
  title: string;
  verdict: string;
  tone: "partial" | "weak" | "win";
  desc: string;
};

const vColor: Record<Props["tone"], string> = {
  partial: "text-[#633806]",
  weak: "text-[#712b13]",
  win: "text-[#085041]",
};

export function ScoreCard({ title, verdict, tone, desc }: Props) {
  return (
    <div className="rounded-lg border border-rule bg-bg-2 p-4 text-center">
      <div className="mb-1 font-mono text-[11px] font-semibold uppercase tracking-wide text-ink-3">
        {title}
      </div>
      <div
        className={clsx(
          "mb-1 font-serif text-[22px] font-normal",
          vColor[tone]
        )}
      >
        {verdict}
      </div>
      <p className="text-xs leading-snug text-ink-3">{desc}</p>
    </div>
  );
}
