import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Row = { level: string; continue: number; refine: number; terminate: number };

type Props = { title: string; caption: string; data: Row[] };

function pctTooltip(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function ActionStackChart({ title, caption, data }: Props) {
  return (
    <div className="rounded-xl border border-rule bg-bg-2 p-4">
      <h3 className="mb-0.5 text-sm font-semibold text-ink">{title}</h3>
      <p className="mb-3 text-xs leading-relaxed text-ink-3">{caption}</p>
      <div className="h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ece9e3" vertical={false} />
            <XAxis dataKey="level" tick={{ fontSize: 11 }} />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
              tick={{ fontSize: 11 }}
            />
            <Tooltip formatter={(v: number) => pctTooltip(v)} />
            <Legend />
            <Bar dataKey="continue" name="continue" stackId="a" fill="#4a8c62" />
            <Bar dataKey="refine" name="refine" stackId="a" fill="#c18a2a" />
            <Bar dataKey="terminate" name="terminate" stackId="a" fill="#b84a2a" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
