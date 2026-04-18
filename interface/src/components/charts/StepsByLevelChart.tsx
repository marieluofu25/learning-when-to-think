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
import { h2BarCombined } from "../../data/results";

export function StepsByLevelChart() {
  const data = h2BarCombined();
  return (
    <div className="rounded-xl border border-rule bg-bg-2 p-4">
      <h3 className="mb-0.5 text-sm font-semibold text-ink">
        Mean steps per problem by difficulty level
      </h3>
      <p className="mb-3 text-xs leading-relaxed text-ink-3">
        Step count stays almost flat (~2.6–3.0). The model uses longer text inside
        a step, not more steps.
      </p>
      <div className="h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ece9e3" vertical={false} />
            <XAxis dataKey="level" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 4]} tick={{ fontSize: 11 }} />
            <Tooltip
              formatter={(v: number, name: string) => [
                `${v.toFixed(2)} steps`,
                name === "+refine" ? "+refine" : "-refine",
              ]}
            />
            <Legend />
            <Bar name="+refine" dataKey="stepsWithRefine" fill="#4a8c62" radius={[4, 4, 0, 0]} />
            <Bar name="-refine" dataKey="stepsNoRefine" fill="#c18a2a" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
