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

export function TokensByLevelChart() {
  const data = h2BarCombined();
  return (
    <div className="rounded-xl border border-rule bg-bg-2 p-4">
      <h3 className="mb-0.5 text-sm font-semibold text-ink">
        Mean tokens per problem by difficulty level
      </h3>
      <p className="mb-3 text-xs leading-relaxed text-ink-3">
        Tokens tend to rise from L1 (easy) to L5 (hard) for both decode modes.
      </p>
      <div className="h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ece9e3" vertical={false} />
            <XAxis
              dataKey="level"
              tick={{ fontSize: 11 }}
              label={{
                value: "MATH-500 level (1 = easy, 5 = hard)",
                position: "insideBottom",
                offset: -2,
                style: { fill: "#6b6760", fontSize: 11 },
              }}
            />
            <YAxis
              tick={{ fontSize: 11 }}
              label={{
                value: "Mean tokens / problem",
                angle: -90,
                position: "insideLeft",
                style: { fill: "#6b6760", fontSize: 11 },
              }}
            />
            <Tooltip
              formatter={(v: number, name: string) => [
                `${v.toFixed(1)} tokens`,
                name === "+refine" ? "+refine" : "-refine",
              ]}
            />
            <Legend />
            <Bar name="+refine" dataKey="tokensWithRefine" fill="#2d5a3d" radius={[4, 4, 0, 0]} />
            <Bar name="-refine" dataKey="tokensNoRefine" fill="#b84a2a" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
