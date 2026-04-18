import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { h1ScatterPoints } from "../../data/results";

type Point = {
  name: string;
  x: number;
  y: number;
  cost: number;
  pareto: boolean;
};

function ParetoTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: Point }[];
}) {
  if (!active || !payload?.[0]) return null;
  const p = payload[0].payload;
  return (
    <div className="rounded-md border border-rule bg-bg px-3 py-2 text-xs shadow-lg">
      <div className="font-semibold text-ink">{p.name}</div>
      <div className="mt-1 font-mono text-ink-2">
        acc={p.y.toFixed(3)}, tokens={p.x.toFixed(1)}
      </div>
      <div className="font-mono text-ink-2">
        cost/correct={p.cost.toFixed(1)}
      </div>
    </div>
  );
}

export function ParetoChart() {
  const all = h1ScatterPoints();
  const pareto = all.filter((d) => d.pareto);
  const dominated = all.filter((d) => !d.pareto);

  return (
    <div className="rounded-xl border border-rule bg-bg-2 p-4">
      <h3 className="mb-0.5 text-sm font-semibold text-ink">
        Accuracy vs avg tokens per problem (MATH-500, n=100)
      </h3>
      <p className="mb-3 text-xs leading-relaxed text-ink-3">
        Top-left is better (high accuracy, fewer tokens). Green = non-dominated
        (Pareto). Grey ring = dominated.
      </p>
      <div className="h-[320px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ece9e3" />
            <XAxis
              type="number"
              dataKey="x"
              name="tokens"
              domain={[0, "auto"]}
              label={{
                value: "Avg tokens per problem",
                position: "insideBottom",
                offset: -2,
                style: { fill: "#6b6760", fontSize: 11 },
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="acc"
              domain={[0, 0.75]}
              tickFormatter={(v) => v.toFixed(2)}
              label={{
                value: "Accuracy",
                angle: -90,
                position: "insideLeft",
                style: { fill: "#6b6760", fontSize: 11 },
              }}
            />
            <Tooltip content={<ParetoTooltip />} cursor={{ strokeDasharray: "3 3" }} />
            <Legend
              verticalAlign="bottom"
              height={28}
              formatter={(v) => <span className="text-xs text-ink-2">{v}</span>}
            />
            <Scatter
              name="Non-dominated (Pareto)"
              data={pareto}
              fill="#2d5a3d"
              shape="circle"
            />
            <Scatter
              name="Dominated"
              data={dominated}
              fill="rgba(168,163,154,0.12)"
              stroke="#6b6760"
              strokeWidth={1.8}
              shape={(props: { cx?: number; cy?: number }) => {
                const cx = props.cx ?? 0;
                const cy = props.cy ?? 0;
                return (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={6}
                    fill="rgba(168,163,154,0.12)"
                    stroke="#6b6760"
                    strokeWidth={1.8}
                  />
                );
              }}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
