/** Numbers copied verbatim from docs/results_h1_h3.md */

export type ParetoStatus = "pareto" | "dominated";

export interface MethodRow {
  method: string;
  correct: number;
  total: number;
  acc: number;
  ciLow: number;
  ciHigh: number;
  avgTokens: number;
  costPerCorrect: number;
  status: ParetoStatus;
  highlight?: boolean;
}

export interface LevelRow {
  level: 1 | 2 | 3 | 4 | 5;
  n: number;
  meanTokens: number;
  meanSteps: number;
  acc: number;
}

export interface ActionRow {
  level: 1 | 2 | 3 | 4 | 5;
  n: number;
  fracContinue: number;
  fracRefine: number;
  fracTerminate: number;
}

export const h1Methods: MethodRow[] = [
  {
    method: "CoT baseline",
    correct: 53,
    total: 100,
    acc: 0.53,
    ciLow: 0.433,
    ciHigh: 0.625,
    avgTokens: 404.4,
    costPerCorrect: 763.0,
    status: "pareto",
  },
  {
    method: "Direct baseline",
    correct: 0,
    total: 100,
    acc: 0.0,
    ciLow: 0.0,
    ciHigh: 0.037,
    avgTokens: 64.0,
    costPerCorrect: 6400.0,
    status: "pareto",
  },
  {
    method: "Adaptive DeGRPO (+refine)",
    correct: 46,
    total: 100,
    acc: 0.46,
    ciLow: 0.366,
    ciHigh: 0.557,
    avgTokens: 754.4,
    costPerCorrect: 1640.1,
    status: "dominated",
  },
  {
    method: "Adaptive DeGRPO (-refine)",
    correct: 59,
    total: 100,
    acc: 0.59,
    ciLow: 0.492,
    ciHigh: 0.681,
    avgTokens: 613.1,
    costPerCorrect: 1039.2,
    status: "pareto",
    highlight: true,
  },
  {
    method: "Adaptive Vanilla (+refine)",
    correct: 41,
    total: 100,
    acc: 0.41,
    ciLow: 0.319,
    ciHigh: 0.508,
    avgTokens: 767.3,
    costPerCorrect: 1871.4,
    status: "dominated",
  },
  {
    method: "Adaptive Vanilla (-refine)",
    correct: 50,
    total: 100,
    acc: 0.5,
    ciLow: 0.404,
    ciHigh: 0.596,
    avgTokens: 594.1,
    costPerCorrect: 1188.3,
    status: "dominated",
  },
];

export const h2WithRefine: LevelRow[] = [
  { level: 1, n: 11, meanTokens: 594.4, meanSteps: 2.64, acc: 0.545 },
  { level: 2, n: 25, meanTokens: 601.7, meanSteps: 2.76, acc: 0.72 },
  { level: 3, n: 19, meanTokens: 672.8, meanSteps: 2.74, acc: 0.474 },
  { level: 4, n: 22, meanTokens: 794.2, meanSteps: 2.77, acc: 0.273 },
  { level: 5, n: 23, meanTokens: 1026.5, meanSteps: 2.87, acc: 0.304 },
];

export const h2NoRefine: LevelRow[] = [
  { level: 1, n: 11, meanTokens: 513.3, meanSteps: 2.82, acc: 0.909 },
  { level: 2, n: 25, meanTokens: 560.0, meanSteps: 2.88, acc: 0.76 },
  { level: 3, n: 19, meanTokens: 570.2, meanSteps: 2.74, acc: 0.789 },
  { level: 4, n: 22, meanTokens: 566.2, meanSteps: 2.95, acc: 0.5 },
  { level: 5, n: 23, meanTokens: 798.9, meanSteps: 2.96, acc: 0.174 },
];

export const h3WithRefine: ActionRow[] = [
  { level: 1, n: 11, fracContinue: 0.862, fracRefine: 0.0, fracTerminate: 0.138 },
  { level: 2, n: 25, fracContinue: 0.841, fracRefine: 0.029, fracTerminate: 0.13 },
  { level: 3, n: 19, fracContinue: 0.769, fracRefine: 0.096, fracTerminate: 0.135 },
  { level: 4, n: 22, fracContinue: 0.836, fracRefine: 0.066, fracTerminate: 0.098 },
  { level: 5, n: 23, fracContinue: 0.879, fracRefine: 0.03, fracTerminate: 0.091 },
];

export const h3NoRefine: ActionRow[] = [
  { level: 1, n: 11, fracContinue: 0.839, fracRefine: 0.0, fracTerminate: 0.161 },
  { level: 2, n: 25, fracContinue: 0.903, fracRefine: 0.0, fracTerminate: 0.097 },
  { level: 3, n: 19, fracContinue: 0.885, fracRefine: 0.0, fracTerminate: 0.115 },
  { level: 4, n: 22, fracContinue: 0.985, fracRefine: 0.0, fracTerminate: 0.015 },
  { level: 5, n: 23, fracContinue: 0.912, fracRefine: 0.0, fracTerminate: 0.088 },
];

export const spearman = {
  withRefine: { tokens: 0.36, steps: 0.138 },
  noRefine: { tokens: 0.288, steps: 0.149 },
} as const;

/** For Recharts scatter: x = tokens, y = accuracy */
export function h1ScatterPoints() {
  return h1Methods.map((m) => ({
    name: m.method,
    x: m.avgTokens,
    y: m.acc,
    cost: m.costPerCorrect,
    pareto: m.status === "pareto",
  }));
}

export function h2BarCombined() {
  return h2WithRefine.map((w, i) => ({
    level: `L${w.level}`,
    tokensWithRefine: w.meanTokens,
    tokensNoRefine: h2NoRefine[i]!.meanTokens,
    stepsWithRefine: w.meanSteps,
    stepsNoRefine: h2NoRefine[i]!.meanSteps,
  }));
}

export function h3StackData(rows: ActionRow[]) {
  return rows.map((r) => ({
    level: `L${r.level}`,
    continue: r.fracContinue,
    refine: r.fracRefine,
    terminate: r.fracTerminate,
  }));
}
