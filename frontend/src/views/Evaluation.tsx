import { BarChart3, Filter, TrendingUp } from "lucide-react";
import { useState } from "react";
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

interface MetricResult {
  metric: string;
  dense: number;
  hybrid: number;
  hybrid_rerank: number;
  dense_ci?: [number, number];
  hybrid_ci?: [number, number];
  hybrid_rerank_ci?: [number, number];
}

interface LatencyStage {
  stage: string;
  p50: number;
  p90: number;
  p95: number;
}

interface CategoryResult {
  category: string;
  dense: number;
  hybrid: number;
  hybrid_rerank: number;
}

interface WinLoss {
  question: string;
  category: string;
  dense_score: number;
  hybrid_score: number;
  hybrid_rerank_score: number;
  winner: string;
}

const DEMO_METRICS: MetricResult[] = [
  {
    metric: "P@5",
    dense: 0.62,
    hybrid: 0.71,
    hybrid_rerank: 0.78,
  },
  {
    metric: "R@5",
    dense: 0.58,
    hybrid: 0.67,
    hybrid_rerank: 0.74,
  },
  {
    metric: "MRR",
    dense: 0.65,
    hybrid: 0.73,
    hybrid_rerank: 0.81,
  },
  {
    metric: "NDCG@5",
    dense: 0.6,
    hybrid: 0.69,
    hybrid_rerank: 0.77,
  },
  {
    metric: "Faithfulness",
    dense: 0.82,
    hybrid: 0.85,
    hybrid_rerank: 0.88,
  },
  {
    metric: "Citation Acc.",
    dense: 0.75,
    hybrid: 0.8,
    hybrid_rerank: 0.86,
  },
];

const DEMO_LATENCY: LatencyStage[] = [
  { stage: "Dense", p50: 45, p90: 78, p95: 95 },
  { stage: "BM25", p50: 12, p90: 22, p95: 28 },
  { stage: "RRF", p50: 3, p90: 5, p95: 7 },
  { stage: "Rerank", p50: 85, p90: 120, p95: 145 },
  { stage: "Generation", p50: 380, p90: 520, p95: 610 },
  { stage: "Verification", p50: 340, p90: 480, p95: 560 },
];

const DEMO_CATEGORIES: CategoryResult[] = [
  { category: "exact-identifier", dense: 0.72, hybrid: 0.82, hybrid_rerank: 0.89 },
  { category: "factual-lookup", dense: 0.68, hybrid: 0.75, hybrid_rerank: 0.83 },
  { category: "multi-hop", dense: 0.45, hybrid: 0.58, hybrid_rerank: 0.67 },
  { category: "unanswerable", dense: 0.55, hybrid: 0.6, hybrid_rerank: 0.62 },
  { category: "ambiguous", dense: 0.5, hybrid: 0.56, hybrid_rerank: 0.64 },
];

const DEMO_WIN_LOSS: WinLoss[] = [
  {
    question: "What is the default chunk size for fixed chunking?",
    category: "exact-identifier",
    dense_score: 0.8,
    hybrid_score: 0.9,
    hybrid_rerank_score: 0.95,
    winner: "hybrid_rerank",
  },
  {
    question: "How does RRF fusion handle alpha weighting?",
    category: "factual-lookup",
    dense_score: 0.6,
    hybrid_score: 0.85,
    hybrid_rerank_score: 0.9,
    winner: "hybrid_rerank",
  },
  {
    question: "What model is used for reranking?",
    category: "exact-identifier",
    dense_score: 0.95,
    hybrid_score: 0.9,
    hybrid_rerank_score: 0.95,
    winner: "dense",
  },
];

const COLORS = {
  dense: "#94a3b8",
  hybrid: "#6366f1",
  hybrid_rerank: "#22c55e",
};

function ExperimentCard({
  name,
  color,
  mrr,
  ndcg,
  faith,
}: {
  name: string;
  color: string;
  mrr: number;
  ndcg: number;
  faith: number;
}) {
  return (
    <div className="bg-[var(--color-bg-secondary)] rounded-xl p-5 border border-[var(--color-border)]">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
        <h4 className="font-semibold text-sm">{name}</h4>
      </div>
      <div className="grid grid-cols-3 gap-3">
        <div>
          <p className="text-xs text-[var(--color-text-tertiary)]">MRR</p>
          <p className="text-lg font-semibold tabular-nums">{mrr.toFixed(3)}</p>
        </div>
        <div>
          <p className="text-xs text-[var(--color-text-tertiary)]">NDCG@5</p>
          <p className="text-lg font-semibold tabular-nums">{ndcg.toFixed(3)}</p>
        </div>
        <div>
          <p className="text-xs text-[var(--color-text-tertiary)]">Faith.</p>
          <p className="text-lg font-semibold tabular-nums">{faith.toFixed(3)}</p>
        </div>
      </div>
    </div>
  );
}

export default function Evaluation() {
  const [categoryFilter, setCategoryFilter] = useState<string>("all");

  const categories = ["all", ...DEMO_CATEGORIES.map((c) => c.category)];
  const filteredWinLoss =
    categoryFilter === "all"
      ? DEMO_WIN_LOSS
      : DEMO_WIN_LOSS.filter((w) => w.category === categoryFilter);

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold">Evaluation Dashboard</h2>
        <div className="flex items-center gap-2 text-xs text-[var(--color-text-tertiary)]">
          <TrendingUp size={14} />
          Experiment 4 Results (placeholder data)
        </div>
      </div>

      {/* Experiment summary cards */}
      <div className="grid grid-cols-3 gap-4">
        <ExperimentCard
          name="Dense Only"
          color={COLORS.dense}
          mrr={0.65}
          ndcg={0.6}
          faith={0.82}
        />
        <ExperimentCard
          name="Hybrid"
          color={COLORS.hybrid}
          mrr={0.73}
          ndcg={0.69}
          faith={0.85}
        />
        <ExperimentCard
          name="Hybrid + Reranker"
          color={COLORS.hybrid_rerank}
          mrr={0.81}
          ndcg={0.77}
          faith={0.88}
        />
      </div>

      {/* Metrics bar chart */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl p-5 border border-[var(--color-border)]">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <BarChart3 size={18} />
          Metrics Comparison
        </h3>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={DEMO_METRICS} barGap={2} barCategoryGap="20%">
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis
              dataKey="metric"
              tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-bg)",
                border: "1px solid var(--color-border)",
                borderRadius: "8px",
                fontSize: "12px",
              }}
            />
            <Legend wrapperStyle={{ fontSize: "12px" }} />
            <Bar dataKey="dense" name="Dense" fill={COLORS.dense} radius={[4, 4, 0, 0]} />
            <Bar
              dataKey="hybrid"
              name="Hybrid"
              fill={COLORS.hybrid}
              radius={[4, 4, 0, 0]}
            />
            <Bar
              dataKey="hybrid_rerank"
              name="Hybrid+Reranker"
              fill={COLORS.hybrid_rerank}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Latency waterfall */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl p-5 border border-[var(--color-border)]">
        <h3 className="font-semibold mb-4">Latency Waterfall (ms)</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart
            data={DEMO_LATENCY}
            layout="vertical"
            barGap={2}
            barCategoryGap="20%"
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--color-border)"
              horizontal={false}
            />
            <XAxis
              type="number"
              tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }}
            />
            <YAxis
              type="category"
              dataKey="stage"
              width={80}
              tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-bg)",
                border: "1px solid var(--color-border)",
                borderRadius: "8px",
                fontSize: "12px",
              }}
            />
            <Legend wrapperStyle={{ fontSize: "12px" }} />
            <Bar dataKey="p50" name="p50" fill="#6366f1" radius={[0, 4, 4, 0]} />
            <Bar dataKey="p90" name="p90" fill="#a78bfa" radius={[0, 4, 4, 0]} />
            <Bar dataKey="p95" name="p95" fill="#c4b5fd" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Per-category breakdown */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl p-5 border border-[var(--color-border)]">
        <h3 className="font-semibold mb-4">Per-Category Breakdown (MRR)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-text-secondary)]">
                <th className="px-4 py-2">Category</th>
                <th className="px-4 py-2">Dense</th>
                <th className="px-4 py-2">Hybrid</th>
                <th className="px-4 py-2">Hybrid+Reranker</th>
                <th className="px-4 py-2">Best</th>
              </tr>
            </thead>
            <tbody>
              {DEMO_CATEGORIES.map((c) => {
                const best = Math.max(c.dense, c.hybrid, c.hybrid_rerank);
                const winner =
                  best === c.hybrid_rerank
                    ? "H+R"
                    : best === c.hybrid
                      ? "H"
                      : "D";
                return (
                  <tr
                    key={c.category}
                    className="border-t border-[var(--color-border)]"
                  >
                    <td className="px-4 py-2 font-medium">{c.category}</td>
                    <td className="px-4 py-2 tabular-nums">
                      {c.dense.toFixed(3)}
                    </td>
                    <td className="px-4 py-2 tabular-nums">
                      {c.hybrid.toFixed(3)}
                    </td>
                    <td className="px-4 py-2 tabular-nums">
                      {c.hybrid_rerank.toFixed(3)}
                    </td>
                    <td className="px-4 py-2">
                      <span className="px-2 py-0.5 bg-[var(--color-success)]/10 text-[var(--color-success)] text-xs font-semibold rounded">
                        {winner}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Win/loss table */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl p-5 border border-[var(--color-border)]">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold">Question-Level Results</h3>
          <div className="flex items-center gap-2">
            <Filter size={14} className="text-[var(--color-text-tertiary)]" />
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded-lg px-2 py-1 text-xs"
            >
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c === "all" ? "All Categories" : c}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-text-secondary)]">
                <th className="px-4 py-2">Question</th>
                <th className="px-4 py-2">Category</th>
                <th className="px-4 py-2">Dense</th>
                <th className="px-4 py-2">Hybrid</th>
                <th className="px-4 py-2">H+Reranker</th>
                <th className="px-4 py-2">Winner</th>
              </tr>
            </thead>
            <tbody>
              {filteredWinLoss.map((w, i) => (
                <tr
                  key={i}
                  className="border-t border-[var(--color-border)]"
                >
                  <td className="px-4 py-2 max-w-xs truncate">
                    {w.question}
                  </td>
                  <td className="px-4 py-2 text-xs">{w.category}</td>
                  <td className="px-4 py-2 tabular-nums">
                    {w.dense_score.toFixed(3)}
                  </td>
                  <td className="px-4 py-2 tabular-nums">
                    {w.hybrid_score.toFixed(3)}
                  </td>
                  <td className="px-4 py-2 tabular-nums">
                    {w.hybrid_rerank_score.toFixed(3)}
                  </td>
                  <td className="px-4 py-2">
                    <span
                      className="px-2 py-0.5 text-xs font-semibold rounded"
                      style={{
                        backgroundColor: `${COLORS[w.winner as keyof typeof COLORS]}20`,
                        color: COLORS[w.winner as keyof typeof COLORS],
                      }}
                    >
                      {w.winner === "hybrid_rerank"
                        ? "H+R"
                        : w.winner === "hybrid"
                          ? "H"
                          : "D"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
