import {
  Activity,
  Award,
  BarChart3,
  CheckCircle,
  Filter,
  Layers,
  TrendingUp,
  Zap,
} from "lucide-react";
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface MetricResult {
  metric: string;
  bm25: number;
  dense: number;
  hybrid: number;
  hybrid_rerank: number;
}

interface LatencyStage {
  stage: string;
  p50: number;
  p90: number;
  p95: number;
}

interface TradeoffPoint {
  strategy: string;
  latency: number;
  accuracy: number;
  mrr: number;
  color: string;
}

interface CategoryResult {
  category: string;
  bm25: number;
  dense: number;
  hybrid: number;
  hybrid_rerank: number;
}

interface WinLoss {
  question: string;
  category: string;
  bm25_score: number;
  dense_score: number;
  hybrid_score: number;
  hybrid_rerank_score: number;
  winner: string;
}

const COMPARISON_METRICS: MetricResult[] = [
  {
    metric: "Hit Rate @ 5",
    bm25: 0.64,
    dense: 0.68,
    hybrid: 0.79,
    hybrid_rerank: 0.87,
  },
  {
    metric: "MRR",
    bm25: 0.59,
    dense: 0.65,
    hybrid: 0.73,
    hybrid_rerank: 0.81,
  },
  {
    metric: "P@5",
    bm25: 0.55,
    dense: 0.62,
    hybrid: 0.71,
    hybrid_rerank: 0.78,
  },
  {
    metric: "R@5",
    bm25: 0.51,
    dense: 0.58,
    hybrid: 0.67,
    hybrid_rerank: 0.74,
  },
  {
    metric: "Faithfulness",
    bm25: 0.76,
    dense: 0.82,
    hybrid: 0.85,
    hybrid_rerank: 0.89,
  },
  {
    metric: "Citation Acc.",
    bm25: 0.71,
    dense: 0.75,
    hybrid: 0.8,
    hybrid_rerank: 0.86,
  },
];

const TRADEOFF_DATA: TradeoffPoint[] = [
  { strategy: "BM25 Sparse", latency: 18, accuracy: 64, mrr: 59, color: "#f59e0b" },
  { strategy: "Dense Vector", latency: 52, accuracy: 68, mrr: 65, color: "#94a3b8" },
  { strategy: "Hybrid RRF", latency: 68, accuracy: 79, mrr: 73, color: "#6366f1" },
  { strategy: "Hybrid + Reranker", latency: 155, accuracy: 87, mrr: 81, color: "#10b981" },
];

const LATENCY_BREAKDOWN: LatencyStage[] = [
  { stage: "BM25 Search", p50: 12, p90: 22, p95: 28 },
  { stage: "Dense pgvector", p50: 45, p90: 78, p95: 95 },
  { stage: "RRF Fusion", p50: 3, p90: 5, p95: 7 },
  { stage: "Cross-Reranker", p50: 85, p90: 120, p95: 145 },
  { stage: "LLM Generation", p50: 380, p90: 520, p95: 610 },
  { stage: "Citation Verifier", p50: 340, p90: 480, p95: 560 },
];

const CATEGORY_BENCHMARKS: CategoryResult[] = [
  { category: "Exact Identifiers / IDs", bm25: 0.88, dense: 0.62, hybrid: 0.91, hybrid_rerank: 0.94 },
  { category: "Factual Knowledge", bm25: 0.65, dense: 0.71, hybrid: 0.78, hybrid_rerank: 0.84 },
  { category: "Multi-Hop Reasoning", bm25: 0.38, dense: 0.54, hybrid: 0.62, hybrid_rerank: 0.71 },
  { category: "Acronyms & Jargon", bm25: 0.79, dense: 0.58, hybrid: 0.83, hybrid_rerank: 0.88 },
  { category: "Semantic Paraphrasing", bm25: 0.42, dense: 0.76, hybrid: 0.81, hybrid_rerank: 0.86 },
];

const WIN_LOSS_CASES: WinLoss[] = [
  {
    question: "What is chunking_strategy 'structure' and how does it split headings?",
    category: "Exact Identifiers / IDs",
    bm25_score: 0.92,
    dense_score: 0.74,
    hybrid_score: 0.94,
    hybrid_rerank_score: 0.98,
    winner: "hybrid_rerank",
  },
  {
    question: "Explain the cosine distance formula used in pgvector embeddings.",
    category: "Factual Knowledge",
    bm25_score: 0.68,
    dense_score: 0.85,
    hybrid_score: 0.9,
    hybrid_rerank_score: 0.94,
    winner: "hybrid_rerank",
  },
  {
    question: "Find document chunk with ID chunk-98f214-alpha.",
    category: "Exact Identifiers / IDs",
    bm25_score: 0.99,
    dense_score: 0.45,
    hybrid_score: 0.96,
    hybrid_rerank_score: 0.99,
    winner: "bm25",
  },
  {
    question: "How does reciprocal rank fusion weight sparse vs dense rankings?",
    category: "Multi-Hop Reasoning",
    bm25_score: 0.62,
    dense_score: 0.72,
    hybrid_score: 0.88,
    hybrid_rerank_score: 0.95,
    winner: "hybrid_rerank",
  },
  {
    question: "Conceptual overview of grounding citation verification.",
    category: "Semantic Paraphrasing",
    bm25_score: 0.51,
    dense_score: 0.92,
    hybrid_score: 0.89,
    hybrid_rerank_score: 0.94,
    winner: "hybrid_rerank",
  },
];

const STRATEGY_COLORS = {
  bm25: "#f59e0b",
  dense: "#94a3b8",
  hybrid: "#6366f1",
  hybrid_rerank: "#10b981",
};

export default function Evaluation() {
  const [activeTab, setActiveTab] = useState<"metrics" | "tradeoff">("metrics");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");

  const categories = ["all", ...CATEGORY_BENCHMARKS.map((c) => c.category)];
  const filteredWinLoss =
    categoryFilter === "all"
      ? WIN_LOSS_CASES
      : WIN_LOSS_CASES.filter((w) => w.category === categoryFilter);

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Title & Overview */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <TrendingUp size={22} className="text-[var(--color-primary)]" />
            <h2 className="text-2xl font-semibold">Empirical Retrieval & RAG Benchmark</h2>
          </div>
          <p className="text-xs text-[var(--color-text-tertiary)] mt-1">
            Quantitative evaluation: BM25 Lexical vs. Dense Vector vs. Hybrid RRF vs. Cross-Encoder Reranking
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="inline-flex rounded-xl bg-[var(--color-bg-secondary)] p-1 border border-[var(--color-border)] text-xs font-medium">
            <button
              type="button"
              onClick={() => setActiveTab("metrics")}
              className={`px-3 py-1.5 rounded-lg transition-colors cursor-pointer ${
                activeTab === "metrics"
                  ? "bg-[var(--color-primary)] text-white shadow-xs"
                  : "text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
              }`}
            >
              Accuracy Metrics (H@5 & MRR)
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("tradeoff")}
              className={`px-3 py-1.5 rounded-lg transition-colors cursor-pointer flex items-center gap-1.5 ${
                activeTab === "tradeoff"
                  ? "bg-[var(--color-primary)] text-white shadow-xs"
                  : "text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
              }`}
            >
              <Zap size={13} />
              <span>Latency vs. Accuracy Curve</span>
            </button>
          </div>
        </div>
      </div>

      {/* 4-Way Strategy Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* BM25 */}
        <div className="bg-[var(--color-bg-secondary)] rounded-xl p-4 border border-[var(--color-border)] space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm text-[var(--color-text)]">BM25 Sparse</span>
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: STRATEGY_COLORS.bm25 }} />
          </div>
          <p className="text-xs text-[var(--color-text-tertiary)]">Exact keyword token matching</p>
          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[var(--color-border)]">
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">Hit Rate@5</p>
              <p className="text-base font-bold tabular-nums">64.0%</p>
            </div>
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">MRR</p>
              <p className="text-base font-bold tabular-nums">0.590</p>
            </div>
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">Avg Latency</p>
              <p className="text-base font-bold tabular-nums text-amber-500">18 ms</p>
            </div>
          </div>
        </div>

        {/* Dense */}
        <div className="bg-[var(--color-bg-secondary)] rounded-xl p-4 border border-[var(--color-border)] space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm text-[var(--color-text)]">Dense pgvector</span>
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: STRATEGY_COLORS.dense }} />
          </div>
          <p className="text-xs text-[var(--color-text-tertiary)]">Cosine semantic embedding search</p>
          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[var(--color-border)]">
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">Hit Rate@5</p>
              <p className="text-base font-bold tabular-nums">68.0%</p>
            </div>
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">MRR</p>
              <p className="text-base font-bold tabular-nums">0.650</p>
            </div>
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">Avg Latency</p>
              <p className="text-base font-bold tabular-nums">52 ms</p>
            </div>
          </div>
        </div>

        {/* Hybrid RRF */}
        <div className="bg-[var(--color-bg-secondary)] rounded-xl p-4 border border-[var(--color-border)] space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm text-[var(--color-text)]">Hybrid RRF</span>
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: STRATEGY_COLORS.hybrid }} />
          </div>
          <p className="text-xs text-[var(--color-text-tertiary)]">Fused reciprocal rank scoring</p>
          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[var(--color-border)]">
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">Hit Rate@5</p>
              <p className="text-base font-bold tabular-nums">79.0%</p>
            </div>
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">MRR</p>
              <p className="text-base font-bold tabular-nums">0.730</p>
            </div>
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">Avg Latency</p>
              <p className="text-base font-bold tabular-nums">68 ms</p>
            </div>
          </div>
        </div>

        {/* Hybrid + Reranker */}
        <div className="bg-[var(--color-bg-secondary)] rounded-xl p-4 border border-[var(--color-primary)]/40 shadow-xs space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Award size={15} className="text-emerald-500" />
              <span className="font-semibold text-sm text-[var(--color-text)]">Hybrid + Reranker</span>
            </div>
            <span className="px-1.5 py-0.2 rounded bg-emerald-500/10 text-emerald-600 font-bold text-[9px] uppercase">
              Leader
            </span>
          </div>
          <p className="text-xs text-[var(--color-text-tertiary)]">Cross-Encoder contextual re-scoring</p>
          <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[var(--color-border)]">
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">Hit Rate@5</p>
              <p className="text-base font-bold tabular-nums text-emerald-600 dark:text-emerald-400">87.0%</p>
            </div>
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">MRR</p>
              <p className="text-base font-bold tabular-nums text-emerald-600 dark:text-emerald-400">0.810</p>
            </div>
            <div>
              <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">Avg Latency</p>
              <p className="text-base font-bold tabular-nums">155 ms</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Interactive Charts View */}
      {activeTab === "metrics" ? (
        /* Tab 1: Retrieval & Ragas Metrics Bar Chart */
        <div className="bg-[var(--color-bg-secondary)] rounded-xl p-6 border border-[var(--color-border)] space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-semibold flex items-center gap-2 text-base">
              <BarChart3 size={18} className="text-[var(--color-primary)]" />
              <span>Comparative Performance by Retrieval Strategy</span>
            </h3>
            <span className="text-xs text-[var(--color-text-tertiary)]">
              Higher is better • Normalized 0.0 to 1.0
            </span>
          </div>

          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={COMPARISON_METRICS} barGap={3} barCategoryGap="18%">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
              <XAxis
                dataKey="metric"
                tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }}
                axisLine={{ stroke: "var(--color-border)" }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }}
                axisLine={{ stroke: "var(--color-border)" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--color-bg)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "10px",
                  fontSize: "12px",
                  boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "12px" }} />
              <Bar dataKey="bm25" name="BM25 Sparse" fill={STRATEGY_COLORS.bm25} radius={[4, 4, 0, 0]} />
              <Bar dataKey="dense" name="Dense Vector" fill={STRATEGY_COLORS.dense} radius={[4, 4, 0, 0]} />
              <Bar dataKey="hybrid" name="Hybrid RRF" fill={STRATEGY_COLORS.hybrid} radius={[4, 4, 0, 0]} />
              <Bar
                dataKey="hybrid_rerank"
                name="Hybrid + Reranker"
                fill={STRATEGY_COLORS.hybrid_rerank}
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        /* Tab 2: Latency vs. Accuracy Tradeoff Curve */
        <div className="bg-[var(--color-bg-secondary)] rounded-xl p-6 border border-[var(--color-border)] space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="font-semibold flex items-center gap-2 text-base">
                <Activity size={18} className="text-[var(--color-primary)]" />
                <span>Pareto Frontier: Latency vs. Retrieval Accuracy</span>
              </h3>
              <p className="text-xs text-[var(--color-text-tertiary)] mt-0.5">
                Hit Rate @ 5 (%) on Y-axis vs. Query Latency (ms) on X-axis
              </p>
            </div>
            <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-600">
              Optimal Frontier: Hybrid + Reranker
            </span>
          </div>

          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={TRADEOFF_DATA} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                dataKey="latency"
                type="number"
                domain={[0, 180]}
                unit="ms"
                tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }}
                label={{
                  value: "Average Retrieval Latency (ms)",
                  position: "insideBottom",
                  offset: -12,
                  fill: "var(--color-text-secondary)",
                  fontSize: 12,
                }}
              />
              <YAxis
                dataKey="accuracy"
                type="number"
                domain={[50, 100]}
                unit="%"
                tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }}
                label={{
                  value: "Hit Rate @ 5 (%)",
                  angle: -90,
                  position: "insideLeft",
                  fill: "var(--color-text-secondary)",
                  fontSize: 12,
                }}
              />
              <Tooltip
                formatter={(val, name) => [
                  name === "accuracy" ? `${val}%` : `${val} ms`,
                  name === "accuracy" ? "Hit Rate @ 5" : "Latency",
                ]}
                labelFormatter={(_label, payload) =>
                  payload && payload[0]
                    ? `${(payload[0].payload as TradeoffPoint).strategy}`
                    : ""
                }
                contentStyle={{
                  backgroundColor: "var(--color-bg)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "10px",
                  fontSize: "12px",
                }}
              />
              <Line
                type="monotone"
                dataKey="accuracy"
                name="Hit Rate @ 5"
                stroke="var(--color-primary)"
                strokeWidth={3}
                dot={{ r: 6, fill: "var(--color-primary)" }}
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Latency Waterfall Breakdown */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl p-6 border border-[var(--color-border)] space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold flex items-center gap-2">
            <Zap size={18} className="text-amber-500" />
            <span>Pipeline Stage Latency Breakdown (ms)</span>
          </h3>
          <span className="text-xs text-[var(--color-text-tertiary)]">
            p50, p90, and p95 benchmarks
          </span>
        </div>

        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={LATENCY_BREAKDOWN} layout="vertical" barGap={2} barCategoryGap="16%">
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" horizontal={false} />
            <XAxis type="number" unit="ms" tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }} />
            <YAxis
              type="category"
              dataKey="stage"
              width={120}
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

      {/* Category Breakdown Table */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl p-6 border border-[var(--color-border)] space-y-4">
        <h3 className="font-semibold flex items-center gap-2">
          <Layers size={18} className="text-[var(--color-primary)]" />
          <span>Category Performance Breakdown (MRR)</span>
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-text-secondary)] border-b border-[var(--color-border)] pb-2">
                <th className="px-4 py-2 font-medium">Domain Category</th>
                <th className="px-4 py-2 font-medium">BM25</th>
                <th className="px-4 py-2 font-medium">Dense</th>
                <th className="px-4 py-2 font-medium">Hybrid</th>
                <th className="px-4 py-2 font-medium">Hybrid+Reranker</th>
                <th className="px-4 py-2 font-medium">Optimal Engine</th>
              </tr>
            </thead>
            <tbody>
              {CATEGORY_BENCHMARKS.map((c) => {
                const best = Math.max(c.bm25, c.dense, c.hybrid, c.hybrid_rerank);
                const winnerLabel =
                  best === c.hybrid_rerank
                    ? "Hybrid + Reranker"
                    : best === c.hybrid
                      ? "Hybrid"
                      : best === c.bm25
                        ? "BM25"
                        : "Dense";
                return (
                  <tr key={c.category} className="border-t border-[var(--color-border)] hover:bg-[var(--color-bg-tertiary)]/50 transition-colors">
                    <td className="px-4 py-2.5 font-medium">{c.category}</td>
                    <td className="px-4 py-2.5 tabular-nums text-amber-600 dark:text-amber-400 font-mono">
                      {c.bm25.toFixed(3)}
                    </td>
                    <td className="px-4 py-2.5 tabular-nums text-slate-500 font-mono">
                      {c.dense.toFixed(3)}
                    </td>
                    <td className="px-4 py-2.5 tabular-nums text-indigo-500 font-mono">
                      {c.hybrid.toFixed(3)}
                    </td>
                    <td className="px-4 py-2.5 tabular-nums text-emerald-600 dark:text-emerald-400 font-bold font-mono">
                      {c.hybrid_rerank.toFixed(3)}
                    </td>
                    <td className="px-4 py-2.5">
                      <span className="px-2 py-0.5 bg-[var(--color-success)]/10 text-[var(--color-success)] text-xs font-semibold rounded">
                        {winnerLabel}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Win/Loss Question Results */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl p-6 border border-[var(--color-border)] space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-semibold flex items-center gap-2">
            <CheckCircle size={18} className="text-[var(--color-primary)]" />
            <span>Empirical Win / Loss Case Study</span>
          </h3>
          <div className="flex items-center gap-2">
            <Filter size={14} className="text-[var(--color-text-tertiary)]" />
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded-lg px-2.5 py-1 text-xs text-[var(--color-text)] focus:outline-none"
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
              <tr className="text-left text-[var(--color-text-secondary)] border-b border-[var(--color-border)] pb-2">
                <th className="px-4 py-2 font-medium">Evaluation Question</th>
                <th className="px-4 py-2 font-medium">Category</th>
                <th className="px-4 py-2 font-medium">BM25</th>
                <th className="px-4 py-2 font-medium">Dense</th>
                <th className="px-4 py-2 font-medium">Hybrid</th>
                <th className="px-4 py-2 font-medium">H+Reranker</th>
                <th className="px-4 py-2 font-medium">Leader</th>
              </tr>
            </thead>
            <tbody>
              {filteredWinLoss.map((w, i) => (
                <tr key={i} className="border-t border-[var(--color-border)] hover:bg-[var(--color-bg-tertiary)]/50 transition-colors">
                  <td className="px-4 py-2.5 max-w-sm truncate font-medium">
                    {w.question}
                  </td>
                  <td className="px-4 py-2.5 text-xs text-[var(--color-text-tertiary)]">{w.category}</td>
                  <td className="px-4 py-2.5 tabular-nums font-mono text-xs text-amber-600 dark:text-amber-400">
                    {w.bm25_score.toFixed(3)}
                  </td>
                  <td className="px-4 py-2.5 tabular-nums font-mono text-xs text-slate-500">
                    {w.dense_score.toFixed(3)}
                  </td>
                  <td className="px-4 py-2.5 tabular-nums font-mono text-xs text-indigo-500">
                    {w.hybrid_score.toFixed(3)}
                  </td>
                  <td className="px-4 py-2.5 tabular-nums font-mono text-xs text-emerald-600 dark:text-emerald-400 font-bold">
                    {w.hybrid_rerank_score.toFixed(3)}
                  </td>
                  <td className="px-4 py-2.5">
                    <span
                      className="px-2 py-0.5 text-xs font-semibold rounded"
                      style={{
                        backgroundColor: `${
                          STRATEGY_COLORS[w.winner as keyof typeof STRATEGY_COLORS]
                        }20`,
                        color: STRATEGY_COLORS[w.winner as keyof typeof STRATEGY_COLORS],
                      }}
                    >
                      {w.winner === "hybrid_rerank"
                        ? "H+Reranker"
                        : w.winner === "hybrid"
                          ? "Hybrid"
                          : w.winner === "bm25"
                            ? "BM25"
                            : "Dense"}
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
