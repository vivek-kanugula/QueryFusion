import {
  Activity,
  Award,
  BarChart3,
  Database,
  Layers,
  Loader2,
  TrendingUp,
  Zap,
} from "lucide-react";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
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
import { fetchEvalResults, type EvalStrategyResult } from "../lib/api";

const STRATEGY_COLORS: Record<string, string> = {
  bm25: "#f59e0b",
  dense: "#94a3b8",
  dense_only: "#94a3b8",
  hybrid: "#6366f1",
  hybrid_rerank: "#10b981",
};

const STRATEGY_LABELS: Record<string, string> = {
  bm25: "BM25 Sparse",
  dense: "Dense pgvector",
  dense_only: "Dense pgvector",
  hybrid: "Hybrid RRF",
  hybrid_rerank: "Hybrid + Reranker",
};

const STRATEGY_DESCRIPTIONS: Record<string, string> = {
  bm25: "Exact keyword token matching",
  dense: "Cosine semantic embedding search",
  dense_only: "Cosine semantic embedding search",
  hybrid: "Fused reciprocal rank scoring",
  hybrid_rerank: "Cross-Encoder contextual re-scoring",
};

function strategyLabel(key: string): string {
  return STRATEGY_LABELS[key] ?? key;
}

function strategyColor(key: string): string {
  return STRATEGY_COLORS[key] ?? "#64748b";
}

interface ComparisonRow {
  metric: string;
  [strategy: string]: string | number;
}

interface TradeoffPoint {
  strategy: string;
  latency: number;
  accuracy: number;
  mrr: number;
  color: string;
}

interface LatencyStage {
  stage: string;
  p50: number;
  p90: number;
  p95: number;
}

function buildComparisonMetrics(strategies: EvalStrategyResult[]): ComparisonRow[] {
  const metricDefs = [
    { key: "mrr", label: "MRR" },
    { key: "precision_at_k", label: "P@5" },
    { key: "recall_at_k", label: "R@5" },
    { key: "ndcg_at_k", label: "NDCG@5" },
    { key: "faithfulness_score", label: "Faithfulness" },
    { key: "citation_accuracy", label: "Citation Acc." },
  ];

  return metricDefs.map((def) => {
    const row: ComparisonRow = { metric: def.label };
    for (const s of strategies) {
      row[s.retrieval_strategy] = Number(
        (s.metrics[def.key as keyof typeof s.metrics] ?? 0).toFixed(3),
      );
    }
    return row;
  });
}

function buildTradeoffData(strategies: EvalStrategyResult[]): TradeoffPoint[] {
  return strategies.map((s) => {
    let avgLatency = 0;
    if (s.latency_breakdown && typeof s.latency_breakdown === "object") {
      const stages = s.latency_breakdown as Record<string, Record<string, number>>;
      const means = Object.values(stages)
        .map((v) => (typeof v === "object" && v !== null ? v.mean ?? 0 : 0))
        .filter((v) => v > 0);
      avgLatency = means.reduce((a, b) => a + b, 0);
    }
    return {
      strategy: strategyLabel(s.retrieval_strategy),
      latency: Math.round(avgLatency),
      accuracy: Math.round((s.metrics.precision_at_k ?? 0) * 100),
      mrr: Number((s.metrics.mrr ?? 0).toFixed(3)),
      color: strategyColor(s.retrieval_strategy),
    };
  });
}

function buildLatencyBreakdown(strategies: EvalStrategyResult[]): LatencyStage[] {
  const best = strategies.find((s) => s.retrieval_strategy === "hybrid_rerank") ?? strategies[0];
  if (!best?.latency_breakdown) return [];

  const breakdown = best.latency_breakdown as Record<
    string,
    { p50?: number; p90?: number; p95?: number; mean?: number }
  >;
  return Object.entries(breakdown)
    .filter(([, v]) => typeof v === "object" && v !== null)
    .map(([stage, v]) => ({
      stage: stage.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      p50: Math.round(v.p50 ?? v.mean ?? 0),
      p90: Math.round(v.p90 ?? 0),
      p95: Math.round(v.p95 ?? 0),
    }))
    .filter((s) => s.p50 > 0 || s.p90 > 0);
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center space-y-4">
      <div className="p-4 rounded-2xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)]">
        <Database size={32} className="text-[var(--color-text-tertiary)]" />
      </div>
      <div className="space-y-2 max-w-md">
        <h3 className="text-lg font-semibold text-[var(--color-text)]">
          No Experiment Results Yet
        </h3>
        <p className="text-sm text-[var(--color-text-secondary)]">
          Run the evaluation benchmark to populate this dashboard with real metrics.
          Results are stored in the database and displayed automatically.
        </p>
        <pre className="mt-4 text-xs bg-[var(--color-bg)] border border-[var(--color-border)] rounded-lg p-3 text-left text-[var(--color-text-secondary)]">
          <code>python -m hybrid_rag.eval.experiment_4</code>
        </pre>
      </div>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="flex items-center justify-center py-20">
      <Loader2 size={24} className="animate-spin text-[var(--color-primary)]" />
      <span className="ml-2 text-sm text-[var(--color-text-secondary)]">Loading evaluation results...</span>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center space-y-2">
      <p className="text-sm text-red-500">Failed to load evaluation results</p>
      <p className="text-xs text-[var(--color-text-tertiary)]">{message}</p>
    </div>
  );
}

export default function Evaluation() {
  const [activeTab, setActiveTab] = useState<"metrics" | "tradeoff">("metrics");

  const { data, isLoading, error } = useQuery({
    queryKey: ["eval-results"],
    queryFn: fetchEvalResults,
    staleTime: 60_000,
  });

  const strategies = data?.strategies ?? [];

  const comparisonMetrics = useMemo(() => buildComparisonMetrics(strategies), [strategies]);
  const tradeoffData = useMemo(() => buildTradeoffData(strategies), [strategies]);
  const latencyBreakdown = useMemo(() => buildLatencyBreakdown(strategies), [strategies]);

  const strategyKeys = strategies.map((s) => s.retrieval_strategy);
  const bestStrategy =
    strategies.length > 0
      ? strategies.reduce((best, s) => (s.metrics.mrr > best.metrics.mrr ? s : best))
      : null;

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto">
        <LoadingState />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto">
        <ErrorState message={error instanceof Error ? error.message : String(error)} />
      </div>
    );
  }

  if (!data?.has_data || strategies.length === 0) {
    return (
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="flex items-center gap-2">
          <TrendingUp size={22} className="text-[var(--color-primary)]" />
          <h2 className="text-2xl font-semibold">Empirical Retrieval & RAG Benchmark</h2>
        </div>
        <EmptyState />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Title & Overview */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <TrendingUp size={22} className="text-[var(--color-primary)]" />
            <h2 className="text-2xl font-semibold">Empirical Retrieval & RAG Benchmark</h2>
            <span className="text-[11px] font-medium px-2.5 py-0.5 rounded-full bg-[var(--color-primary)]/10 text-[var(--color-primary)] border border-[var(--color-primary)]/20">
              {strategies.length} Strategies • N={strategies[0]?.n_questions ?? "?"}
            </span>
          </div>
          <p className="text-xs text-[var(--color-text-tertiary)] mt-1">
            Empirical benchmark from stored evaluation runs: {strategyKeys.map(strategyLabel).join(" vs. ")}
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
              Accuracy Metrics
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
              <span>Latency vs. Accuracy</span>
            </button>
          </div>
        </div>
      </div>

      {/* Strategy Cards */}
      <div className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-${Math.min(strategies.length, 4)} gap-4`}>
        {strategies.map((s) => {
          const isBest = bestStrategy?.retrieval_strategy === s.retrieval_strategy;
          return (
            <div
              key={s.retrieval_strategy}
              className={`bg-[var(--color-bg-secondary)] rounded-xl p-4 border space-y-2 ${
                isBest
                  ? "border-[var(--color-primary)]/40 shadow-xs"
                  : "border-[var(--color-border)]"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  {isBest && <Award size={15} className="text-emerald-500" />}
                  <span className="font-semibold text-sm text-[var(--color-text)]">
                    {strategyLabel(s.retrieval_strategy)}
                  </span>
                </div>
                {isBest ? (
                  <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 font-bold text-[9px] uppercase">
                    Leader
                  </span>
                ) : (
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: strategyColor(s.retrieval_strategy) }}
                  />
                )}
              </div>
              <p className="text-xs text-[var(--color-text-tertiary)]">
                {STRATEGY_DESCRIPTIONS[s.retrieval_strategy] ?? s.retrieval_strategy}
              </p>
              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[var(--color-border)]">
                <div>
                  <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">P@5</p>
                  <p className={`text-base font-bold tabular-nums ${isBest ? "text-emerald-600 dark:text-emerald-400" : ""}`}>
                    {(s.metrics.precision_at_k * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">MRR</p>
                  <p className={`text-base font-bold tabular-nums ${isBest ? "text-emerald-600 dark:text-emerald-400" : ""}`}>
                    {s.metrics.mrr.toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-semibold">N</p>
                  <p className="text-base font-bold tabular-nums">{s.n_questions}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main Interactive Charts View */}
      {activeTab === "metrics" ? (
        <div className="bg-[var(--color-bg-secondary)] rounded-xl p-6 border border-[var(--color-border)] space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-semibold flex items-center gap-2 text-base">
              <BarChart3 size={18} className="text-[var(--color-primary)]" />
              <span>Comparative Performance by Retrieval Strategy</span>
            </h3>
            <span className="text-xs text-[var(--color-text-tertiary)]">
              Higher is better &bull; Normalized 0.0 to 1.0
            </span>
          </div>

          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={comparisonMetrics} barGap={3} barCategoryGap="18%">
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
              {strategyKeys.map((key) => (
                <Bar
                  key={key}
                  dataKey={key}
                  name={strategyLabel(key)}
                  fill={strategyColor(key)}
                  radius={[4, 4, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="bg-[var(--color-bg-secondary)] rounded-xl p-6 border border-[var(--color-border)] space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="font-semibold flex items-center gap-2 text-base">
                <Activity size={18} className="text-[var(--color-primary)]" />
                <span>Pareto Frontier: Latency vs. Retrieval Accuracy</span>
              </h3>
              <p className="text-xs text-[var(--color-text-tertiary)] mt-0.5">
                P@5 (%) on Y-axis vs. Query Latency (ms) on X-axis
              </p>
            </div>
            {bestStrategy && (
              <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-600">
                Optimal: {strategyLabel(bestStrategy.retrieval_strategy)}
              </span>
            )}
          </div>

          {tradeoffData.length > 0 ? (
            <ResponsiveContainer width="100%" height={340}>
              <LineChart data={tradeoffData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis
                  dataKey="latency"
                  type="number"
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
                  unit="%"
                  tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }}
                  label={{
                    value: "P@5 (%)",
                    angle: -90,
                    position: "insideLeft",
                    fill: "var(--color-text-secondary)",
                    fontSize: 12,
                  }}
                />
                <Tooltip
                  formatter={(val: number, name: string) => [
                    name === "accuracy" ? `${val}%` : `${val} ms`,
                    name === "accuracy" ? "P@5" : "Latency",
                  ]}
                  labelFormatter={(_label, payload) => {
                    const item = payload?.[0]?.payload as TradeoffPoint | undefined;
                    return item?.strategy ?? "";
                  }}
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
                  name="P@5"
                  stroke="var(--color-primary)"
                  strokeWidth={3}
                  dot={{ r: 6, fill: "var(--color-primary)" }}
                  activeDot={{ r: 8 }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-[var(--color-text-tertiary)] py-8 text-center">
              Latency data not available. Run experiments with stage-level timing to populate.
            </p>
          )}
        </div>
      )}

      {/* Latency Waterfall Breakdown */}
      {latencyBreakdown.length > 0 && (
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
            <BarChart data={latencyBreakdown} layout="vertical" barGap={2} barCategoryGap="16%">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" horizontal={false} />
              <XAxis type="number" unit="ms" tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }} />
              <YAxis
                type="category"
                dataKey="stage"
                width={140}
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
      )}

      {/* Metrics Detail Table */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl p-6 border border-[var(--color-border)] space-y-4">
        <h3 className="font-semibold flex items-center gap-2">
          <Layers size={18} className="text-[var(--color-primary)]" />
          <span>Full Metrics Comparison</span>
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-text-secondary)] border-b border-[var(--color-border)] pb-2">
                <th className="px-4 py-2 font-medium">Metric</th>
                {strategies.map((s) => (
                  <th key={s.retrieval_strategy} className="px-4 py-2 font-medium">
                    {strategyLabel(s.retrieval_strategy)}
                  </th>
                ))}
                <th className="px-4 py-2 font-medium">Best</th>
              </tr>
            </thead>
            <tbody>
              {comparisonMetrics.map((row) => {
                const values = strategyKeys.map(
                  (k) => (typeof row[k] === "number" ? row[k] : 0) as number,
                );
                const best = Math.max(...values);
                const bestIdx = values.indexOf(best);
                return (
                  <tr
                    key={row.metric}
                    className="border-t border-[var(--color-border)] hover:bg-[var(--color-bg-tertiary)]/50 transition-colors"
                  >
                    <td className="px-4 py-2.5 font-medium">{row.metric}</td>
                    {strategyKeys.map((k, i) => (
                      <td
                        key={k}
                        className={`px-4 py-2.5 tabular-nums font-mono ${
                          i === bestIdx ? "font-bold text-emerald-600 dark:text-emerald-400" : ""
                        }`}
                        style={i !== bestIdx ? { color: strategyColor(k) } : undefined}
                      >
                        {typeof row[k] === "number" ? (row[k] as number).toFixed(3) : "—"}
                      </td>
                    ))}
                    <td className="px-4 py-2.5">
                      <span className="px-2 py-0.5 bg-[var(--color-success)]/10 text-[var(--color-success)] text-xs font-semibold rounded">
                        {strategyLabel(strategyKeys[bestIdx])}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Run Info */}
      <div className="flex items-center gap-3 p-3.5 rounded-xl bg-[var(--color-bg-secondary)] border border-[var(--color-border)] text-xs text-[var(--color-text-secondary)]">
        <div className="p-1.5 rounded-lg bg-[var(--color-primary)]/10 text-[var(--color-primary)] shrink-0">
          <Award size={16} />
        </div>
        <div>
          <span className="font-semibold text-[var(--color-text)]">Experiment Data:</span>{" "}
          Results from stored evaluation runs. Re-run experiments to update metrics.
          Last run: {strategies[0]?.created_at ? new Date(strategies[0].created_at).toLocaleDateString() : "unknown"}.
        </div>
      </div>
    </div>
  );
}
