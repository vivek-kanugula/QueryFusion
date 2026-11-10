import {
  Database,
  Loader2,
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
  bm25: "#d97706",
  dense: "#6b7280",
  dense_only: "#6b7280",
  hybrid: "#8b5cf6",
  hybrid_rerank: "#059669",
};

const STRATEGY_LABELS: Record<string, string> = {
  bm25: "BM25",
  dense: "Dense",
  dense_only: "Dense",
  hybrid: "Hybrid RRF",
  hybrid_rerank: "Hybrid + Reranker",
};

const STRATEGY_DESCRIPTIONS: Record<string, string> = {
  bm25: "Keyword token matching",
  dense: "Cosine embedding search",
  dense_only: "Cosine embedding search",
  hybrid: "Reciprocal rank fusion",
  hybrid_rerank: "Cross-encoder re-scoring",
};

function strategyLabel(key: string): string {
  return STRATEGY_LABELS[key] ?? key;
}

function strategyColor(key: string): string {
  return STRATEGY_COLORS[key] ?? "#6b7280";
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
    { key: "citation_accuracy", label: "Citation acc." },
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
      <div className="max-w-7xl mx-auto flex items-center justify-center py-20">
        <Loader2 size={20} strokeWidth={1.5} className="animate-spin text-[var(--accent-fg)]" />
        <span className="ml-2 text-[13px] text-[var(--text-secondary)]">Loading results...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto text-center py-20 space-y-2">
        <p className="text-[13px] text-[var(--negative)]">Failed to load evaluation results</p>
        <p className="text-[12px] text-[var(--text-muted)]">
          {error instanceof Error ? error.message : String(error)}
        </p>
      </div>
    );
  }

  if (!data?.has_data || strategies.length === 0) {
    return (
      <div className="max-w-7xl mx-auto space-y-6">
        <div>
          <h2 className="text-xl font-medium">Evaluation</h2>
          <p className="text-[13px] text-[var(--text-secondary)] mt-1">
            Benchmark results across retrieval strategies
          </p>
        </div>
        <div className="text-center py-16 text-[var(--text-muted)]">
          <Database size={32} strokeWidth={1.5} className="mx-auto mb-3 opacity-40" />
          <p className="text-[14px] text-[var(--text-secondary)]">No results yet</p>
          <p className="text-[12px] mt-1">
            Run the evaluation benchmark to populate this dashboard
          </p>
          <pre className="mt-4 text-[12px] bg-[var(--surface-card)] border border-[var(--border)] rounded-lg p-3 inline-block text-left text-[var(--text-secondary)] font-mono">
            python -m hybrid_rag.eval.experiment_4
          </pre>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-medium">Evaluation</h2>
          <p className="text-[13px] text-[var(--text-secondary)] mt-1">
            {strategies.length} strategies, N={strategies[0]?.n_questions ?? "?"} questions
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex rounded-lg bg-[var(--surface-card)] p-1 border border-[var(--border)]">
          <button
            type="button"
            onClick={() => setActiveTab("metrics")}
            className={`px-3 py-1.5 text-[12px] font-medium rounded-md transition-colors cursor-pointer ${
              activeTab === "metrics"
                ? "bg-[var(--surface-raised)] text-[var(--text-primary)]"
                : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
            }`}
          >
            Metrics
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("tradeoff")}
            className={`px-3 py-1.5 text-[12px] font-medium rounded-md transition-colors cursor-pointer ${
              activeTab === "tradeoff"
                ? "bg-[var(--surface-raised)] text-[var(--text-primary)]"
                : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
            }`}
          >
            Latency vs accuracy
          </button>
        </div>
      </div>

      {/* Strategy cards */}
      <div className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-${Math.min(strategies.length, 4)} gap-4`}>
        {strategies.map((s) => {
          const isBest = bestStrategy?.retrieval_strategy === s.retrieval_strategy;
          const color = strategyColor(s.retrieval_strategy);
          return (
            <div
              key={s.retrieval_strategy}
              className={`bg-[var(--surface-card)] rounded-xl p-4 border space-y-2 ${
                isBest ? "border-[var(--accent)]/30" : "border-[var(--border)]"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: color }}
                  />
                  <span className="font-medium text-[13px] text-[var(--text-primary)]">
                    {strategyLabel(s.retrieval_strategy)}
                  </span>
                </div>
                {isBest && (
                  <span className="text-[10px] font-medium text-[var(--positive)]">
                    best MRR
                  </span>
                )}
              </div>
              <p className="text-[12px] text-[var(--text-muted)]">
                {STRATEGY_DESCRIPTIONS[s.retrieval_strategy] ?? s.retrieval_strategy}
              </p>
              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[var(--border)]">
                <div>
                  <p className="text-[11px] text-[var(--text-muted)]">P@5</p>
                  <p className="text-[18px] font-medium tabular-nums">
                    {(s.metrics.precision_at_k * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-[11px] text-[var(--text-muted)]">MRR</p>
                  <p className="text-[18px] font-medium tabular-nums font-mono">
                    {s.metrics.mrr.toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className="text-[11px] text-[var(--text-muted)]">N</p>
                  <p className="text-[18px] font-medium tabular-nums">{s.n_questions}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main chart */}
      {activeTab === "metrics" ? (
        <div className="bg-[var(--surface-card)] rounded-xl p-6 border border-[var(--border)] space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="font-medium text-[14px]">Strategy comparison</h3>
            <span className="text-[11px] text-[var(--text-muted)]">
              Higher is better, 0.0 to 1.0
            </span>
          </div>

          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={comparisonMetrics} barGap={3} barCategoryGap="18%">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis
                dataKey="metric"
                tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
                axisLine={{ stroke: "var(--border)" }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
                axisLine={{ stroke: "var(--border)" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--surface-card)",
                  border: "1px solid var(--border)",
                  borderRadius: "8px",
                  fontSize: "12px",
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
        <div className="bg-[var(--surface-card)] rounded-xl p-6 border border-[var(--border)] space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="font-medium text-[14px]">Latency vs accuracy</h3>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
                P@5 (%) vs average query latency (ms)
              </p>
            </div>
            {bestStrategy && (
              <span className="text-[12px] text-[var(--positive)] font-medium">
                Best: {strategyLabel(bestStrategy.retrieval_strategy)}
              </span>
            )}
          </div>

          {tradeoffData.length > 0 ? (
            <ResponsiveContainer width="100%" height={340}>
              <LineChart data={tradeoffData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  dataKey="latency"
                  type="number"
                  unit="ms"
                  tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
                  label={{
                    value: "Latency (ms)",
                    position: "insideBottom",
                    offset: -12,
                    fill: "var(--text-secondary)",
                    fontSize: 12,
                  }}
                />
                <YAxis
                  dataKey="accuracy"
                  type="number"
                  unit="%"
                  tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
                  label={{
                    value: "P@5 (%)",
                    angle: -90,
                    position: "insideLeft",
                    fill: "var(--text-secondary)",
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
                    backgroundColor: "var(--surface-card)",
                    border: "1px solid var(--border)",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  name="P@5"
                  stroke="var(--accent)"
                  strokeWidth={2}
                  dot={{ r: 5, fill: "var(--accent)" }}
                  activeDot={{ r: 7 }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-[13px] text-[var(--text-muted)] py-8 text-center">
              Latency data not available. Run experiments with stage-level timing.
            </p>
          )}
        </div>
      )}

      {/* Latency breakdown */}
      {latencyBreakdown.length > 0 && (
        <div className="bg-[var(--surface-card)] rounded-xl p-6 border border-[var(--border)] space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-medium text-[14px]">Pipeline latency (ms)</h3>
            <span className="text-[11px] text-[var(--text-muted)]">
              p50, p90, p95
            </span>
          </div>

          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={latencyBreakdown} layout="vertical" barGap={2} barCategoryGap="16%">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
              <XAxis type="number" unit="ms" tick={{ fontSize: 12, fill: "var(--text-secondary)" }} />
              <YAxis
                type="category"
                dataKey="stage"
                width={140}
                tick={{ fontSize: 12, fill: "var(--text-secondary)" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--surface-card)",
                  border: "1px solid var(--border)",
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Bar dataKey="p50" name="p50" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              <Bar dataKey="p90" name="p90" fill="#a78bfa" radius={[0, 4, 4, 0]} />
              <Bar dataKey="p95" name="p95" fill="#c4b5fd" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Metrics table */}
      <div className="bg-[var(--surface-card)] rounded-xl border border-[var(--border)] overflow-hidden">
        <div className="px-6 py-4 border-b border-[var(--border)]">
          <h3 className="font-medium text-[14px]">Full metrics</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="text-left text-[var(--text-muted)] border-b border-[var(--border)]">
                <th className="px-4 py-3 font-medium">Metric</th>
                {strategies.map((s) => (
                  <th key={s.retrieval_strategy} className="px-4 py-3 font-medium">
                    {strategyLabel(s.retrieval_strategy)}
                  </th>
                ))}
                <th className="px-4 py-3 font-medium">Best</th>
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
                    className="border-t border-[var(--border)] hover:bg-[var(--surface-raised)]/50 transition-colors"
                  >
                    <td className="px-4 py-2.5 font-medium text-[var(--text-primary)]">{row.metric}</td>
                    {strategyKeys.map((k, i) => (
                      <td
                        key={k}
                        className={`px-4 py-2.5 tabular-nums font-mono ${
                          i === bestIdx ? "font-medium text-[var(--positive)]" : "text-[var(--text-secondary)]"
                        }`}
                      >
                        {typeof row[k] === "number" ? (row[k] as number).toFixed(3) : "—"}
                      </td>
                    ))}
                    <td className="px-4 py-2.5">
                      <span className="px-2 py-0.5 bg-[var(--positive)]/10 text-[var(--positive)] text-[12px] font-medium rounded">
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

      {/* Run info */}
      <p className="text-[12px] text-[var(--text-muted)]">
        Last run: {strategies[0]?.created_at ? new Date(strategies[0].created_at).toLocaleDateString() : "unknown"}
      </p>
    </div>
  );
}
