import { ArrowDown, ArrowUp, Minus, Search } from "lucide-react";
import { useState } from "react";
import { fetchTrace, type Trace, type TraceStage } from "../lib/api";

function RankIndicator({ prev, curr }: { prev: number; curr: number }) {
  if (prev === -1) {
    return (
      <span className="text-[var(--color-success)] text-xs font-semibold">
        NEW
      </span>
    );
  }
  const diff = prev - curr;
  if (diff > 0) {
    return (
      <span className="inline-flex items-center gap-0.5 text-[var(--color-success)] text-xs">
        <ArrowUp size={12} />+{diff}
      </span>
    );
  }
  if (diff < 0) {
    return (
      <span className="inline-flex items-center gap-0.5 text-[var(--color-danger)] text-xs">
        <ArrowDown size={12} />
        {diff}
      </span>
    );
  }
  return (
    <span className="text-[var(--color-text-tertiary)]">
      <Minus size={12} />
    </span>
  );
}

function StageColumn({
  stage,
  prevStage,
  label,
  scoreLabel,
}: {
  stage: TraceStage | undefined;
  prevStage: TraceStage | undefined;
  label: string;
  scoreLabel: string;
}) {
  if (!stage) {
    return (
      <div className="flex-1 min-w-0">
        <h4 className="text-xs font-semibold uppercase text-[var(--color-text-tertiary)] mb-3">
          {label}
        </h4>
        <p className="text-xs text-[var(--color-text-tertiary)] italic">
          Not available
        </p>
      </div>
    );
  }

  const prevRanks = new Map<string, number>();
  if (prevStage) {
    prevStage.top_scores.forEach((c, i) => prevRanks.set(c.chunk_id, i + 1));
  }

  return (
    <div className="flex-1 min-w-0">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-semibold uppercase text-[var(--color-text-tertiary)]">
          {label}
        </h4>
        <span className="text-xs text-[var(--color-text-tertiary)]">
          {stage.duration_ms.toFixed(0)}ms
        </span>
      </div>
      <div className="space-y-1.5">
        {stage.top_scores.map((c, i) => (
          <div
            key={c.chunk_id}
            className="flex items-center gap-2 p-2 bg-[var(--color-bg-secondary)] rounded-lg text-xs border border-[var(--color-border)]"
          >
            <span className="w-5 text-center font-semibold text-[var(--color-text-tertiary)]">
              {i + 1}
            </span>
            <span className="flex-1 font-mono truncate text-[var(--color-text)]">
              {c.chunk_id.slice(0, 12)}...
            </span>
            <span className="text-[var(--color-text-secondary)] tabular-nums">
              {c.score.toFixed(4)}
            </span>
            {prevStage && (
              <RankIndicator
                prev={prevRanks.get(c.chunk_id) ?? -1}
                curr={i + 1}
              />
            )}
          </div>
        ))}
        {stage.top_scores.length === 0 && (
          <p className="text-xs text-[var(--color-text-tertiary)] italic">
            No candidates
          </p>
        )}
      </div>
      <p className="mt-2 text-xs text-[var(--color-text-tertiary)]">
        {scoreLabel} &middot; {stage.candidate_count} total
      </p>
    </div>
  );
}

export default function Diagnostics() {
  const [traceId, setTraceId] = useState("");
  const [trace, setTrace] = useState<Trace | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLoad = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!traceId.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const t = await fetchTrace(traceId.trim());
      setTrace(t);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load trace");
      setTrace(null);
    } finally {
      setLoading(false);
    }
  };

  const getStage = (name: string): TraceStage | undefined =>
    trace?.stages.find((s) => s.stage === name);

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <h2 className="text-2xl font-semibold">Retrieval Diagnostics</h2>

      <form onSubmit={handleLoad} className="flex gap-2">
        <div className="flex-1 relative">
          <Search
            size={18}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-tertiary)]"
          />
          <input
            type="text"
            value={traceId}
            onChange={(e) => setTraceId(e.target.value)}
            placeholder="Enter trace ID from Q&A results..."
            className="w-full pl-10 pr-4 py-2.5 border border-[var(--color-border)] rounded-xl bg-[var(--color-bg)] text-sm focus:outline-none focus:border-[var(--color-primary)]"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="px-5 py-2.5 bg-[var(--color-primary)] text-white rounded-xl text-sm font-medium hover:bg-[var(--color-primary-hover)] disabled:opacity-50"
        >
          Load Trace
        </button>
      </form>

      {error && (
        <p className="text-sm text-[var(--color-danger)]">{error}</p>
      )}

      {trace && (
        <>
          <div className="text-sm text-[var(--color-text-secondary)]">
            Query: <span className="font-medium text-[var(--color-text)]">{trace.query}</span>
            <span className="ml-4">
              Total: {trace.total_duration_ms.toFixed(0)}ms
            </span>
          </div>

          <div className="grid grid-cols-4 gap-4">
            <StageColumn
              stage={getStage("dense")}
              prevStage={undefined}
              label="Dense Search"
              scoreLabel="Cosine similarity"
            />
            <StageColumn
              stage={getStage("sparse")}
              prevStage={undefined}
              label="BM25 Search"
              scoreLabel="BM25 score"
            />
            <StageColumn
              stage={getStage("rrf")}
              prevStage={getStage("dense")}
              label="RRF Fused"
              scoreLabel="RRF score"
            />
            <StageColumn
              stage={getStage("rerank")}
              prevStage={getStage("rrf")}
              label="Reranked"
              scoreLabel="Sigmoid score"
            />
          </div>
        </>
      )}

      {!trace && !error && (
        <div className="text-center py-16 text-[var(--color-text-tertiary)]">
          <Search size={48} className="mx-auto mb-3 opacity-50" />
          <p>Enter a trace ID to inspect retrieval pipeline stages</p>
          <p className="text-xs mt-1">
            Trace IDs appear in Q&A results after each query
          </p>
        </div>
      )}
    </div>
  );
}
