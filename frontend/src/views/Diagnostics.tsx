import { ArrowDown, ArrowUp, Minus, Search } from "lucide-react";
import { useState } from "react";
import { fetchTrace, type Trace, type TraceStage } from "../lib/api";

function RankIndicator({ prev, curr }: { prev: number; curr: number }) {
  if (prev === -1) {
    return (
      <span className="text-[var(--positive)] text-[11px] font-medium">
        new
      </span>
    );
  }
  const diff = prev - curr;
  if (diff > 0) {
    return (
      <span className="inline-flex items-center gap-0.5 text-[var(--positive)] text-[11px] font-mono tabular-nums">
        <ArrowUp size={12} strokeWidth={1.5} />+{diff}
      </span>
    );
  }
  if (diff < 0) {
    return (
      <span className="inline-flex items-center gap-0.5 text-[var(--negative)] text-[11px] font-mono tabular-nums">
        <ArrowDown size={12} strokeWidth={1.5} />{diff}
      </span>
    );
  }
  return (
    <span className="text-[var(--text-muted)]">
      <Minus size={12} strokeWidth={1.5} />
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
        <h4 className="text-[12px] font-medium text-[var(--text-muted)] mb-3">
          {label}
        </h4>
        <p className="text-[12px] text-[var(--text-muted)]">
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
        <h4 className="text-[12px] font-medium text-[var(--text-secondary)]">
          {label}
        </h4>
        <span className="text-[11px] text-[var(--text-muted)] font-mono tabular-nums">
          {stage.duration_ms.toFixed(0)}ms
        </span>
      </div>
      <div className="space-y-1.5">
        {stage.top_scores.map((c, i) => (
          <div
            key={c.chunk_id}
            className="flex items-center gap-2 p-2 bg-[var(--surface-card)] rounded-lg text-[12px] border border-[var(--border)]"
          >
            <span className="w-5 text-center font-medium text-[var(--text-muted)] tabular-nums">
              {i + 1}
            </span>
            <span className="flex-1 font-mono truncate text-[var(--text-primary)]">
              {c.chunk_id.slice(0, 12)}...
            </span>
            <span className="text-[var(--text-secondary)] font-mono tabular-nums">
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
          <p className="text-[12px] text-[var(--text-muted)]">
            No candidates
          </p>
        )}
      </div>
      <p className="mt-2 text-[11px] text-[var(--text-muted)]">
        {scoreLabel} · {stage.candidate_count} total
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
      <div>
        <h2 className="text-xl font-medium">Diagnostics</h2>
        <p className="text-[13px] text-[var(--text-secondary)] mt-1">
          Inspect retrieval pipeline stages for a trace
        </p>
      </div>

      <form onSubmit={handleLoad} className="flex gap-2">
        <div className="flex-1 relative">
          <Search
            size={16}
            strokeWidth={1.5}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]"
          />
          <input
            type="text"
            value={traceId}
            onChange={(e) => setTraceId(e.target.value)}
            placeholder="Enter trace ID..."
            className="w-full pl-9 pr-4 py-2.5 border border-[var(--border)] rounded-lg bg-[var(--surface)] text-[13px] focus:outline-none focus:border-[var(--accent)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)]"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="px-5 py-2.5 bg-[var(--accent)] text-white rounded-lg text-[13px] font-medium hover:opacity-90 disabled:opacity-50 transition-colors cursor-pointer"
        >
          {loading ? "Loading..." : "Load trace"}
        </button>
      </form>

      {error && (
        <p className="text-[13px] text-[var(--negative)]">{error}</p>
      )}

      {trace && (
        <>
          <div className="text-[13px] text-[var(--text-secondary)] flex flex-wrap items-center gap-4">
            <span>
              Query: <span className="font-medium text-[var(--text-primary)]">{trace.query}</span>
            </span>
            <span className="font-mono tabular-nums text-[var(--text-muted)]">
              {trace.total_duration_ms.toFixed(0)}ms total
            </span>
          </div>

          <div className="grid grid-cols-4 gap-4">
            <StageColumn
              stage={getStage("dense")}
              prevStage={undefined}
              label="Dense"
              scoreLabel="Cosine similarity"
            />
            <StageColumn
              stage={getStage("sparse")}
              prevStage={undefined}
              label="BM25"
              scoreLabel="BM25 score"
            />
            <StageColumn
              stage={getStage("rrf")}
              prevStage={getStage("dense")}
              label="RRF fused"
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
        <div className="text-center py-16 text-[var(--text-muted)]">
          <Search size={32} strokeWidth={1.5} className="mx-auto mb-3 opacity-40" />
          <p className="text-[14px]">Enter a trace ID to inspect pipeline stages</p>
          <p className="text-[12px] mt-1">
            Trace IDs appear in Q&A results after each query
          </p>
        </div>
      )}
    </div>
  );
}
