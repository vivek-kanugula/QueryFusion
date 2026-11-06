import { useQuery } from "@tanstack/react-query";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  FileText,
  Filter,
  Loader2,
  RotateCcw,
  Search,
  Shield,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { useLocation } from "react-router-dom";
import { fetchDocuments, type Document } from "../lib/api";
import { useSSEQuery } from "../hooks/useSSEQuery";

const MODES = [
  { value: "dense", label: "Dense", rerank: false },
  { value: "hybrid", label: "Hybrid", rerank: false },
  { value: "hybrid_rerank", label: "Hybrid + Reranker", rerank: true },
  { value: "bm25", label: "BM25", rerank: false },
] as const;

const STAGE_ORDER = [
  "dense",
  "sparse",
  "rrf",
  "rerank",
  "generation",
  "verification",
];

function StageBadge({
  name,
  active,
  done,
}: {
  name: string;
  active: boolean;
  done: boolean;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium transition-all ${
        done
          ? "bg-[var(--color-success)]/10 text-[var(--color-success)]"
          : active
            ? "bg-[var(--color-primary)]/10 text-[var(--color-primary)] animate-pulse"
            : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-tertiary)]"
      }`}
    >
      {done && <CheckCircle2 size={12} />}
      {active && !done && <Loader2 size={12} className="animate-spin" />}
      {name}
    </span>
  );
}

function ConfidenceBadge({ score }: { score: number }) {
  const color =
    score >= 0.75
      ? "var(--color-success)"
      : score >= 0.5
        ? "var(--color-warning)"
        : "var(--color-danger)";
  return (
    <span
      className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold"
      style={{ backgroundColor: `${color}20`, color }}
    >
      <Shield size={14} />
      {(score * 100).toFixed(0)}%
    </span>
  );
}

interface PersistedQASession {
  question: string;
  mode: string;
  alpha: number;
  selectedDocIds: string[];
  stages: string[];
  answer: Record<string, unknown> | null;
  verification: Record<string, unknown> | null;
  confidence: Record<string, unknown> | null;
  completed: Record<string, unknown> | null;
  error: string | null;
  traceId: string | null;
}

const SESSION_STORAGE_KEY = "hybrid_rag_qa_session";

function loadSessionState(): PersistedQASession | null {
  try {
    const raw = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as PersistedQASession;
  } catch {
    return null;
  }
}

function saveSessionState(data: PersistedQASession | null) {
  try {
    if (!data) {
      sessionStorage.removeItem(SESSION_STORAGE_KEY);
    } else {
      sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(data));
    }
  } catch {
    // Gracefully handle storage quota or privacy restrictions
  }
}

export default function QA() {
  const location = useLocation();
  const locationState = location.state as { selectedDocIds?: string[] } | null;

  const [savedSession] = useState<PersistedQASession | null>(() => loadSessionState());

  const [question, setQuestion] = useState(() => savedSession?.question ?? "");
  const [mode, setMode] = useState<string>(() => savedSession?.mode ?? "hybrid_rerank");
  const [alpha, setAlpha] = useState<number>(() => savedSession?.alpha ?? 0.5);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>(() => {
    if (locationState?.selectedDocIds && locationState.selectedDocIds.length > 0) {
      return locationState.selectedDocIds;
    }
    return savedSession?.selectedDocIds ?? [];
  });
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const [inspectChunk, setInspectChunk] = useState<{
    content: string;
    quote: string;
    citation_id?: string;
    document_id?: string;
  } | null>(null);

  const { data: docs = [] } = useQuery({
    queryKey: ["documents"],
    queryFn: fetchDocuments,
  });

  const sse = useSSEQuery(
    savedSession
      ? {
          stages: savedSession.stages || [],
          answer: savedSession.answer || null,
          verification: savedSession.verification || null,
          confidence: savedSession.confidence || null,
          completed: savedSession.completed || null,
          error: savedSession.error || null,
          traceId: savedSession.traceId || null,
        }
      : undefined,
  );

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const toggleDoc = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId],
    );
  };

  const selectAllDocs = () => {
    setSelectedDocIds(docs.map((d: Document) => d.id));
  };

  const clearDocSelection = () => {
    setSelectedDocIds([]);
  };

  const removeDoc = (docId: string) => {
    setSelectedDocIds((prev) => prev.filter((id) => id !== docId));
  };

  const selectedMode = MODES.find((m) => m.value === mode)!;
  const isHybrid = mode === "hybrid" || mode === "hybrid_rerank";

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || sse.isStreaming) return;
    sse.execute(question, {
      retrieval_strategy: mode,
      alpha,
      rerank_enabled: selectedMode.rerank,
      verify: true,
      document_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
    });
  };

  // Persist session to sessionStorage on changes
  useEffect(() => {
    if (sse.isStreaming) return;

    const hasState =
      question.trim() !== "" ||
      sse.answer !== null ||
      sse.stages.length > 0 ||
      sse.error !== null;

    if (hasState) {
      saveSessionState({
        question,
        mode,
        alpha,
        selectedDocIds,
        stages: sse.stages,
        answer: sse.answer,
        verification: sse.verification,
        confidence: sse.confidence,
        completed: sse.completed,
        error: sse.error,
        traceId: sse.traceId,
      });
    }
  }, [
    question,
    mode,
    alpha,
    selectedDocIds,
    sse.isStreaming,
    sse.stages,
    sse.answer,
    sse.verification,
    sse.confidence,
    sse.completed,
    sse.error,
    sse.traceId,
  ]);

  const handleResetSession = () => {
    setQuestion("");
    sse.reset();
    saveSessionState(null);
  };

  const answer = sse.answer as {
    answer?: string;
    citations?: Array<{
      citation_id: string;
      chunk_id: string;
      document_id?: string;
      evidence_quote: string;
    }>;
    has_sufficient_context?: boolean;
    fallback_reason?: string;
  } | null;

  const verification = sse.verification as {
    citation_coverage?: number;
    all_supported?: boolean;
    claims?: Array<{
      claim_text: string;
      citation_id: string;
      verdict: string;
      explanation: string;
    }>;
  } | null;

  const confidence = sse.confidence as {
    overall_score?: number;
    retrieval_score?: number;
    citation_coverage?: number;
    completeness_score?: number;
  } | null;

  const resolveCitation = (
    rawToken: string,
    citations?: Array<{
      citation_id?: string;
      chunk_id?: string;
      document_id?: string;
      evidence_quote?: string;
    }>,
  ): { citation_id: string; chunk_id: string; document_id?: string; evidence_quote: string } | undefined => {
    if (!citations || citations.length === 0) return undefined;

    const cleanToken = rawToken.replace(/^\[|\]$/g, "").trim();
    const lowerClean = cleanToken.toLowerCase();

    // 1. Exact match on citation_id (raw or bracket-stripped)
    const exact = citations.find((c) => {
      if (!c.citation_id) return false;
      const cid = c.citation_id.trim();
      const cleanCid = cid.replace(/^\[|\]$/g, "").trim();
      return (
        cid === rawToken ||
        cleanCid === cleanToken ||
        cid.toLowerCase() === rawToken.toLowerCase() ||
        cleanCid.toLowerCase() === lowerClean
      );
    });
    if (exact) {
      return {
        citation_id: exact.citation_id || rawToken,
        chunk_id: exact.chunk_id || "",
        document_id: exact.document_id,
        evidence_quote: exact.evidence_quote || "",
      };
    }

    // 2. Direct match on chunk_id or document_id
    const byId = citations.find(
      (c) =>
        (c.chunk_id && (c.chunk_id === rawToken || c.chunk_id === cleanToken)) ||
        (c.document_id && (c.document_id === rawToken || c.document_id === cleanToken)),
    );
    if (byId) {
      return {
        citation_id: byId.citation_id || rawToken,
        chunk_id: byId.chunk_id || "",
        document_id: byId.document_id,
        evidence_quote: byId.evidence_quote || "",
      };
    }

    // 3. Normalized numeric comparison (e.g. "[C1]", "[1]", "[^1]")
    const numMatch = cleanToken.match(/^(?:c|\^|doc-)?(\d+)$/i);
    if (numMatch) {
      const num = parseInt(numMatch[1], 10);
      const byNum = citations.find((c) => {
        if (!c.citation_id) return false;
        const cidClean = c.citation_id.replace(/^\[|\]$/g, "").trim();
        const m = cidClean.match(/^(?:c|\^|doc-)?(\d+)$/i);
        return m && parseInt(m[1], 10) === num;
      });
      if (byNum) {
        return {
          citation_id: byNum.citation_id || rawToken,
          chunk_id: byNum.chunk_id || "",
          document_id: byNum.document_id,
          evidence_quote: byNum.evidence_quote || "",
        };
      }

      // 4. Safe 1-based index fallback
      const idx = num - 1;
      if (idx >= 0 && idx < citations.length) {
        const fb = citations[idx];
        return {
          citation_id: fb.citation_id || rawToken,
          chunk_id: fb.chunk_id || "",
          document_id: fb.document_id,
          evidence_quote: fb.evidence_quote || "",
        };
      }
    }

    return undefined;
  };

  const renderAnswer = (text: string) => {
    const processed = text.replace(
      /(```[\s\S]*?```|`[^`\n]+`)|(\[(?:C\d+|\d+|\^\d+|doc[-_][\w-]+|chunk[-_][\w-]+)\])/gi,
      (_match, code, citation) => {
        if (code) return code;
        return `[${citation}](#citation:${encodeURIComponent(citation)})`;
      },
    );

    return (
      <ReactMarkdown
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith("#citation:")) {
              const rawToken = decodeURIComponent(href.slice(10));
              const citation = resolveCitation(rawToken, answer?.citations);
              return (
                <button
                  type="button"
                  onClick={() => {
                    if (citation) {
                      setInspectChunk({
                        content: citation.chunk_id,
                        quote: citation.evidence_quote || "",
                        citation_id: citation.citation_id,
                        document_id: citation.document_id,
                      });
                    }
                  }}
                  title={
                    citation?.evidence_quote
                      ? `Evidence: ${citation.evidence_quote.slice(0, 120)}...`
                      : citation?.chunk_id
                        ? `Chunk: ${citation.chunk_id}`
                        : `Citation ${rawToken}`
                  }
                  className="inline-flex items-center px-1.5 py-0.5 mx-0.5 text-xs font-mono font-semibold bg-[var(--color-primary)]/10 text-[var(--color-primary)] rounded hover:bg-[var(--color-primary)]/20 transition-colors cursor-pointer align-baseline"
                >
                  {rawToken}
                </button>
              );
            }
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[var(--color-primary)] underline hover:opacity-80"
              >
                {children}
              </a>
            );
          },
        }}
      >
        {processed}
      </ReactMarkdown>
    );
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold">Q&A Intelligence</h2>
        {(question || sse.answer || sse.stages.length > 0 || sse.error) && (
          <button
            type="button"
            onClick={handleResetSession}
            className="text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] px-3 py-1.5 rounded-lg border border-[var(--color-border)] hover:bg-[var(--color-bg-secondary)] transition-colors flex items-center gap-1.5 cursor-pointer"
            title="Reset question, answer, and session cache"
          >
            <RotateCcw size={13} />
            <span>Reset Session</span>
          </button>
        )}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Mode selector and Document scope selector */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          {/* Mode selector */}
          <div className="flex gap-2">
            {MODES.map((m) => (
              <button
                key={m.value}
                type="button"
                onClick={() => setMode(m.value)}
                className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                  mode === m.value
                    ? "bg-[var(--color-primary)] text-white border-[var(--color-primary)]"
                    : "border-[var(--color-border)] hover:border-[var(--color-border-hover)]"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          {/* Document Scope Dropdown */}
          {docs.length > 0 && (
            <div className="relative" ref={dropdownRef}>
              <button
                type="button"
                onClick={() => setDropdownOpen((prev) => !prev)}
                className={`inline-flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                  selectedDocIds.length > 0
                    ? "bg-[var(--color-primary)]/10 text-[var(--color-primary)] border-[var(--color-primary)]/30 hover:bg-[var(--color-primary)]/20"
                    : "bg-[var(--color-bg-secondary)] border-[var(--color-border)] text-[var(--color-text-secondary)] hover:border-[var(--color-border-hover)]"
                }`}
              >
                <Filter size={13} />
                <span>
                  {selectedDocIds.length === 0
                    ? `Scope: All Documents (${docs.length})`
                    : `Scope: ${selectedDocIds.length} of ${docs.length} selected`}
                </span>
                <ChevronDown size={13} />
              </button>

              {dropdownOpen && (
                <div className="absolute right-0 mt-1 w-72 bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl shadow-lg z-20 p-2 text-xs space-y-1">
                  <div className="flex items-center justify-between px-2 py-1 border-b border-[var(--color-border)] pb-2 mb-1">
                    <span className="font-semibold text-[var(--color-text)]">
                      Target Documents
                    </span>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={selectAllDocs}
                        className="text-[var(--color-primary)] hover:underline"
                      >
                        All
                      </button>
                      <span className="text-[var(--color-text-tertiary)]">|</span>
                      <button
                        type="button"
                        onClick={clearDocSelection}
                        className="text-[var(--color-text-tertiary)] hover:underline"
                      >
                        Reset
                      </button>
                    </div>
                  </div>

                  <div className="max-h-48 overflow-y-auto space-y-0.5">
                    {docs.map((d: Document) => {
                      const isChecked = selectedDocIds.includes(d.id);
                      return (
                        <label
                          key={d.id}
                          className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-[var(--color-bg-tertiary)] cursor-pointer text-[var(--color-text)] select-none"
                        >
                          <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={() => toggleDoc(d.id)}
                            className="accent-[var(--color-primary)] rounded"
                          />
                          <span className="truncate flex-1" title={d.filename}>
                            {d.filename}
                          </span>
                          <span className="text-[10px] text-[var(--color-text-tertiary)] uppercase font-mono">
                            {d.file_type}
                          </span>
                        </label>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Selected Document Chips (when filtered) */}
        {selectedDocIds.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            <span className="text-xs text-[var(--color-text-tertiary)] flex items-center gap-1">
              <FileText size={12} />
              Filtering by:
            </span>
            {selectedDocIds.map((id) => {
              const doc = docs.find((d: Document) => d.id === id);
              return (
                <span
                  key={id}
                  className="inline-flex items-center gap-1 px-2 py-0.5 bg-[var(--color-primary)]/10 text-[var(--color-primary)] rounded-md text-xs font-medium"
                >
                  <span className="max-w-[150px] truncate">
                    {doc ? doc.filename : id.slice(0, 8)}
                  </span>
                  <button
                    type="button"
                    onClick={() => removeDoc(id)}
                    className="hover:opacity-75 focus:outline-none"
                    title="Remove filter"
                  >
                    <X size={12} />
                  </button>
                </span>
              );
            })}
            <button
              type="button"
              onClick={clearDocSelection}
              className="text-xs text-[var(--color-text-tertiary)] hover:text-[var(--color-text)] underline ml-1"
            >
              Clear all
            </button>
          </div>
        )}

        {/* Alpha slider */}
        {isHybrid && (
          <div className="flex items-center gap-4">
            <label className="text-sm text-[var(--color-text-secondary)] w-32">
              Alpha: {alpha.toFixed(2)}
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={alpha}
              onChange={(e) => setAlpha(+e.target.value)}
              className="flex-1 accent-[var(--color-primary)]"
            />
            <span className="text-xs text-[var(--color-text-tertiary)] w-32 text-right">
              Dense ← → BM25
            </span>
          </div>
        )}

        {/* Search bar */}
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search
              size={18}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-tertiary)]"
            />
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about your documents..."
              className="w-full pl-10 pr-4 py-3 border border-[var(--color-border)] rounded-xl bg-[var(--color-bg)] text-sm focus:outline-none focus:border-[var(--color-primary)] focus:ring-1 focus:ring-[var(--color-primary)]"
            />
          </div>
          <button
            type="submit"
            disabled={sse.isStreaming || !question.trim()}
            className="px-6 py-3 bg-[var(--color-primary)] text-white rounded-xl text-sm font-medium hover:bg-[var(--color-primary-hover)] disabled:opacity-50 transition-colors"
          >
            {sse.isStreaming ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              "Ask"
            )}
          </button>
        </div>
      </form>

      {/* Stage badges */}
      {sse.stages.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {STAGE_ORDER.map((s) => {
            const idx = sse.stages.indexOf(s);
            const nextIdx = sse.stages.indexOf(
              STAGE_ORDER[STAGE_ORDER.indexOf(s) + 1],
            );
            const done = idx !== -1 && (nextIdx !== -1 || !sse.isStreaming);
            const active = idx !== -1 && !done;
            if (idx === -1 && !sse.stages.includes(s)) {
              if (s === "rerank" && !selectedMode.rerank) return null;
              if (s === "sparse" && mode === "dense") return null;
              if (s === "rrf" && mode === "dense") return null;
            }
            return <StageBadge key={s} name={s} active={active} done={done} />;
          })}
        </div>
      )}

      {/* Error */}
      {sse.error && (
        <div className="flex items-center gap-2 p-4 bg-[var(--color-danger)]/10 text-[var(--color-danger)] rounded-xl text-sm">
          <AlertCircle size={18} />
          {sse.error}
        </div>
      )}

      {/* Insufficient context warning */}
      {answer && answer.has_sufficient_context === false && (
        <div className="flex items-center gap-2 p-4 bg-[var(--color-warning)]/10 text-[var(--color-warning)] rounded-xl text-sm">
          <AlertCircle size={18} />
          Insufficient context.{" "}
          {answer.fallback_reason && <span>{answer.fallback_reason}</span>}
        </div>
      )}

      {/* Answer */}
      {answer?.answer && (
        <div className="bg-[var(--color-bg-secondary)] rounded-xl p-6 border border-[var(--color-border)] space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-lg">Answer</h3>
            {confidence?.overall_score != null && (
              <ConfidenceBadge score={confidence.overall_score} />
            )}
          </div>
          <div className="prose prose-sm max-w-none text-[var(--color-text)]">
            {renderAnswer(answer.answer)}
          </div>

          {/* Verification claims */}
          {verification?.claims && verification.claims.length > 0 && (
            <div className="mt-4 pt-4 border-t border-[var(--color-border)]">
              <h4 className="text-xs font-semibold uppercase text-[var(--color-text-tertiary)] mb-2">
                Citation Verification
              </h4>
              <div className="space-y-1.5">
                {verification.claims.map((c, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 text-xs"
                  >
                    <span
                      className={`shrink-0 px-1.5 py-0.5 rounded font-mono font-semibold ${
                        c.verdict === "SUPPORTED"
                          ? "bg-[var(--color-success)]/10 text-[var(--color-success)]"
                          : c.verdict === "UNSUPPORTED"
                            ? "bg-[var(--color-danger)]/10 text-[var(--color-danger)]"
                            : "bg-[var(--color-warning)]/10 text-[var(--color-warning)]"
                      }`}
                    >
                      {c.verdict}
                    </span>
                    <span className="text-[var(--color-text-secondary)]">
                      {c.claim_text}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Timing */}
          {sse.completed && (
            <div className="flex gap-4 text-xs text-[var(--color-text-tertiary)] pt-2">
              <span>
                Retrieval:{" "}
                {(sse.completed as Record<string, number>).retrieval_ms?.toFixed(
                  0,
                )}
                ms
              </span>
              <span>
                Generation:{" "}
                {(
                  sse.completed as Record<string, number>
                ).generation_ms?.toFixed(0)}
                ms
              </span>
              <span>
                Model: {(sse.completed as Record<string, string>).model}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Evidence inspector */}
      {inspectChunk && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-[var(--color-bg)] rounded-2xl p-6 w-[520px] max-h-[80vh] overflow-y-auto shadow-xl border border-[var(--color-border)]">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold">Evidence Details</h3>
              <button
                onClick={() => setInspectChunk(null)}
                className="p-1 hover:bg-[var(--color-bg-tertiary)] rounded"
              >
                <X size={18} />
              </button>
            </div>
            <div className="space-y-1 mb-3">
              {inspectChunk.citation_id && (
                <div className="text-xs font-mono text-[var(--color-primary)] font-semibold">
                  Citation: {inspectChunk.citation_id}
                </div>
              )}
              <div className="text-xs font-mono text-[var(--color-text-tertiary)]">
                Chunk ID: {inspectChunk.content || "N/A"}
              </div>
              {inspectChunk.document_id && (
                <div className="text-xs font-mono text-[var(--color-text-tertiary)]">
                  Document: {inspectChunk.document_id}
                </div>
              )}
            </div>
            {inspectChunk.quote ? (
              <div className="bg-[var(--color-warning)]/10 border-l-4 border-[var(--color-warning)] p-3 rounded text-sm">
                <ReactMarkdown>{inspectChunk.quote}</ReactMarkdown>
              </div>
            ) : (
              <p className="text-xs text-[var(--color-text-tertiary)] italic">
                No direct quote recorded for this citation.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
