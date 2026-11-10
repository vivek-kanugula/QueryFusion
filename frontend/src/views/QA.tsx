import { useQuery } from "@tanstack/react-query";
import {
  AlertCircle,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  FileText,
  Filter,
  Layers,
  Loader2,
  MessageSquare,
  RotateCcw,
  Search,
  User,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { useLocation } from "react-router-dom";
import {
  fetchChunkContext,
  fetchDocuments,
  type ChunkContextResponse,
  type Document,
} from "../lib/api";
import { useSSEQuery } from "../hooks/useSSEQuery";

const MODES = [
  { value: "dense", label: "Dense", rerank: false },
  { value: "hybrid", label: "Hybrid", rerank: false },
  { value: "hybrid_rerank", label: "Hybrid + Reranker", rerank: true },
  { value: "bm25", label: "BM25", rerank: false },
] as const;

export interface ModelOption {
  id: string;
  name: string;
  provider: "gemini" | "groq" | "ollama";
  group: "Google Gemini" | "Groq Cloud" | "Local Ollama";
  description: string;
}

const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: "gemini-3.5-flash-lite",
    name: "Gemini 3.5 Flash Lite",
    provider: "gemini",
    group: "Google Gemini",
    description: "Fastest, lowest latency",
  },
  {
    id: "gemini-3.8-flash",
    name: "Gemini 3.8 Flash",
    provider: "gemini",
    group: "Google Gemini",
    description: "Balanced reasoning",
  },
  {
    id: "gemini-2.5-flash",
    name: "Gemini 2.5 Flash",
    provider: "gemini",
    group: "Google Gemini",
    description: "Production workhorse",
  },
  {
    id: "gemini-3.1-pro-preview",
    name: "Gemini 3.1 Pro",
    provider: "gemini",
    group: "Google Gemini",
    description: "Deep reasoning, citation scrutiny",
  },
  {
    id: "llama-3.3-70b-versatile",
    name: "Llama 3.3 70B",
    provider: "groq",
    group: "Groq Cloud",
    description: "70B open weights via Groq",
  },
  {
    id: "llama-3.1-8b-instant",
    name: "Llama 3.1 8B",
    provider: "groq",
    group: "Groq Cloud",
    description: "Sub-100ms generation",
  },
  {
    id: "ollama/llama3.2",
    name: "Llama 3.2 (Local)",
    provider: "ollama",
    group: "Local Ollama",
    description: "Runs on localhost:11434",
  },
];

const STAGE_ORDER = [
  "contextualize",
  "expansion",
  "dense",
  "sparse",
  "rrf",
  "rerank",
  "generation",
  "verification",
  "corrective_fallback",
];

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  contextualizedQuery?: string;
  expandedQueries?: string[];
  wasCorrected?: boolean;
  correctionReason?: string;
  citations?: Array<{
    citation_id: string;
    chunk_id: string;
    document_id?: string;
    evidence_quote: string;
  }>;
  verification?: {
    citation_coverage?: number;
    all_supported?: boolean;
    claims?: Array<{
      claim_text: string;
      citation_id: string;
      verdict: string;
      explanation: string;
    }>;
  };
  confidence?: {
    overall_score?: number;
    retrieval_score?: number;
    citation_coverage?: number;
    completeness_score?: number;
  };
  completed?: {
    retrieval_ms?: number;
    generation_ms?: number;
    total_ms?: number;
    stages?: Record<string, number>;
    cached?: boolean;
    model?: string;
    contextualized_query?: string | null;
    expanded_queries?: string[] | null;
    was_corrected?: boolean;
    correction_reason?: string | null;
  };
  has_sufficient_context?: boolean;
  fallback_reason?: string;
  error?: string;
}

interface PersistedQASession {
  question: string;
  mode: string;
  alpha: number;
  selectedDocIds: string[];
  selectedModel?: string;
  selectedProvider?: string;
  queryExpansion?: boolean;
  messages: ChatMessage[];
  stages?: string[];
  answer?: Record<string, unknown> | null;
  verification?: Record<string, unknown> | null;
  confidence?: Record<string, unknown> | null;
  completed?: Record<string, unknown> | null;
  error?: string | null;
  traceId?: string | null;
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
    // storage quota
  }
}

function loadInitialMessages(saved: PersistedQASession | null): ChatMessage[] {
  if (saved?.messages && Array.isArray(saved.messages) && saved.messages.length > 0) {
    return saved.messages;
  }
  if (saved?.question && (saved.answer || saved.completed || saved.error)) {
    const legacy: ChatMessage[] = [
      {
        id: "msg-user-init",
        role: "user",
        content: saved.question,
        timestamp: "Initial",
      },
    ];
    if (saved.answer?.answer || saved.error) {
      legacy.push({
        id: "msg-assistant-init",
        role: "assistant",
        content: String((saved.answer as { answer?: string })?.answer || ""),
        timestamp: "Initial",
        citations: (saved.answer as { citations?: ChatMessage["citations"] })?.citations || [],
        verification: saved.verification as ChatMessage["verification"],
        confidence: saved.confidence as ChatMessage["confidence"],
        completed: saved.completed as ChatMessage["completed"],
        has_sufficient_context: (saved.answer as { has_sufficient_context?: boolean })
          ?.has_sufficient_context,
        fallback_reason: (saved.answer as { fallback_reason?: string })?.fallback_reason,
        error: saved.error || undefined,
      });
    }
    return legacy;
  }
  return [];
}

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
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium transition-colors ${
        done
          ? "bg-[var(--surface-raised)] text-[var(--positive)]"
          : active
            ? "bg-[var(--surface-raised)] text-[var(--accent-fg)]"
            : "bg-[var(--surface-raised)] text-[var(--text-muted)]"
      }`}
    >
      {done && <CheckCircle2 size={12} strokeWidth={1.5} />}
      {active && !done && <Loader2 size={12} strokeWidth={1.5} className="animate-spin" />}
      {name}
    </span>
  );
}

function ConfidenceBadge({ score }: { score: number }) {
  const color =
    score >= 0.75
      ? "var(--positive)"
      : score >= 0.5
        ? "var(--warning)"
        : "var(--negative)";
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg text-[11px] font-medium font-mono tabular-nums"
      style={{ backgroundColor: `color-mix(in srgb, ${color} 12%, transparent)`, color }}
    >
      {(score * 100).toFixed(0)}%
    </span>
  );
}

export default function QA() {
  const location = useLocation();
  const locationState = location.state as { selectedDocIds?: string[] } | null;

  const [savedSession] = useState<PersistedQASession | null>(() => loadSessionState());

  const [messages, setMessages] = useState<ChatMessage[]>(() =>
    loadInitialMessages(savedSession),
  );
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<string>(() => savedSession?.mode ?? "hybrid_rerank");
  const [alpha, setAlpha] = useState<number>(() => savedSession?.alpha ?? 0.5);
  const [queryExpansion, setQueryExpansion] = useState<boolean>(
    () => savedSession?.queryExpansion ?? false,
  );
  const [adaptiveAlpha, setAdaptiveAlpha] = useState(false);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>(() => {
    if (locationState?.selectedDocIds && locationState.selectedDocIds.length > 0) {
      return locationState.selectedDocIds;
    }
    return savedSession?.selectedDocIds ?? [];
  });

  const [selectedModel, setSelectedModel] = useState<string>(
    () => savedSession?.selectedModel ?? "gemini-3.5-flash-lite",
  );
  const [selectedProvider, setSelectedProvider] = useState<string>(
    () => savedSession?.selectedProvider ?? "gemini",
  );

  const [modelDropdownOpen, setModelDropdownOpen] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const modelDropdownRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [copiedId, setCopiedId] = useState<string | null>(null);

  const [inspectCitation, setInspectCitation] = useState<{
    citation_id: string;
    chunk_id: string;
    document_id?: string;
    evidence_quote: string;
  } | null>(null);

  const [citationContextData, setCitationContextData] = useState<ChunkContextResponse | null>(null);
  const [citationLoading, setCitationLoading] = useState(false);
  const [activeInspectorTab, setActiveInspectorTab] = useState<"target" | "adjacent">("target");

  const { data: docs = [] } = useQuery({
    queryKey: ["documents"],
    queryFn: fetchDocuments,
  });

  const sse = useSSEQuery();
  const lastTraceAppendedRef = useRef<string | null>(null);

  useEffect(() => {
    const chunkId = inspectCitation?.chunk_id;
    if (!chunkId) return;

    let isMounted = true;
    fetchChunkContext(chunkId)
      .then((data) => {
        if (isMounted) {
          setCitationContextData(data);
          setCitationLoading(false);
        }
      })
      .catch(() => {
        if (isMounted) {
          setCitationContextData(null);
          setCitationLoading(false);
        }
      });

    return () => { isMounted = false; };
  }, [inspectCitation?.chunk_id]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
      if (modelDropdownRef.current && !modelDropdownRef.current.contains(event.target as Node)) {
        setModelDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sse.isStreaming, sse.stages]);

  useEffect(() => {
    if (sse.isStreaming) return;
    if (!sse.answer?.answer && !sse.error) return;

    const traceId =
      sse.traceId ||
      ((sse.completed as { trace_id?: string })?.trace_id) ||
      String(Date.now());

    if (lastTraceAppendedRef.current === traceId) return;
    lastTraceAppendedRef.current = traceId;

    const completedData = sse.completed as ChatMessage["completed"];
    const answerData = sse.answer as {
      answer?: string;
      citations?: ChatMessage["citations"];
      has_sufficient_context?: boolean;
      fallback_reason?: string;
    } | null;

    const assistantMsg: ChatMessage = {
      id: `assistant-${traceId}`,
      role: "assistant",
      content: answerData?.answer || (sse.error ? `Error: ${sse.error}` : ""),
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      contextualizedQuery:
        (completedData as { contextualized_query?: string | null })?.contextualized_query || undefined,
      expandedQueries:
        (completedData as { expanded_queries?: string[] | null })?.expanded_queries || undefined,
      wasCorrected: (completedData as { was_corrected?: boolean })?.was_corrected,
      correctionReason:
        (completedData as { correction_reason?: string | null })?.correction_reason || undefined,
      citations: answerData?.citations || [],
      verification: sse.verification as ChatMessage["verification"],
      confidence: sse.confidence as ChatMessage["confidence"],
      completed: completedData || undefined,
      has_sufficient_context: answerData?.has_sufficient_context,
      fallback_reason: answerData?.fallback_reason,
      error: sse.error || undefined,
    };

    setMessages((prev) => [...prev, assistantMsg]);
  }, [sse.isStreaming, sse.answer, sse.verification, sse.confidence, sse.completed, sse.error, sse.traceId]);

  useEffect(() => {
    if (sse.isStreaming) return;
    if (messages.length > 0 || question.trim() !== "") {
      saveSessionState({
        question, mode, alpha, queryExpansion, selectedDocIds, selectedModel, selectedProvider, messages,
      });
    }
  }, [messages, question, mode, alpha, queryExpansion, selectedDocIds, selectedModel, selectedProvider, sse.isStreaming]);

  const toggleDoc = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId],
    );
  };

  const selectAllDocs = () => setSelectedDocIds(docs.map((d: Document) => d.id));
  const clearDocSelection = () => setSelectedDocIds([]);
  const removeDoc = (docId: string) => setSelectedDocIds((prev) => prev.filter((id) => id !== docId));

  const selectedMode = MODES.find((m) => m.value === mode)!;
  const isHybrid = mode === "hybrid" || mode === "hybrid_rerank";
  const activeModel = AVAILABLE_MODELS.find((m) => m.id === selectedModel) || AVAILABLE_MODELS[0];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || sse.isStreaming) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setQuestion("");

    const chatHistory = messages.map((m) => ({ role: m.role, content: m.content }));

    sse.execute(trimmed, {
      retrieval_strategy: mode,
      alpha,
      rerank_enabled: selectedMode.rerank,
      verify: true,
      document_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
      chat_history: chatHistory.length > 0 ? chatHistory : undefined,
      model: selectedModel,
      provider: selectedProvider,
      query_expansion: queryExpansion,
      enable_crag: true,
      adaptive_alpha: adaptiveAlpha,
    });
  };

  const handleResetSession = () => {
    setQuestion("");
    setMessages([]);
    setInspectCitation(null);
    setCitationContextData(null);
    sse.reset();
    saveSessionState(null);
  };

  const handleCopy = (id: string, text: string) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleExportMarkdown = (msg?: ChatMessage) => {
    const targetMessages = msg ? [msg] : messages;
    if (targetMessages.length === 0) return;

    let md = `# Q&A Session\n\n`;
    md += `**Date:** ${new Date().toISOString()}\n`;
    md += `**Model:** ${activeModel.name} (${activeModel.provider})\n`;
    md += `**Retrieval:** ${mode} (alpha ${alpha})\n`;
    if (queryExpansion) md += `**Query expansion:** enabled\n`;
    md += `\n---\n\n`;

    targetMessages.forEach((m, idx) => {
      const speaker = m.role === "user" ? "### User" : `### Assistant (${activeModel.name})`;
      md += `${speaker} [${m.timestamp}]\n\n${m.content}\n\n`;

      if (m.contextualizedQuery) {
        md += `> **Reformulated query:** ${m.contextualizedQuery}\n\n`;
      }
      if (m.expandedQueries && m.expandedQueries.length > 1) {
        md += `> **Expanded queries:** ${m.expandedQueries.join(", ")}\n\n`;
      }
      if (m.wasCorrected) {
        md += `> **Self-corrected:** ${m.correctionReason || "Re-retrieved with broader scope"}\n\n`;
      }

      if (m.verification?.claims && m.verification.claims.length > 0) {
        md += `**Verification:**\n`;
        m.verification.claims.forEach((c) => {
          md += `- [${c.verdict}] ${c.claim_text}\n`;
        });
        md += `\n`;
      }

      if (m.citations && m.citations.length > 0) {
        md += `**Citations:**\n`;
        m.citations.forEach((c) => {
          md += `- [${c.citation_id}] Chunk: \`${c.chunk_id}\` "${c.evidence_quote}"\n`;
        });
        md += `\n`;
      }

      if (idx < targetMessages.length - 1) md += `---\n\n`;
    });

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `qa_session_${Date.now()}.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleExportJSON = () => {
    const payload = {
      model: activeModel,
      mode,
      alpha,
      queryExpansion,
      timestamp: new Date().toISOString(),
      messages,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `qa_session_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const resolveCitation = (
    rawToken: string,
    citations?: ChatMessage["citations"],
  ): { citation_id: string; chunk_id: string; document_id?: string; evidence_quote: string } | undefined => {
    if (!citations || citations.length === 0) return undefined;

    const cleanToken = rawToken.replace(/^\[|\]$/g, "").trim();
    const lowerClean = cleanToken.toLowerCase();

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

  const renderAnswer = (text: string, citations?: ChatMessage["citations"]) => {
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
              const citation = resolveCitation(rawToken, citations);
              return (
                <button
                  type="button"
                  onClick={() => {
                    if (citation) {
                      setCitationContextData(null);
                      setCitationLoading(true);
                      setInspectCitation({
                        citation_id: citation.citation_id,
                        chunk_id: citation.chunk_id,
                        document_id: citation.document_id,
                        evidence_quote: citation.evidence_quote || "",
                      });
                      setActiveInspectorTab("target");
                    }
                  }}
                  title={
                    citation?.evidence_quote
                      ? `Evidence: ${citation.evidence_quote.slice(0, 120)}...`
                      : citation?.chunk_id
                        ? `Chunk: ${citation.chunk_id}`
                        : `Citation ${rawToken}`
                  }
                  className="inline-flex items-center px-1.5 py-0.5 mx-0.5 text-[11px] font-mono font-medium bg-[var(--surface-raised)] text-[var(--accent-fg)] rounded hover:bg-[var(--accent)]/10 transition-colors cursor-pointer align-baseline"
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
                className="text-[var(--accent-fg)] underline hover:opacity-80"
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
    <div className="relative min-h-[calc(100vh-80px)] flex">
      <div
        className={`flex-1 transition-all duration-300 ${
          inspectCitation ? "lg:mr-[440px]" : ""
        }`}
      >
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Header */}
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-xl font-medium">Q&A</h2>
              <p className="text-[13px] text-[var(--text-secondary)] mt-1">
                Ask questions about your documents
              </p>
            </div>

            <div className="flex items-center gap-2">
              {messages.length > 0 && (
                <>
                  <button
                    type="button"
                    onClick={() => handleExportMarkdown()}
                    className="inline-flex items-center gap-1.5 px-3 py-2 text-[12px] font-medium rounded-lg border border-[var(--border)] hover:bg-[var(--surface-raised)] text-[var(--text-secondary)] transition-colors cursor-pointer"
                    title="Export as Markdown"
                  >
                    <Download size={13} strokeWidth={1.5} />
                    Export
                  </button>
                  <button
                    type="button"
                    onClick={handleExportJSON}
                    className="inline-flex items-center px-2.5 py-2 text-[12px] font-medium font-mono rounded-lg border border-[var(--border)] hover:bg-[var(--surface-raised)] text-[var(--text-secondary)] transition-colors cursor-pointer"
                    title="Export as JSON"
                  >
                    JSON
                  </button>
                  <button
                    type="button"
                    onClick={handleResetSession}
                    className="inline-flex items-center gap-1.5 px-3 py-2 text-[12px] font-medium rounded-lg border border-[var(--border)] hover:bg-[var(--surface-raised)] text-[var(--text-muted)] hover:text-[var(--negative)] transition-colors cursor-pointer"
                    title="Clear conversation"
                  >
                    <RotateCcw size={13} strokeWidth={1.5} />
                    New chat
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="bg-[var(--surface-card)] border border-[var(--border)] rounded-xl p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              {/* Strategy buttons */}
              <div className="flex items-center gap-1.5">
                <span className="text-[12px] font-medium text-[var(--text-muted)] mr-1">
                  Search
                </span>
                {MODES.map((m) => (
                  <button
                    key={m.value}
                    type="button"
                    onClick={() => setMode(m.value)}
                    className={`px-2.5 py-1.5 text-[12px] rounded-lg border transition-colors ${
                      mode === m.value
                        ? "bg-[var(--accent)] text-white border-[var(--accent)] font-medium"
                        : "border-[var(--border)] hover:bg-[var(--surface-raised)] text-[var(--text-secondary)]"
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {/* Query expansion toggle */}
                <button
                  type="button"
                  onClick={() => setQueryExpansion((prev) => !prev)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium rounded-lg border transition-colors cursor-pointer ${
                    queryExpansion
                      ? "bg-[var(--accent)] text-white border-[var(--accent)]"
                      : "bg-[var(--surface)] border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--surface-raised)]"
                  }`}
                  title="Generate alternate query variations to maximize retrieval recall"
                >
                  HyDE
                  <span
                    className={`text-[10px] px-1 rounded font-medium ${
                      queryExpansion
                        ? "bg-white/20 text-white"
                        : "bg-[var(--surface-raised)] text-[var(--text-muted)]"
                    }`}
                  >
                    {queryExpansion ? "on" : "off"}
                  </span>
                </button>

                {/* Adaptive alpha toggle */}
                <button
                  type="button"
                  onClick={() => setAdaptiveAlpha((prev) => !prev)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium rounded-lg border transition-colors cursor-pointer ${
                    adaptiveAlpha
                      ? "bg-[var(--accent)] text-white border-[var(--accent)]"
                      : "bg-[var(--surface)] border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--surface-raised)]"
                  }`}
                  title="Auto-tune alpha based on query difficulty classification"
                >
                  Auto-α
                  <span
                    className={`text-[10px] px-1 rounded font-medium ${
                      adaptiveAlpha
                        ? "bg-white/20 text-white"
                        : "bg-[var(--surface-raised)] text-[var(--text-muted)]"
                    }`}
                  >
                    {adaptiveAlpha ? "on" : "off"}
                  </span>
                </button>

                {/* Model selector */}
                <div className="relative" ref={modelDropdownRef}>
                  <button
                    type="button"
                    onClick={() => setModelDropdownOpen((prev) => !prev)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium rounded-lg border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--surface-raised)] text-[var(--text-primary)] transition-colors cursor-pointer"
                    title="Select generation model"
                  >
                    <span>{activeModel.name}</span>
                    <ChevronDown size={12} strokeWidth={1.5} className="text-[var(--text-muted)]" />
                  </button>

                  {modelDropdownOpen && (
                    <div className="absolute right-0 mt-1 w-72 bg-[var(--surface-card)] border border-[var(--border)] rounded-xl shadow-lg z-30 p-2 text-[12px]">
                      <div className="max-h-72 overflow-y-auto space-y-1">
                        {(["Google Gemini", "Groq Cloud", "Local Ollama"] as const).map(
                          (groupName) => {
                            const groupModels = AVAILABLE_MODELS.filter((m) => m.group === groupName);
                            return (
                              <div key={groupName} className="space-y-0.5">
                                <div className="px-2 pt-2 pb-1 text-[11px] font-medium text-[var(--text-muted)]">
                                  {groupName}
                                </div>
                                {groupModels.map((m) => {
                                  const isSelected = selectedModel === m.id;
                                  return (
                                    <button
                                      key={m.id}
                                      type="button"
                                      onClick={() => {
                                        setSelectedModel(m.id);
                                        setSelectedProvider(m.provider);
                                        setModelDropdownOpen(false);
                                      }}
                                      className={`w-full text-left px-2 py-1.5 rounded-lg flex items-center justify-between gap-2 transition-colors cursor-pointer ${
                                        isSelected
                                          ? "bg-[var(--surface-raised)] text-[var(--text-primary)] font-medium"
                                          : "hover:bg-[var(--surface-raised)] text-[var(--text-secondary)]"
                                      }`}
                                    >
                                      <div className="flex-1 min-w-0">
                                        <div className="truncate">{m.name}</div>
                                        <div className="text-[11px] text-[var(--text-muted)] truncate">
                                          {m.description}
                                        </div>
                                      </div>
                                      {isSelected && <Check size={14} strokeWidth={1.5} className="shrink-0 text-[var(--accent-fg)]" />}
                                    </button>
                                  );
                                })}
                              </div>
                            );
                          },
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Document scope */}
                {docs.length > 0 && (
                  <div className="relative" ref={dropdownRef}>
                    <button
                      type="button"
                      onClick={() => setDropdownOpen((prev) => !prev)}
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium rounded-lg border transition-colors cursor-pointer ${
                        selectedDocIds.length > 0
                          ? "bg-[var(--accent)]/10 text-[var(--accent-fg)] border-[var(--accent)]/30 hover:bg-[var(--accent)]/15"
                          : "bg-[var(--surface)] border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--surface-raised)]"
                      }`}
                    >
                      <Filter size={13} strokeWidth={1.5} />
                      <span>
                        {selectedDocIds.length === 0
                          ? `All docs (${docs.length})`
                          : `${selectedDocIds.length} selected`}
                      </span>
                      <ChevronDown size={12} strokeWidth={1.5} />
                    </button>

                    {dropdownOpen && (
                      <div className="absolute right-0 mt-1 w-72 bg-[var(--surface-card)] border border-[var(--border)] rounded-xl shadow-lg z-30 p-2 text-[12px] space-y-1">
                        <div className="flex items-center justify-between px-2 py-1 border-b border-[var(--border)] pb-2 mb-1">
                          <span className="font-medium text-[var(--text-primary)]">
                            Filter documents
                          </span>
                          <div className="flex gap-2 text-[11px]">
                            <button type="button" onClick={selectAllDocs} className="text-[var(--accent-fg)] hover:underline">
                              All
                            </button>
                            <span className="text-[var(--text-muted)]">|</span>
                            <button type="button" onClick={clearDocSelection} className="text-[var(--text-muted)] hover:underline">
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
                                className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-[var(--surface-raised)] cursor-pointer text-[var(--text-primary)] select-none"
                              >
                                <input
                                  type="checkbox"
                                  checked={isChecked}
                                  onChange={() => toggleDoc(d.id)}
                                />
                                <span className="truncate flex-1" title={d.filename}>
                                  {d.filename}
                                </span>
                                <span className="text-[10px] text-[var(--text-muted)] font-mono">
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
            </div>

            {/* Alpha slider */}
            {isHybrid && (
              <div className="flex items-center gap-4 pt-2 border-t border-[var(--border)]">
                <label className="text-[12px] font-medium text-[var(--text-secondary)] w-28">
                  Alpha: <span className="font-mono tabular-nums">{alpha.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={alpha}
                  onChange={(e) => setAlpha(+e.target.value)}
                  className="flex-1"
                />
                <span className="text-[11px] text-[var(--text-muted)] w-28 text-right font-mono">
                  Dense - BM25
                </span>
              </div>
            )}
          </div>

          {/* Selected doc chips */}
          {selectedDocIds.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[12px] text-[var(--text-muted)] flex items-center gap-1">
                <FileText size={12} strokeWidth={1.5} />
                Filtered:
              </span>
              {selectedDocIds.map((id) => {
                const doc = docs.find((d: Document) => d.id === id);
                return (
                  <span
                    key={id}
                    className="inline-flex items-center gap-1 px-2 py-0.5 bg-[var(--surface-raised)] text-[var(--text-secondary)] rounded-lg text-[12px]"
                  >
                    <span className="max-w-[150px] truncate">
                      {doc ? doc.filename : id.slice(0, 8)}
                    </span>
                    <button
                      type="button"
                      onClick={() => removeDoc(id)}
                      className="hover:text-[var(--text-primary)] transition-colors"
                    >
                      <X size={12} strokeWidth={1.5} />
                    </button>
                  </span>
                );
              })}
              <button
                type="button"
                onClick={clearDocSelection}
                className="text-[12px] text-[var(--text-muted)] hover:text-[var(--text-primary)] underline ml-1"
              >
                Clear all
              </button>
            </div>
          )}

          {/* Conversation */}
          <div className="space-y-4 min-h-[160px]">
            {messages.length === 0 && !sse.isStreaming && (
              <div className="text-center py-16 text-[var(--text-muted)]">
                <MessageSquare size={32} strokeWidth={1.5} className="mx-auto mb-3 opacity-40" />
                <p className="text-[14px] text-[var(--text-secondary)]">Ask a question about your documents</p>
                <p className="text-[12px] mt-1">
                  Retrieval, reranking, and verification happen automatically
                </p>
              </div>
            )}

            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}
              >
                {/* Speaker */}
                <div className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)] mb-1 px-1">
                  {m.role === "user" ? (
                    <>
                      <span>You</span>
                      <User size={12} strokeWidth={1.5} />
                      <span>{m.timestamp}</span>
                    </>
                  ) : (
                    <>
                      <span className="text-[var(--text-secondary)] font-medium">Assistant</span>
                      <span>{m.completed?.model || activeModel.name}</span>
                      <span>{m.timestamp}</span>
                    </>
                  )}
                </div>

                {/* Bubble */}
                <div
                  className={`max-w-[85%] rounded-xl p-4 text-[13px] ${
                    m.role === "user"
                      ? "bg-[var(--accent)] text-white rounded-tr-sm"
                      : "bg-[var(--surface-card)] text-[var(--text-primary)] border border-[var(--border)] rounded-tl-sm space-y-3"
                  }`}
                >
                  {/* Contextualized query */}
                  {m.contextualizedQuery && (
                    <div className="flex items-center gap-1.5 text-[11px] px-2.5 py-1.5 rounded-lg bg-[var(--surface-raised)] text-[var(--accent-fg)] font-mono">
                      <Search size={11} strokeWidth={1.5} />
                      <span>Reformulated: "{m.contextualizedQuery}"</span>
                    </div>
                  )}

                  {/* HyDE expansions */}
                  {m.expandedQueries && m.expandedQueries.length > 1 && (
                    <div className="flex flex-wrap items-center gap-1 text-[11px] px-2.5 py-1.5 rounded-lg bg-[var(--surface-raised)] text-[var(--text-secondary)]">
                      <span className="font-medium">Expanded:</span>
                      {m.expandedQueries.slice(1).map((eq, i) => (
                        <span
                          key={i}
                          className="px-1.5 py-0.5 rounded bg-[var(--surface-deep)] font-mono text-[10px] truncate max-w-[200px]"
                          title={eq}
                        >
                          {eq}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* CRAG correction */}
                  {m.wasCorrected && (
                    <div className="flex items-center gap-1.5 text-[11px] px-2.5 py-1.5 rounded-lg bg-[var(--positive)]/10 text-[var(--positive)] border border-[var(--positive)]/20">
                      <CheckCircle2 size={13} strokeWidth={1.5} className="shrink-0" />
                      <span>
                        Self-corrected: {m.correctionReason || "Re-retrieved with broader scope"}
                      </span>
                    </div>
                  )}

                  {/* Insufficient context */}
                  {m.has_sufficient_context === false && (
                    <div className="flex items-center gap-2 p-2.5 bg-[var(--warning)]/10 text-[var(--warning)] rounded-lg text-[12px]">
                      <AlertCircle size={14} strokeWidth={1.5} />
                      <span>
                        Insufficient context.{m.fallback_reason && ` ${m.fallback_reason}`}
                      </span>
                    </div>
                  )}

                  {/* Message content */}
                  {m.role === "user" ? (
                    <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
                  ) : (
                    <div className="markdown-body max-w-none">
                      {renderAnswer(m.content, m.citations)}
                    </div>
                  )}

                  {/* Verification claims */}
                  {m.role === "assistant" && m.verification?.claims && m.verification.claims.length > 0 && (
                    <div className="pt-3 border-t border-[var(--border)] space-y-2">
                      <div className="flex items-center justify-between text-[11px] text-[var(--text-muted)]">
                        <span className="font-medium">Verification</span>
                        {m.confidence?.overall_score != null && (
                          <ConfidenceBadge score={m.confidence.overall_score} />
                        )}
                      </div>
                      <div className="space-y-1">
                        {m.verification.claims.map((c, i) => (
                          <div key={i} className="flex items-start gap-2 text-[12px]">
                            <span
                              className={`shrink-0 px-1.5 py-0.5 rounded font-mono font-medium text-[10px] ${
                                c.verdict === "SUPPORTED"
                                  ? "bg-[var(--positive)]/10 text-[var(--positive)]"
                                  : c.verdict === "UNSUPPORTED"
                                    ? "bg-[var(--negative)]/10 text-[var(--negative)]"
                                    : "bg-[var(--warning)]/10 text-[var(--warning)]"
                              }`}
                            >
                              {c.verdict}
                            </span>
                            <span className="text-[var(--text-secondary)]">{c.claim_text}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Telemetry and actions */}
                  {m.role === "assistant" && (
                    <div className="pt-2 border-t border-[var(--border)] flex flex-wrap items-center justify-between gap-2 text-[11px]">
                      {m.completed && (
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="font-mono tabular-nums text-[var(--text-muted)]">
                            {m.completed.total_ms ??
                              Math.round((m.completed.retrieval_ms ?? 0) + (m.completed.generation_ms ?? 0))}ms
                          </span>
                          {m.completed.cached && (
                            <span className="text-[10px] px-1 py-0.5 rounded bg-[var(--surface-raised)] text-[var(--positive)] font-medium">
                              cached
                            </span>
                          )}
                          {m.completed.stages && Object.keys(m.completed.stages).length > 0 && (
                            <div className="hidden sm:flex flex-wrap items-center gap-1 text-[var(--text-muted)]">
                              {Object.entries(m.completed.stages).map(([stg, ms]) => (
                                <span
                                  key={stg}
                                  className="px-1.5 py-0.5 rounded bg-[var(--surface-raised)] text-[10px] font-mono tabular-nums"
                                >
                                  {stg} {ms}ms
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      <div className="flex items-center gap-1 ml-auto">
                        <button
                          type="button"
                          onClick={() => handleCopy(m.id, m.content)}
                          className="p-1 text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-raised)] rounded-lg transition-colors"
                          title="Copy"
                        >
                          {copiedId === m.id ? (
                            <Check size={13} strokeWidth={1.5} className="text-[var(--positive)]" />
                          ) : (
                            <Copy size={13} strokeWidth={1.5} />
                          )}
                        </button>
                        <button
                          type="button"
                          onClick={() => handleExportMarkdown(m)}
                          className="p-1 text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-raised)] rounded-lg transition-colors"
                          title="Export"
                        >
                          <Download size={13} strokeWidth={1.5} />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Streaming state */}
            {sse.isStreaming && (
              <div className="flex flex-col items-start space-y-2">
                <div className="flex items-center gap-1.5 text-[11px] text-[var(--text-muted)] px-1">
                  <Loader2 size={13} strokeWidth={1.5} className="animate-spin text-[var(--accent-fg)]" />
                  <span className="text-[var(--text-secondary)]">
                    Searching ({activeModel.name})
                  </span>
                </div>

                <div className="w-full max-w-[85%] bg-[var(--surface-card)] border border-[var(--border)] rounded-xl rounded-tl-sm p-4 space-y-3">
                  <div className="flex flex-wrap gap-1.5">
                    {STAGE_ORDER.map((s) => {
                      const idx = sse.stages.indexOf(s);
                      const nextIdx = sse.stages.indexOf(STAGE_ORDER[STAGE_ORDER.indexOf(s) + 1]);
                      const done = idx !== -1 && (nextIdx !== -1 || !sse.isStreaming);
                      const active = idx !== -1 && !done;
                      if (idx === -1 && !sse.stages.includes(s)) {
                        if (s === "contextualize" && messages.length === 0) return null;
                        if (s === "expansion" && !queryExpansion) return null;
                        if (s === "corrective_fallback") return null;
                        if (s === "rerank" && !selectedMode.rerank) return null;
                        if (s === "sparse" && mode === "dense") return null;
                        if (s === "rrf" && mode === "dense") return null;
                      }
                      return <StageBadge key={s} name={s} active={active} done={done} />;
                    })}
                  </div>

                  {sse.answer && (sse.answer as { answer?: string }).answer ? (
                    <div className="markdown-body max-w-none">
                      {renderAnswer(
                        (sse.answer as { answer: string }).answer,
                        (sse.answer as { citations?: ChatMessage["citations"] }).citations,
                      )}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-[12px] text-[var(--text-muted)] py-2">
                      <Loader2 size={14} strokeWidth={1.5} className="animate-spin text-[var(--accent-fg)]" />
                      <span>Retrieving passages...</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Stream error */}
            {sse.error && !sse.isStreaming && (
              <div className="flex items-center gap-2 p-3 bg-[var(--negative)]/10 text-[var(--negative)] rounded-xl text-[12px]">
                <AlertCircle size={16} strokeWidth={1.5} />
                <span>{sse.error}</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <form onSubmit={handleSubmit} className="sticky bottom-4 z-10">
            <div className="flex gap-2 p-1.5 bg-[var(--surface-card)] rounded-xl border border-[var(--border)] shadow-lg">
              <div className="flex-1 relative flex items-center">
                <Search size={16} strokeWidth={1.5} className="absolute left-3 text-[var(--text-muted)]" />
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder={
                    messages.length > 0
                      ? "Ask a follow-up..."
                      : "Ask about your documents..."
                  }
                  className="w-full pl-9 pr-4 py-2.5 bg-transparent text-[13px] focus:outline-none text-[var(--text-primary)] placeholder:text-[var(--text-muted)]"
                />
              </div>
              <button
                type="submit"
                disabled={sse.isStreaming || !question.trim()}
                className="px-5 py-2.5 bg-[var(--accent)] text-white rounded-lg text-[13px] font-medium hover:opacity-90 disabled:opacity-50 transition-colors flex items-center gap-1.5 cursor-pointer"
              >
                {sse.isStreaming ? (
                  <>
                    <Loader2 size={14} strokeWidth={1.5} className="animate-spin" />
                    Searching
                  </>
                ) : (
                  "Ask"
                )}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Citation inspector drawer */}
      {inspectCitation && (
        <aside className="fixed top-0 right-0 bottom-0 w-full sm:w-[400px] lg:w-[440px] bg-[var(--surface)] border-l border-[var(--border)] z-30 flex flex-col">
          {/* Drawer header */}
          <div className="flex items-center justify-between px-5 h-14 border-b border-[var(--border)] shrink-0">
            <div className="flex items-center gap-2">
              <Layers size={16} strokeWidth={1.5} className="text-[var(--text-muted)]" />
              <h3 className="font-medium text-[14px] flex items-center gap-2">
                Citation
                <span className="px-1.5 py-0.5 rounded bg-[var(--surface-raised)] text-[var(--accent-fg)] font-mono text-[12px]">
                  {inspectCitation.citation_id}
                </span>
              </h3>
            </div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() =>
                  handleCopy(
                    "inspect-chunk",
                    citationContextData?.target_chunk?.content ||
                      inspectCitation.evidence_quote ||
                      inspectCitation.chunk_id,
                  )
                }
                className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-raised)] transition-colors cursor-pointer"
                title="Copy chunk text"
              >
                {copiedId === "inspect-chunk" ? (
                  <Check size={15} strokeWidth={1.5} className="text-[var(--positive)]" />
                ) : (
                  <Copy size={15} strokeWidth={1.5} />
                )}
              </button>
              <button
                type="button"
                onClick={() => { setInspectCitation(null); setCitationContextData(null); }}
                className="p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--surface-raised)] transition-colors cursor-pointer"
                title="Close"
              >
                <X size={16} strokeWidth={1.5} />
              </button>
            </div>
          </div>

          {/* Drawer body */}
          <div className="flex-1 overflow-y-auto p-5 space-y-4 text-[12px]">
            {/* Document info */}
            <div className="p-3 rounded-xl border border-[var(--border)] bg-[var(--surface-card)] space-y-2">
              <div className="flex items-center gap-2">
                <FileText size={14} strokeWidth={1.5} className="text-[var(--text-muted)] shrink-0" />
                <span className="font-medium text-[13px] truncate flex-1" title={citationContextData?.document?.filename}>
                  {citationContextData?.document?.filename || "Source document"}
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--surface-raised)] text-[var(--text-muted)]">
                  {citationContextData?.document?.file_type || "doc"}
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-2 text-[11px] text-[var(--text-muted)] font-mono">
                {citationContextData?.target_chunk && (
                  <span className="px-2 py-0.5 rounded bg-[var(--surface-raised)] tabular-nums">
                    Chunk {citationContextData.target_chunk.index + 1} of {citationContextData.document.total_chunks}
                  </span>
                )}
                {citationContextData?.target_chunk?.token_count != null && (
                  <span className="px-2 py-0.5 rounded bg-[var(--surface-raised)] tabular-nums">
                    {citationContextData.target_chunk.token_count} tokens
                  </span>
                )}
                {citationContextData?.document?.chunking_strategy && (
                  <span className="px-2 py-0.5 rounded bg-[var(--surface-raised)]">
                    {citationContextData.document.chunking_strategy}
                  </span>
                )}
              </div>

              <div className="text-[10px] text-[var(--text-muted)] font-mono truncate">
                {inspectCitation.chunk_id}
              </div>
            </div>

            {/* Tab switcher */}
            <div className="flex rounded-lg bg-[var(--surface-card)] p-1 border border-[var(--border)]">
              <button
                type="button"
                onClick={() => setActiveInspectorTab("target")}
                className={`flex-1 py-1.5 rounded-md text-[12px] font-medium transition-colors cursor-pointer ${
                  activeInspectorTab === "target"
                    ? "bg-[var(--surface-raised)] text-[var(--text-primary)]"
                    : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                }`}
              >
                Evidence
              </button>
              <button
                type="button"
                onClick={() => setActiveInspectorTab("adjacent")}
                className={`flex-1 py-1.5 rounded-md text-[12px] font-medium transition-colors cursor-pointer flex items-center justify-center gap-1.5 ${
                  activeInspectorTab === "adjacent"
                    ? "bg-[var(--surface-raised)] text-[var(--text-primary)]"
                    : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
                }`}
              >
                <span>Adjacent</span>
                {citationLoading && <Loader2 size={11} strokeWidth={1.5} className="animate-spin" />}
              </button>
            </div>

            {/* Target evidence */}
            {activeInspectorTab === "target" && (
              <div className="space-y-3">
                {inspectCitation.evidence_quote && (
                  <div className="space-y-1.5">
                    <span className="text-[11px] font-medium text-[var(--text-muted)]">
                      Cited quote
                    </span>
                    <div className="p-3 rounded-lg border-l-2 border-[var(--accent)] bg-[var(--surface-card)] text-[var(--text-primary)] leading-relaxed text-[12px]">
                      "{inspectCitation.evidence_quote}"
                    </div>
                  </div>
                )}

                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-[11px] text-[var(--text-muted)]">
                    <span className="font-medium">Full chunk</span>
                    {citationContextData?.target_chunk?.section_heading && (
                      <span className="text-[var(--accent-fg)] font-mono truncate max-w-[200px]">
                        {citationContextData.target_chunk.section_heading}
                      </span>
                    )}
                  </div>
                  <div className="p-3 rounded-xl border border-[var(--border)] bg-[var(--surface-card)] leading-relaxed text-[var(--text-secondary)] whitespace-pre-wrap font-mono text-[11px]">
                    {citationContextData?.target_chunk?.content ||
                      inspectCitation.evidence_quote ||
                      "Loading..."}
                  </div>
                </div>
              </div>
            )}

            {/* Adjacent chunks */}
            {activeInspectorTab === "adjacent" && (
              <div className="space-y-4">
                {citationLoading ? (
                  <div className="py-8 text-center text-[var(--text-muted)] space-y-2">
                    <Loader2 size={20} strokeWidth={1.5} className="animate-spin mx-auto text-[var(--accent-fg)]" />
                    <p>Loading adjacent passages...</p>
                  </div>
                ) : (
                  <>
                    {/* Previous */}
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-1.5 text-[11px] font-medium text-[var(--text-muted)]">
                        <ChevronLeft size={13} strokeWidth={1.5} />
                        <span>
                          Previous{" "}
                          {citationContextData?.previous_chunk
                            ? `(#${citationContextData.previous_chunk.index + 1})`
                            : "(start of document)"}
                        </span>
                      </div>
                      {citationContextData?.previous_chunk ? (
                        <div className="p-3 rounded-xl border border-[var(--border)] bg-[var(--surface-card)] text-[var(--text-muted)] whitespace-pre-wrap font-mono text-[11px] leading-relaxed max-h-44 overflow-y-auto">
                          {citationContextData.previous_chunk.content}
                        </div>
                      ) : (
                        <p className="text-[11px] text-[var(--text-muted)] pl-1">
                          No prior chunk. Opening section of document.
                        </p>
                      )}
                    </div>

                    {/* Target indicator */}
                    <div className="p-2.5 rounded-lg border border-[var(--accent)]/20 bg-[var(--accent)]/5 text-[12px] text-[var(--accent-fg)] font-medium flex items-center justify-between">
                      <span>Cited chunk (#{(citationContextData?.target_chunk?.index ?? 0) + 1})</span>
                      <span className="font-mono text-[11px]">current</span>
                    </div>

                    {/* Next */}
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-1.5 text-[11px] font-medium text-[var(--text-muted)]">
                        <ChevronRight size={13} strokeWidth={1.5} />
                        <span>
                          Next{" "}
                          {citationContextData?.next_chunk
                            ? `(#${citationContextData.next_chunk.index + 1})`
                            : "(end of document)"}
                        </span>
                      </div>
                      {citationContextData?.next_chunk ? (
                        <div className="p-3 rounded-xl border border-[var(--border)] bg-[var(--surface-card)] text-[var(--text-muted)] whitespace-pre-wrap font-mono text-[11px] leading-relaxed max-h-44 overflow-y-auto">
                          {citationContextData.next_chunk.content}
                        </div>
                      ) : (
                        <p className="text-[11px] text-[var(--text-muted)] pl-1">
                          No subsequent chunk. End of document.
                        </p>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </aside>
      )}
    </div>
  );
}
