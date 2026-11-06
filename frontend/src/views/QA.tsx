import { useQuery } from "@tanstack/react-query";
import {
  AlertCircle,
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Copy,
  Cpu,
  Download,
  FileText,
  Filter,
  Layers,
  Loader2,
  MessageSquare,
  RotateCcw,
  Search,
  Shield,
  ShieldCheck,
  Sparkles,
  User,
  X,
  Zap,
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
  badge: string;
  description: string;
}

const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: "gemini-3.5-flash-lite",
    name: "Gemini 3.5 Flash Lite",
    provider: "gemini",
    group: "Google Gemini",
    badge: "Default",
    description: "Ultra-fast, lowest latency, high accuracy",
  },
  {
    id: "gemini-3.8-flash",
    name: "Gemini 3.8 Flash",
    provider: "gemini",
    group: "Google Gemini",
    badge: "Balanced",
    description: "Balanced reasoning with sub-second response",
  },
  {
    id: "gemini-2.5-flash",
    name: "Gemini 2.5 Flash",
    provider: "gemini",
    group: "Google Gemini",
    badge: "Flash",
    description: "Production-proven low latency workhorse",
  },
  {
    id: "gemini-3.1-pro-preview",
    name: "Gemini 3.1 Pro Preview",
    provider: "gemini",
    group: "Google Gemini",
    badge: "Pro Preview",
    description: "Deep reasoning & citation scrutiny",
  },
  {
    id: "llama-3.3-70b-versatile",
    name: "Llama 3.3 70B (Groq)",
    provider: "groq",
    group: "Groq Cloud",
    badge: "Groq LPU",
    description: "Fast 70B open weights reasoning via Groq",
  },
  {
    id: "llama-3.1-8b-instant",
    name: "Llama 3.1 8B (Groq)",
    provider: "groq",
    group: "Groq Cloud",
    badge: "Instant",
    description: "Sub-100ms ultra-rapid 8B generation",
  },
  {
    id: "ollama/llama3.2",
    name: "Llama 3.2 (Local)",
    provider: "ollama",
    group: "Local Ollama",
    badge: "Offline",
    description: "Runs privately on localhost:11434",
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
    // Gracefully handle storage quota
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
      className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold"
      style={{ backgroundColor: `${color}20`, color }}
    >
      <Shield size={13} />
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

  // Side-by-Side Split-Screen Citation Inspector State
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

  // Fetch adjacent chunk context when a citation is selected
  useEffect(() => {
    const chunkId = inspectCitation?.chunk_id;
    if (!chunkId) {
      return;
    }

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

    return () => {
      isMounted = false;
    };
  }, [inspectCitation?.chunk_id]);

  // Close dropdowns on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setDropdownOpen(false);
      }
      if (
        modelDropdownRef.current &&
        !modelDropdownRef.current.contains(event.target as Node)
      ) {
        setModelDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Auto-scroll when messages or streaming status update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sse.isStreaming, sse.stages]);

  // Append completed assistant message into conversation thread
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
        (completedData as { contextualized_query?: string | null })?.contextualized_query ||
        undefined,
      expandedQueries:
        (completedData as { expanded_queries?: string[] | null })?.expanded_queries ||
        undefined,
      wasCorrected: (completedData as { was_corrected?: boolean })?.was_corrected,
      correctionReason:
        (completedData as { correction_reason?: string | null })?.correction_reason ||
        undefined,
      citations: answerData?.citations || [],
      verification: sse.verification as ChatMessage["verification"],
      confidence: sse.confidence as ChatMessage["confidence"],
      completed: completedData || undefined,
      has_sufficient_context: answerData?.has_sufficient_context,
      fallback_reason: answerData?.fallback_reason,
      error: sse.error || undefined,
    };

    setMessages((prev) => [...prev, assistantMsg]);
  }, [
    sse.isStreaming,
    sse.answer,
    sse.verification,
    sse.confidence,
    sse.completed,
    sse.error,
    sse.traceId,
  ]);

  // Persist session state
  useEffect(() => {
    if (sse.isStreaming) return;

    if (messages.length > 0 || question.trim() !== "") {
      saveSessionState({
        question,
        mode,
        alpha,
        queryExpansion,
        selectedDocIds,
        selectedModel,
        selectedProvider,
        messages,
      });
    }
  }, [
    messages,
    question,
    mode,
    alpha,
    queryExpansion,
    selectedDocIds,
    selectedModel,
    selectedProvider,
    sse.isStreaming,
  ]);

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
  const activeModel =
    AVAILABLE_MODELS.find((m) => m.id === selectedModel) || AVAILABLE_MODELS[0];

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

    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setQuestion("");

    // Build chat history for conversational contextualization
    const chatHistory = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

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

    let md = `# QueryFusion Multi-Turn Q&A Session\n\n`;
    md += `**Date:** ${new Date().toISOString()}\n`;
    md += `**Model:** ${activeModel.name} (${activeModel.provider})\n`;
    md += `**Retrieval Mode:** ${mode} (Alpha: ${alpha})\n`;
    if (queryExpansion) md += `**Query Expansion (HyDE):** Enabled\n`;
    md += `\n---\n\n`;

    targetMessages.forEach((m, idx) => {
      const speaker = m.role === "user" ? "### User" : `### Assistant (${activeModel.name})`;
      md += `${speaker} [${m.timestamp}]\n\n${m.content}\n\n`;

      if (m.contextualizedQuery) {
        md += `> **Reformulated Search Query:** ${m.contextualizedQuery}\n\n`;
      }
      if (m.expandedQueries && m.expandedQueries.length > 1) {
        md += `> **Expanded Queries:** ${m.expandedQueries.join(", ")}\n\n`;
      }
      if (m.wasCorrected) {
        md += `> **Self-Correction (CRAG):** ${m.correctionReason || "Applied"}\n\n`;
      }

      if (m.verification?.claims && m.verification.claims.length > 0) {
        md += `**Citation Verification:**\n`;
        m.verification.claims.forEach((c) => {
          md += `- [${c.verdict}] ${c.claim_text}\n`;
        });
        md += `\n`;
      }

      if (m.citations && m.citations.length > 0) {
        md += `**Citations Used:**\n`;
        m.citations.forEach((c) => {
          md += `- [${c.citation_id}] Chunk: \`${c.chunk_id}\` quote: "${c.evidence_quote}"\n`;
        });
        md += `\n`;
      }

      if (idx < targetMessages.length - 1) {
        md += `---\n\n`;
      }
    });

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `queryfusion_chat_${Date.now()}.md`;
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
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `queryfusion_chat_${Date.now()}.json`;
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
    <div className="relative min-h-[calc(100vh-80px)] flex">
      {/* Main Chat View Container (dynamically accommodates the side drawer) */}
      <div
        className={`flex-1 transition-all duration-300 ${
          inspectCitation ? "lg:mr-[440px]" : ""
        }`}
      >
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="text-[var(--color-primary)]" size={22} />
              <div>
                <h2 className="text-xl font-semibold">Conversational Q&A</h2>
                <p className="text-xs text-[var(--color-text-tertiary)]">
                  Hybrid RAG with multi-query expansion, CRAG self-correction, & citation inspection
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {messages.length > 0 && (
                <>
                  <button
                    type="button"
                    onClick={() => handleExportMarkdown()}
                    className="text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] px-2.5 py-1.5 rounded-lg border border-[var(--color-border)] hover:bg-[var(--color-bg-secondary)] transition-colors flex items-center gap-1.5 cursor-pointer"
                    title="Export conversation as Markdown"
                  >
                    <Download size={13} />
                    <span>Export Chat</span>
                  </button>
                  <button
                    type="button"
                    onClick={handleExportJSON}
                    className="text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text)] px-2 py-1.5 rounded-lg border border-[var(--color-border)] hover:bg-[var(--color-bg-secondary)] transition-colors flex items-center cursor-pointer font-mono font-semibold"
                    title="Export conversation as JSON"
                  >
                    JSON
                  </button>
                  <button
                    type="button"
                    onClick={handleResetSession}
                    className="text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-danger)] px-2.5 py-1.5 rounded-lg border border-[var(--color-border)] hover:bg-[var(--color-bg-secondary)] transition-colors flex items-center gap-1.5 cursor-pointer"
                    title="Clear conversation and reset memory"
                  >
                    <RotateCcw size={13} />
                    <span>New Chat</span>
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Control Bar: Retrieval Mode, Model Selector, Query Expansion, & Document Scope */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-xl">
            {/* Retrieval Strategy Selector */}
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs font-medium text-[var(--color-text-tertiary)] mr-1">
                Search:
              </span>
              {MODES.map((m) => (
                <button
                  key={m.value}
                  type="button"
                  onClick={() => setMode(m.value)}
                  className={`px-2.5 py-1 text-xs rounded-lg border transition-colors ${
                    mode === m.value
                      ? "bg-[var(--color-primary)] text-white border-[var(--color-primary)] font-medium"
                      : "border-[var(--color-border)] hover:border-[var(--color-border-hover)] text-[var(--color-text-secondary)]"
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              {/* Query Expansion (HyDE) Toggle */}
              <button
                type="button"
                onClick={() => setQueryExpansion((prev) => !prev)}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors cursor-pointer ${
                  queryExpansion
                    ? "bg-[var(--color-primary)] text-white border-[var(--color-primary)] shadow-xs"
                    : "bg-[var(--color-bg)] border-[var(--color-border)] text-[var(--color-text-secondary)] hover:border-[var(--color-border-hover)]"
                }`}
                title="Generate alternate query variations and a hypothetical passage (HyDE) to maximize retrieval recall"
              >
                <Sparkles
                  size={13}
                  className={queryExpansion ? "text-amber-300 fill-amber-300" : ""}
                />
                <span>HyDE Expansion</span>
                <span
                  className={`text-[9px] px-1 py-0.2 rounded font-semibold uppercase ${
                    queryExpansion
                      ? "bg-white/20 text-white"
                      : "bg-[var(--color-bg-tertiary)] text-[var(--color-text-tertiary)]"
                  }`}
                >
                  {queryExpansion ? "ON" : "OFF"}
                </span>
              </button>

              {/* Model Provider Selector Dropdown */}
              <div className="relative" ref={modelDropdownRef}>
                <button
                  type="button"
                  onClick={() => setModelDropdownOpen((prev) => !prev)}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] hover:border-[var(--color-border-hover)] text-[var(--color-text)] transition-colors cursor-pointer"
                  title="Select LLM generation model and provider"
                >
                  <Cpu size={13} className="text-[var(--color-primary)]" />
                  <span>{activeModel.name}</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--color-primary)]/10 text-[var(--color-primary)] uppercase font-semibold">
                    {activeModel.badge}
                  </span>
                  <ChevronDown size={12} className="text-[var(--color-text-tertiary)]" />
                </button>

                {modelDropdownOpen && (
                  <div className="absolute right-0 mt-1 w-80 bg-[var(--color-bg)] border border-[var(--color-border)] rounded-xl shadow-xl z-30 p-2 text-xs space-y-2">
                    <div className="px-2 py-1 text-[11px] font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider border-b border-[var(--color-border)] pb-1.5">
                      LLM Provider & Model
                    </div>

                    <div className="max-h-72 overflow-y-auto space-y-1">
                      {(["Google Gemini", "Groq Cloud", "Local Ollama"] as const).map(
                        (groupName) => {
                          const groupModels = AVAILABLE_MODELS.filter(
                            (m) => m.group === groupName,
                          );
                          return (
                            <div key={groupName} className="space-y-0.5">
                              <div className="px-2 pt-1.5 pb-0.5 text-[10px] font-bold text-[var(--color-text-tertiary)] uppercase">
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
                                    className={`w-full text-left px-2 py-1.5 rounded-lg flex items-start justify-between gap-2 transition-colors cursor-pointer ${
                                      isSelected
                                        ? "bg-[var(--color-primary)]/10 text-[var(--color-primary)] font-medium"
                                        : "hover:bg-[var(--color-bg-secondary)] text-[var(--color-text)]"
                                    }`}
                                  >
                                    <div className="flex-1 truncate">
                                      <div className="flex items-center gap-1.5">
                                        <span>{m.name}</span>
                                        <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--color-bg-tertiary)] text-[var(--color-text-tertiary)] uppercase font-semibold">
                                          {m.badge}
                                        </span>
                                      </div>
                                      <div className="text-[10px] text-[var(--color-text-tertiary)] truncate">
                                        {m.description}
                                      </div>
                                    </div>
                                    {isSelected && <Check size={14} className="shrink-0 mt-0.5" />}
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

              {/* Document Scope Dropdown */}
              {docs.length > 0 && (
                <div className="relative" ref={dropdownRef}>
                  <button
                    type="button"
                    onClick={() => setDropdownOpen((prev) => !prev)}
                    className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                      selectedDocIds.length > 0
                        ? "bg-[var(--color-primary)]/10 text-[var(--color-primary)] border-[var(--color-primary)]/30 hover:bg-[var(--color-primary)]/20"
                        : "bg-[var(--color-bg)] border-[var(--color-border)] text-[var(--color-text-secondary)] hover:border-[var(--color-border-hover)]"
                    }`}
                  >
                    <Filter size={13} />
                    <span>
                      {selectedDocIds.length === 0
                        ? `Scope: All (${docs.length})`
                        : `Scope: ${selectedDocIds.length}`}
                    </span>
                    <ChevronDown size={12} />
                  </button>

                  {dropdownOpen && (
                    <div className="absolute right-0 mt-1 w-72 bg-[var(--color-bg)] border border-[var(--color-border)] rounded-xl shadow-xl z-30 p-2 text-xs space-y-1">
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
                              className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-[var(--color-bg-secondary)] cursor-pointer text-[var(--color-text)] select-none"
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
          </div>

          {/* Selected Document Chips */}
          {selectedDocIds.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
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
            <div className="flex items-center gap-4 px-3 py-2 bg-[var(--color-bg-secondary)]/50 rounded-xl border border-[var(--color-border)]">
              <label className="text-xs font-medium text-[var(--color-text-secondary)] w-28">
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
              <span className="text-xs text-[var(--color-text-tertiary)] w-28 text-right font-mono">
                Dense ← → BM25
              </span>
            </div>
          )}

          {/* Conversation Thread */}
          <div className="space-y-4 min-h-[160px]">
            {messages.length === 0 && !sse.isStreaming && (
              <div className="text-center py-12 px-4 rounded-2xl border border-dashed border-[var(--color-border)] bg-[var(--color-bg-secondary)]/30 space-y-3">
                <div className="w-12 h-12 rounded-full bg-[var(--color-primary)]/10 text-[var(--color-primary)] flex items-center justify-center mx-auto">
                  <Sparkles size={24} />
                </div>
                <h3 className="font-semibold text-[var(--color-text)]">
                  Ask anything about your ingested documents
                </h3>
                <p className="text-xs text-[var(--color-text-secondary)] max-w-md mx-auto">
                  QueryFusion uses Hybrid Dense + Sparse BM25 retrieval, Cross-Encoder reranking,
                  HyDE query expansion, and Corrective RAG (CRAG) self-healing.
                </p>
              </div>
            )}

            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex flex-col ${
                  m.role === "user" ? "items-end" : "items-start"
                }`}
              >
                {/* Speaker Label */}
                <div className="flex items-center gap-1.5 text-[11px] text-[var(--color-text-tertiary)] mb-1 px-1">
                  {m.role === "user" ? (
                    <>
                      <span>You</span>
                      <User size={12} />
                      <span>• {m.timestamp}</span>
                    </>
                  ) : (
                    <>
                      <Bot size={13} className="text-[var(--color-primary)]" />
                      <span className="font-medium text-[var(--color-text)]">Assistant</span>
                      <span>• {m.completed?.model || activeModel.name}</span>
                      <span>• {m.timestamp}</span>
                    </>
                  )}
                </div>

                {/* Bubble Content */}
                <div
                  className={`max-w-[85%] rounded-2xl p-4 text-sm shadow-xs ${
                    m.role === "user"
                      ? "bg-[var(--color-primary)] text-white rounded-tr-xs"
                      : "bg-[var(--color-bg-secondary)] text-[var(--color-text)] border border-[var(--color-border)] rounded-tl-xs space-y-3"
                  }`}
                >
                  {/* Contextualized Query Tag */}
                  {m.contextualizedQuery && (
                    <div className="flex items-center gap-1 text-[11px] px-2.5 py-1 rounded-md bg-[var(--color-primary)]/10 text-[var(--color-primary)] font-mono">
                      <Search size={11} />
                      <span>Contextualized search: &ldquo;{m.contextualizedQuery}&rdquo;</span>
                    </div>
                  )}

                  {/* Multi-Query Expanded Tags */}
                  {m.expandedQueries && m.expandedQueries.length > 1 && (
                    <div className="flex flex-wrap items-center gap-1 text-[10px] px-2.5 py-1 rounded-md bg-amber-500/10 text-amber-700 dark:text-amber-300">
                      <Sparkles size={11} className="text-amber-500 shrink-0" />
                      <span className="font-semibold">HyDE Expansions:</span>
                      {m.expandedQueries.slice(1).map((eq, i) => (
                        <span
                          key={i}
                          className="px-1.5 py-0.2 rounded bg-amber-500/20 font-mono truncate max-w-[200px]"
                          title={eq}
                        >
                          {eq}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* CRAG Self-Correction Banner */}
                  {m.wasCorrected && (
                    <div className="flex items-center gap-1.5 text-[11px] px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border border-emerald-500/20">
                      <ShieldCheck size={13} className="text-emerald-500 shrink-0" />
                      <span>
                        Self-Corrected via CRAG:{" "}
                        <span className="font-medium italic">
                          {m.correctionReason || "Re-retrieved with broadened scope"}
                        </span>
                      </span>
                    </div>
                  )}

                  {/* Insufficient context warning */}
                  {m.has_sufficient_context === false && (
                    <div className="flex items-center gap-2 p-2.5 bg-[var(--color-warning)]/10 text-[var(--color-warning)] rounded-lg text-xs">
                      <AlertCircle size={15} />
                      <span>
                        Insufficient context.{" "}
                        {m.fallback_reason && <span>{m.fallback_reason}</span>}
                      </span>
                    </div>
                  )}

                  {/* Main message text */}
                  {m.role === "user" ? (
                    <div className="whitespace-pre-wrap">{m.content}</div>
                  ) : (
                    <div className="prose prose-sm max-w-none text-[var(--color-text)]">
                      {renderAnswer(m.content, m.citations)}
                    </div>
                  )}

                  {/* Verification Claims if Assistant */}
                  {m.role === "assistant" &&
                    m.verification?.claims &&
                    m.verification.claims.length > 0 && (
                      <div className="pt-2 border-t border-[var(--color-border)] space-y-1.5">
                        <div className="flex items-center justify-between text-[11px] font-semibold text-[var(--color-text-tertiary)] uppercase">
                          <span>Citation Verification</span>
                          {m.confidence?.overall_score != null && (
                            <ConfidenceBadge score={m.confidence.overall_score} />
                          )}
                        </div>
                        <div className="space-y-1">
                          {m.verification.claims.map((c, i) => (
                            <div key={i} className="flex items-start gap-2 text-xs">
                              <span
                                className={`shrink-0 px-1.5 py-0.5 rounded font-mono font-semibold text-[10px] ${
                                  c.verdict === "SUPPORTED"
                                    ? "bg-[var(--color-success)]/10 text-[var(--color-success)]"
                                    : c.verdict === "UNSUPPORTED"
                                      ? "bg-[var(--color-danger)]/10 text-[var(--color-danger)]"
                                      : "bg-[var(--color-warning)]/10 text-[var(--color-warning)]"
                                }`}
                              >
                                {c.verdict}
                              </span>
                              <span className="text-[var(--color-text-secondary)] text-[12px]">
                                {c.claim_text}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                  {/* Telemetry pill & Actions for Assistant */}
                  {m.role === "assistant" && (
                    <div className="pt-2 border-t border-[var(--color-border)] flex flex-wrap items-center justify-between gap-2 text-xs">
                      {m.completed && (
                        <div className="flex flex-wrap items-center gap-1.5">
                          <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[var(--color-primary)]/10 text-[var(--color-primary)] text-[11px] font-medium">
                            <Zap size={12} className="text-amber-500 fill-amber-500" />
                            <span>
                              {m.completed.total_ms ??
                                Math.round(
                                  (m.completed.retrieval_ms ?? 0) +
                                    (m.completed.generation_ms ?? 0),
                                )}{" "}
                              ms
                            </span>
                            {m.completed.cached && (
                              <span className="ml-1 text-[9px] px-1 py-0.2 rounded bg-emerald-500/10 text-emerald-600 font-semibold uppercase">
                                Cached
                              </span>
                            )}
                          </div>

                          {m.completed.stages &&
                            Object.keys(m.completed.stages).length > 0 && (
                              <div className="hidden sm:flex flex-wrap items-center gap-1 text-[var(--color-text-secondary)]">
                                {Object.entries(m.completed.stages).map(([stg, ms]) => (
                                  <span
                                    key={stg}
                                    className="px-1.5 py-0.5 rounded bg-[var(--color-bg-tertiary)] text-[10px] font-mono"
                                  >
                                    {stg}: {ms}ms
                                  </span>
                                ))}
                              </div>
                            )}
                        </div>
                      )}

                      <div className="flex items-center gap-1.5 ml-auto">
                        <button
                          type="button"
                          onClick={() => handleCopy(m.id, m.content)}
                          className="p-1 text-[var(--color-text-tertiary)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-tertiary)] rounded transition-colors"
                          title="Copy answer"
                        >
                          {copiedId === m.id ? (
                            <Check size={13} className="text-[var(--color-success)]" />
                          ) : (
                            <Copy size={13} />
                          )}
                        </button>
                        <button
                          type="button"
                          onClick={() => handleExportMarkdown(m)}
                          className="p-1 text-[var(--color-text-tertiary)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-tertiary)] rounded transition-colors"
                          title="Export this response as Markdown"
                        >
                          <Download size={13} />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Live Streaming Assistant State */}
            {sse.isStreaming && (
              <div className="flex flex-col items-start space-y-2">
                <div className="flex items-center gap-1.5 text-[11px] text-[var(--color-text-tertiary)] px-1">
                  <Bot size={13} className="text-[var(--color-primary)] animate-pulse" />
                  <span className="font-medium text-[var(--color-text)]">
                    Assistant is researching ({activeModel.name})...
                  </span>
                </div>

                <div className="w-full max-w-[85%] bg-[var(--color-bg-secondary)] border border-[var(--color-border)] rounded-2xl rounded-tl-xs p-4 space-y-3 shadow-xs">
                  {/* Stage Progress Badges */}
                  <div className="flex flex-wrap gap-1.5">
                    {STAGE_ORDER.map((s) => {
                      const idx = sse.stages.indexOf(s);
                      const nextIdx = sse.stages.indexOf(
                        STAGE_ORDER[STAGE_ORDER.indexOf(s) + 1],
                      );
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

                  {/* Streaming Partial Answer */}
                  {sse.answer && (sse.answer as { answer?: string }).answer ? (
                    <div className="prose prose-sm max-w-none text-[var(--color-text)]">
                      {renderAnswer(
                        (sse.answer as { answer: string }).answer,
                        (sse.answer as { citations?: ChatMessage["citations"] }).citations,
                      )}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-xs text-[var(--color-text-tertiary)] py-2">
                      <Loader2 size={14} className="animate-spin text-[var(--color-primary)]" />
                      <span>Retrieving passages & formulating response...</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Stream Error Display */}
            {sse.error && !sse.isStreaming && (
              <div className="flex items-center gap-2 p-3 bg-[var(--color-danger)]/10 text-[var(--color-danger)] rounded-xl text-xs">
                <AlertCircle size={16} />
                <span>{sse.error}</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Bar Form */}
          <form onSubmit={handleSubmit} className="sticky bottom-4 z-10 space-y-2">
            <div className="flex gap-2 p-1.5 bg-[var(--color-bg)]/80 backdrop-blur-md rounded-2xl border border-[var(--color-border)] shadow-lg">
              <div className="flex-1 relative flex items-center">
                <Search
                  size={18}
                  className="absolute left-3.5 text-[var(--color-text-tertiary)]"
                />
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder={
                    messages.length > 0
                      ? "Ask a follow-up question (context is automatically preserved)..."
                      : "Ask a question about your documents..."
                  }
                  className="w-full pl-10 pr-4 py-2.5 bg-transparent text-sm focus:outline-none text-[var(--color-text)] placeholder:text-[var(--color-text-tertiary)]"
                />
              </div>
              <button
                type="submit"
                disabled={sse.isStreaming || !question.trim()}
                className="px-5 py-2.5 bg-[var(--color-primary)] text-white rounded-xl text-xs font-semibold hover:bg-[var(--color-primary-hover)] disabled:opacity-50 transition-colors flex items-center gap-1.5 cursor-pointer shadow-xs"
              >
                {sse.isStreaming ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    <span>Searching</span>
                  </>
                ) : (
                  <span>Ask</span>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Side-by-Side Split-Screen Citation Inspector Drawer */}
      {inspectCitation && (
        <aside className="fixed top-16 right-0 bottom-0 w-full sm:w-[400px] lg:w-[440px] bg-[var(--color-bg)] border-l border-[var(--color-border)] shadow-2xl z-30 flex flex-col transition-transform animate-in slide-in-from-right duration-200">
          {/* Drawer Header */}
          <div className="flex items-center justify-between p-4 border-b border-[var(--color-border)] bg-[var(--color-bg-secondary)]/50">
            <div className="flex items-center gap-2">
              <Layers size={18} className="text-[var(--color-primary)]" />
              <div>
                <h3 className="font-semibold text-sm text-[var(--color-text)] flex items-center gap-1.5">
                  <span>Citation Inspector</span>
                  <span className="px-1.5 py-0.5 rounded bg-[var(--color-primary)]/10 text-[var(--color-primary)] font-mono text-xs font-bold">
                    {inspectCitation.citation_id}
                  </span>
                </h3>
                <p className="text-[10px] text-[var(--color-text-tertiary)]">
                  Verbatim evidence & adjacent document context
                </p>
              </div>
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
                className="p-1.5 rounded-lg text-[var(--color-text-tertiary)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-tertiary)] transition-colors cursor-pointer"
                title="Copy chunk text"
              >
                {copiedId === "inspect-chunk" ? (
                  <Check size={15} className="text-[var(--color-success)]" />
                ) : (
                  <Copy size={15} />
                )}
              </button>
              <button
                type="button"
                onClick={() => {
                  setInspectCitation(null);
                  setCitationContextData(null);
                }}
                className="p-1.5 rounded-lg text-[var(--color-text-tertiary)] hover:text-[var(--color-text)] hover:bg-[var(--color-bg-tertiary)] transition-colors cursor-pointer"
                title="Close drawer"
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Drawer Body (Scrollable) */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs">
            {/* Document Info Card */}
            <div className="p-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)] space-y-2">
              <div className="flex items-center gap-2">
                <FileText size={16} className="text-[var(--color-primary)] shrink-0" />
                <span className="font-semibold text-sm truncate flex-1" title={citationContextData?.document?.filename}>
                  {citationContextData?.document?.filename || "Source Document"}
                </span>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-[var(--color-bg-tertiary)] text-[var(--color-text-tertiary)] font-bold">
                  {citationContextData?.document?.file_type || "doc"}
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-2 text-[11px] text-[var(--color-text-secondary)] font-mono">
                {citationContextData?.target_chunk && (
                  <span className="px-2 py-0.5 rounded bg-[var(--color-bg-tertiary)]">
                    Chunk {citationContextData.target_chunk.index + 1} of{" "}
                    {citationContextData.document.total_chunks}
                  </span>
                )}
                {citationContextData?.target_chunk?.token_count != null && (
                  <span className="px-2 py-0.5 rounded bg-[var(--color-bg-tertiary)]">
                    {citationContextData.target_chunk.token_count} tokens
                  </span>
                )}
                {citationContextData?.document?.chunking_strategy && (
                  <span className="px-2 py-0.5 rounded bg-[var(--color-bg-tertiary)]">
                    {citationContextData.document.chunking_strategy}
                  </span>
                )}
              </div>

              <div className="text-[10px] text-[var(--color-text-tertiary)] font-mono truncate">
                ID: {inspectCitation.chunk_id}
              </div>
            </div>

            {/* Inspector Tab Switcher */}
            <div className="flex rounded-lg bg-[var(--color-bg-secondary)] p-1 border border-[var(--color-border)]">
              <button
                type="button"
                onClick={() => setActiveInspectorTab("target")}
                className={`flex-1 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                  activeInspectorTab === "target"
                    ? "bg-[var(--color-bg)] text-[var(--color-text)] shadow-xs"
                    : "text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                }`}
              >
                Target Evidence
              </button>
              <button
                type="button"
                onClick={() => setActiveInspectorTab("adjacent")}
                className={`flex-1 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer flex items-center justify-center gap-1.5 ${
                  activeInspectorTab === "adjacent"
                    ? "bg-[var(--color-bg)] text-[var(--color-text)] shadow-xs"
                    : "text-[var(--color-text-secondary)] hover:text-[var(--color-text)]"
                }`}
              >
                <span>Adjacent Context</span>
                {citationLoading && <Loader2 size={11} className="animate-spin" />}
              </button>
            </div>

            {/* Tab 1: Target Evidence Quote */}
            {activeInspectorTab === "target" && (
              <div className="space-y-3">
                {inspectCitation.evidence_quote ? (
                  <div className="space-y-1">
                    <span className="text-[11px] font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider">
                      Cited Grounding Quote
                    </span>
                    <div className="p-3 rounded-lg border-l-4 border-amber-500 bg-amber-500/10 text-[var(--color-text)] leading-relaxed text-xs">
                      &ldquo;{inspectCitation.evidence_quote}&rdquo;
                    </div>
                  </div>
                ) : null}

                <div className="space-y-1">
                  <div className="flex items-center justify-between text-[11px] font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider">
                    <span>Full Chunk Content</span>
                    {citationContextData?.target_chunk?.section_heading && (
                      <span className="text-[10px] text-[var(--color-primary)] font-mono truncate max-w-[200px]">
                        § {citationContextData.target_chunk.section_heading}
                      </span>
                    )}
                  </div>
                  <div className="p-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)]/50 leading-relaxed text-[var(--color-text)] whitespace-pre-wrap font-mono text-[11px]">
                    {citationContextData?.target_chunk?.content ||
                      inspectCitation.evidence_quote ||
                      "Loading chunk content..."}
                  </div>
                </div>
              </div>
            )}

            {/* Tab 2: Adjacent Chunks (Surrounding Narrative) */}
            {activeInspectorTab === "adjacent" && (
              <div className="space-y-4">
                {citationLoading ? (
                  <div className="py-8 text-center text-[var(--color-text-tertiary)] space-y-2">
                    <Loader2 size={20} className="animate-spin mx-auto text-[var(--color-primary)]" />
                    <p>Loading adjacent document passages...</p>
                  </div>
                ) : (
                  <>
                    {/* Previous Chunk */}
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-1.5 text-[11px] font-semibold text-[var(--color-text-tertiary)] uppercase">
                        <ChevronLeft size={13} />
                        <span>
                          Previous Chunk{" "}
                          {citationContextData?.previous_chunk
                            ? `(#${citationContextData.previous_chunk.index + 1})`
                            : "(Beginning of Document)"}
                        </span>
                      </div>
                      {citationContextData?.previous_chunk ? (
                        <div className="p-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)]/30 text-[var(--color-text-secondary)] whitespace-pre-wrap font-mono text-[11px] leading-relaxed max-h-44 overflow-y-auto">
                          {citationContextData.previous_chunk.content}
                        </div>
                      ) : (
                        <p className="text-[11px] text-[var(--color-text-tertiary)] italic pl-1">
                          No prior chunk exists. This is the opening section of the document.
                        </p>
                      )}
                    </div>

                    {/* Target Chunk Focus Indicator */}
                    <div className="p-2.5 rounded-lg border border-[var(--color-primary)]/30 bg-[var(--color-primary)]/10 text-xs text-[var(--color-primary)] font-medium flex items-center justify-between">
                      <span>Target Cited Chunk (#{((citationContextData?.target_chunk?.index ?? 0) + 1)})</span>
                      <span className="font-bold font-mono">ACTIVE EVIDENCE</span>
                    </div>

                    {/* Next Chunk */}
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-1.5 text-[11px] font-semibold text-[var(--color-text-tertiary)] uppercase">
                        <ChevronRight size={13} />
                        <span>
                          Next Chunk{" "}
                          {citationContextData?.next_chunk
                            ? `(#${citationContextData.next_chunk.index + 1})`
                            : "(End of Document)"}
                        </span>
                      </div>
                      {citationContextData?.next_chunk ? (
                        <div className="p-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-secondary)]/30 text-[var(--color-text-secondary)] whitespace-pre-wrap font-mono text-[11px] leading-relaxed max-h-44 overflow-y-auto">
                          {citationContextData.next_chunk.content}
                        </div>
                      ) : (
                        <p className="text-[11px] text-[var(--color-text-tertiary)] italic pl-1">
                          No subsequent chunk. This is the conclusion of the document.
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
