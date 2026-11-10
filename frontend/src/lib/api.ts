/**
 * Frontend API client for Hybrid-RAG platform.
 * Aligned with FastAPI backend routes and authentication contracts.
 */

// Base URL configuration (honors Vite environment if configured)
const BASE =
  typeof import.meta !== "undefined" && import.meta.env?.VITE_API_BASE_URL
    ? String(import.meta.env.VITE_API_BASE_URL).replace(/\/$/, "")
    : "";

/**
 * Structured API error containing HTTP status and backend detail message.
 */
export class ApiError extends Error {
  status: number;
  statusText: string;
  detail: string;
  data: unknown;

  constructor(
    status: number,
    statusText: string,
    detail: string,
    data?: unknown,
  ) {
    super(`${status} ${statusText}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.statusText = statusText;
    this.detail = detail;
    this.data = data;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);

  if (!res.ok) {
    const rawBody = await res.text();
    let detail = rawBody;
    let data: unknown = null;

    try {
      data = JSON.parse(rawBody);
      if (typeof data === "object" && data !== null && "detail" in data) {
        detail = String((data as { detail: unknown }).detail);
      }
    } catch {
      // Body is not JSON
    }

    if (!detail || detail === rawBody) {
      if (res.status === 413) {
        detail = "File exceeds the maximum upload size limit";
      } else if (res.status === 422) {
        detail = detail || "Storage quota or document limit exceeded";
      }
    }

    throw new ApiError(res.status, res.statusText, detail, data);
  }

  // Handle 204 No Content or empty bodies
  if (res.status === 204) {
    return undefined as unknown as T;
  }

  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return res.json();
  }
  return (await res.text()) as unknown as T;
}

export interface Document {
  id: string;
  filename: string;
  file_type: string;
  chunk_count: number;
  created_at: string;
  status?: string;
  error_message?: string;
  size_bytes?: number;
  chunking_strategy?: string;
  is_evictable?: boolean;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  file_type: string;
  chunks_created: number;
  chunking_strategy: string;
  status?: string;
  job_id?: string | null;
}

export interface Chunk {
  chunk_id: string;
  index: number;
  token_count: number;
  content: string;
  section_heading: string;
}

export interface TraceStage {
  stage: string;
  duration_ms: number;
  candidate_count: number;
  top_scores: { chunk_id: string; score: number; rank: number }[];
  metadata: Record<string, unknown>;
}

export interface Trace {
  trace_id: string;
  query: string;
  stages: TraceStage[];
  total_duration_ms: number;
}

export interface JobStatus {
  job_id: string;
  job_type: string;
  status: string;
  progress_percent: number;
  status_message: string;
  created_at: string;
  completed_at: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface ConsistencyResult {
  strategy: string;
  postgres_chunks: number;
  embedded_chunks: number;
  qdrant_points: number;
  consistent: boolean;
}

export interface StorageStats {
  used_bytes: number;
  used_mb: number;
  free_bytes: number;
  free_mb: number;
  max_storage_mb: number;
  max_storage_bytes: number;
  usage_percent: number;
  max_upload_size_mb: number;
  max_upload_size_bytes: number;
  document_count: number;
  max_documents_count: number;
  chunk_count: number;
}

export function fetchDocuments(): Promise<Document[]> {
  return request("/api/v1/documents");
}

export function fetchStorageStats(): Promise<StorageStats> {
  return request("/api/v1/documents/storage");
}

export interface DemoLoadResult {
  status: string;
  loaded_count: number;
  existing_count: number;
  created_documents: Array<{
    document_id: string;
    filename: string;
    chunks_created: number;
    chunking_strategy: string;
  }>;
  already_existing: string[];
}

export function loadDemoDocuments(): Promise<DemoLoadResult> {
  return request("/api/v1/documents/demo", { method: "POST" });
}

export function deleteDocument(id: string): Promise<void> {
  return request(`/api/v1/documents/${id}`, { method: "DELETE" });
}

export function batchDeleteDocuments(ids: string[]): Promise<void> {
  return request("/api/v1/documents/batch-delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_ids: ids }),
  });
}

export interface DocumentStatusResponse {
  document_id: string;
  filename: string;
  status: string;
  error_message?: string | null;
  chunk_count: number;
  chunking_strategy: string;
}

export function fetchDocumentStatus(documentId: string): Promise<DocumentStatusResponse> {
  return request(`/api/v1/documents/${documentId}/status`);
}

/**
 * Upload a document file to the backend for ingestion.
 *
 * Route: POST /api/v1/documents?chunking_strategy={strategy}&background={background}
 */
export async function uploadDocument(
  file: File,
  strategy: string,
  chunkSize: number,
  overlap: number,
  similarityThreshold: number,
  background: boolean = false,
): Promise<DocumentUploadResponse> {
  const normalizedStrategy =
    strategy === "structure_aware" ? "structure" : strategy;

  const form = new FormData();
  form.append("file", file);
  form.append("chunking_strategy", normalizedStrategy);
  form.append("chunk_size", String(chunkSize));
  form.append("overlap", String(overlap));
  form.append("similarity_threshold", String(similarityThreshold));

  const queryParams = new URLSearchParams({
    chunking_strategy: normalizedStrategy,
  });
  if (background) {
    queryParams.set("background", "true");
  }

  return request<DocumentUploadResponse>(
    `/api/v1/documents?${queryParams.toString()}`,
    {
      method: "POST",
      body: form,
    },
  );
}

/**
 * Retrieve chunks for an ingested document.
 * Returns empty array if document has no chunks or backend endpoint is missing (404).
 */
export async function fetchDocumentChunks(id: string): Promise<Chunk[]> {
  try {
    return await request<Chunk[]>(`/api/v1/documents/${id}/chunks`);
  } catch (err: unknown) {
    if (err instanceof ApiError && err.status === 404) {
      return [];
    }
    throw err;
  }
}

export function fetchTrace(traceId: string): Promise<Trace> {
  return request(`/api/traces/${traceId}`);
}

export function fetchJobStatus(jobId: string): Promise<JobStatus> {
  return request(`/api/jobs/${jobId}`);
}

export function cancelJob(
  jobId: string,
): Promise<{ job_id: string; status: string }> {
  return request(`/api/jobs/${jobId}/cancel`, { method: "POST" });
}

export function fetchConsistency(
  strategy: string = "fixed",
): Promise<ConsistencyResult> {
  const queryParams = new URLSearchParams({ strategy });
  return request(`/api/v1/documents/consistency?${queryParams.toString()}`);
}

export interface ChunkContextItem {
  chunk_id: string;
  index: number;
  token_count: number;
  content: string;
  section_heading: string;
}

export interface ChunkContextResponse {
  document: {
    id: string;
    filename: string;
    file_type: string;
    chunking_strategy: string;
    size_bytes: number;
    total_chunks: number;
  };
  target_chunk: ChunkContextItem;
  previous_chunk: ChunkContextItem | null;
  next_chunk: ChunkContextItem | null;
}

export function fetchChunkContext(chunkId: string): Promise<ChunkContextResponse> {
  return request(`/api/v1/documents/chunks/${chunkId}/context`);
}

export interface SSEEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface StreamQueryOptions {
  retrieval_strategy: string;
  alpha: number;
  rerank_enabled: boolean;
  verify: boolean;
  chunking_strategy?: string;
  temperature?: number;
  document_ids?: string[];
  chat_history?: Array<{ role: string; content: string }>;
  model?: string;
  provider?: string;
  query_expansion?: boolean;
  enable_crag?: boolean;
  adaptive_alpha?: boolean;
}

export function streamQuery(
  question: string,
  options: StreamQueryOptions,
  onEvent: (evt: SSEEvent) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): AbortController {
  const controller = new AbortController();

  fetch(`${BASE}/api/query/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question, ...options }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const bodyText = await res.text();
          try {
            const bodyJson = JSON.parse(bodyText);
            if (bodyJson && typeof bodyJson.detail === "string") {
              detail = bodyJson.detail;
            } else if (bodyJson && Array.isArray(bodyJson.detail)) {
              detail = bodyJson.detail
                .map((d: any) => d.msg || JSON.stringify(d))
                .join("; ");
            } else if (bodyText) {
              detail = bodyText;
            }
          } catch {
            if (bodyText) detail = bodyText;
          }
        } catch {
          // ignore stream read error on error response
        }

        throw new ApiError(res.status, res.statusText, detail);
      }

      if (!res.body) {
        throw new Error("Response body is not readable");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Normalize CRLF to LF to prevent trailing \r issues
        buffer = buffer.replace(/\r\n/g, "\n");

        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          if (!part.trim()) continue;
          let eventType = "message";
          const dataLines: string[] = [];

          for (const line of part.split("\n")) {
            const trimmedLine = line.trimEnd();
            if (trimmedLine.startsWith("event:")) {
              eventType = trimmedLine.slice(6).trim();
            } else if (trimmedLine.startsWith("data:")) {
              dataLines.push(trimmedLine.slice(5).trimStart());
            }
          }

          const rawData = dataLines.join("\n");
          try {
            const parsedData = rawData ? JSON.parse(rawData) : {};
            onEvent({ event: eventType, data: parsedData });
          } catch {
            // skip malformed data payload
          }
        }
      }

      if (buffer.trim()) {
        let eventType = "message";
        const dataLines: string[] = [];

        for (const line of buffer.split("\n")) {
          const trimmedLine = line.trimEnd();
          if (trimmedLine.startsWith("event:")) {
            eventType = trimmedLine.slice(6).trim();
          } else if (trimmedLine.startsWith("data:")) {
            dataLines.push(trimmedLine.slice(5).trimStart());
          }
        }

        const rawData = dataLines.join("\n");
        try {
          const parsedData = rawData ? JSON.parse(rawData) : {};
          onEvent({ event: eventType, data: parsedData });
        } catch {
          // skip malformed data payload
        }
      }

      onDone();
    })
    .catch((err: unknown) => {
      const errorObj =
        err instanceof Error ? err : new Error(String(err));
      if (errorObj.name !== "AbortError") {
        onError(errorObj);
      }
    });

  return controller;
}

export function fetchHealth(): Promise<{ status: string }> {
  return request("/health");
}

export interface EvalStrategyMetrics {
  precision_at_k: number;
  recall_at_k: number;
  mrr: number;
  ndcg_at_k: number;
  faithfulness_score: number;
  relevancy_score: number;
  citation_accuracy: number;
  citation_coverage: number;
}

export interface EvalStrategyResult {
  id: string;
  experiment_name: string;
  retrieval_strategy: string;
  chunking_strategy: string;
  created_at: string;
  n_questions: number;
  metrics: EvalStrategyMetrics;
  latency_breakdown: Record<string, unknown> | null;
}

export interface EvalResults {
  has_data: boolean;
  strategies: EvalStrategyResult[];
}

export function fetchEvalResults(): Promise<EvalResults> {
  return request("/api/v1/evaluation/results");
}

export interface ChunkComparisonItem {
  index: number;
  token_count: number;
  char_start: number;
  char_end: number;
  section_heading: string | null;
  preview: string;
}

export interface ChunkComparisonResult {
  document_id: string;
  filename: string;
  current_strategy: string;
  text_length: number;
  strategies: Record<string, ChunkComparisonItem[] | { error: string }>;
}

export function fetchChunkComparison(
  documentId: string,
): Promise<ChunkComparisonResult> {
  return request(`/api/v1/documents/${documentId}/chunk-comparison`);
}

export interface QueryClassification {
  category: string;
  confidence: number;
  recommended_alpha: number;
  reason: string;
}

export function classifyQuery(question: string): Promise<QueryClassification> {
  return request("/api/query/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}
