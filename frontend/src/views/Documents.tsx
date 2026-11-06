import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  AlertTriangle,
  ChevronDown,
  Database,
  File,
  FileText,
  HardDrive,
  MessageSquare,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useNavigate } from "react-router-dom";
import {
  batchDeleteDocuments,
  deleteDocument,
  fetchDocumentChunks,
  fetchDocuments,
  fetchStorageStats,
  uploadDocument,
  type Chunk,
  type Document,
} from "../lib/api";

const STRATEGIES = ["fixed", "structure", "semantic"] as const;
const STRATEGY_LABELS: Record<string, string> = {
  fixed: "Fixed-Size",
  structure: "Structure-Aware",
  semantic: "Semantic",
};

function parseApiError(error: unknown): string {
  if (!error) return "";
  if (typeof error === "object" && error !== null && "detail" in error) {
    return String((error as { detail: unknown }).detail);
  }
  const msg = error instanceof Error ? error.message : String(error);
  const match = msg.match(/^(\d{3}):\s*(.*)$/);
  if (match) {
    const [, status, body] = match;
    try {
      const parsed = JSON.parse(body);
      if (parsed.detail) {
        return typeof parsed.detail === "string"
          ? parsed.detail
          : JSON.stringify(parsed.detail);
      }
    } catch {
      if (body.trim()) return body;
    }
    if (status === "413") return "File exceeds maximum upload size limit (10 MB).";
    if (status === "422") return "Storage quota exceeded or document limit reached. Delete existing documents first.";
    if (status === "409") return "Duplicate file detected.";
    if (status === "403") return "Action forbidden: document may be protected golden evaluation data.";
    return `Server error (${status})`;
  }
  return msg;
}

export default function Documents() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [inspecting, setInspecting] = useState<string | null>(null);
  const [wipeoutOpen, setWipeoutOpen] = useState(false);

  const [strategy, setStrategy] = useState<string>("fixed");
  const [chunkSize, setChunkSize] = useState(400);
  const [overlap, setOverlap] = useState(50);
  const [threshold, setThreshold] = useState(0.5);
  const [clientError, setClientError] = useState<string | null>(null);

  const { data: docs = [], isLoading } = useQuery({
    queryKey: ["documents"],
    queryFn: fetchDocuments,
  });

  const { data: storage } = useQuery({
    queryKey: ["storage"],
    queryFn: fetchStorageStats,
    refetchInterval: 8_000,
  });

  const uploadMut = useMutation({
    mutationFn: (file: File) =>
      uploadDocument(file, strategy, chunkSize, overlap, threshold),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents"] });
      qc.invalidateQueries({ queryKey: ["storage"] });
    },
  });

  const deleteMut = useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents"] });
      qc.invalidateQueries({ queryKey: ["storage"] });
      setSelected(new Set());
    },
  });

  const batchMut = useMutation({
    mutationFn: batchDeleteDocuments,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents"] });
      qc.invalidateQueries({ queryKey: ["storage"] });
      setSelected(new Set());
    },
  });

  const onDrop = useCallback(
    (files: File[]) => {
      setClientError(null);
      for (const f of files) {
        const fileSizeMb = f.size / (1024 * 1024);
        const maxUploadMb = storage?.max_upload_size_mb ?? 50;
        const freeMb =
          storage?.free_mb ??
          Math.max(
            0,
            200 -
              docs.reduce((acc, d) => acc + (d.size_bytes || 0), 0) /
                (1024 * 1024),
          );

        // Guard 1: Single file upload limit (prevent dyno memory spike / crash)
        if (fileSizeMb > maxUploadMb) {
          setClientError(
            `"${f.name}" (${fileSizeMb.toFixed(1)} MB) exceeds the maximum single-file upload limit (${maxUploadMb} MB). Upload blocked to protect server memory.`,
          );
          return;
        }

        // Guard 2: Free database storage headroom
        if (fileSizeMb > freeMb) {
          setClientError(
            `"${f.name}" (${fileSizeMb.toFixed(1)} MB) exceeds remaining database storage (${freeMb.toFixed(1)} MB free). Delete existing documents to free up space.`,
          );
          return;
        }

        // Guard 3: Document count limit
        const maxDocs = storage?.max_documents_count ?? 50;
        const currentDocCount = storage?.document_count ?? docs.length;
        if (currentDocCount >= maxDocs) {
          setClientError(
            `Document count limit reached (${currentDocCount}/${maxDocs}). Delete existing documents first.`,
          );
          return;
        }

        uploadMut.mutate(f);
      }
    },
    [uploadMut, storage, docs],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "text/plain": [".txt"],
      "text/markdown": [".md"],
      "text/html": [".html"],
    },
  });

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === docs.length) setSelected(new Set());
    else setSelected(new Set(docs.map((d) => d.id)));
  };

  const maxDocs = storage?.max_documents_count ?? 50;
  const docPct = Math.min(100, Math.round((docs.length / maxDocs) * 100));
  const storageUsagePct = storage?.usage_percent ?? 0;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold">Documents</h2>
        <div className="flex items-center gap-2">
          <span className="text-xs text-[var(--color-text-tertiary)] flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                storageUsagePct >= 85
                  ? "bg-[var(--color-danger)] animate-pulse"
                  : storageUsagePct >= 70
                    ? "bg-[var(--color-warning)]"
                    : "bg-[var(--color-success)]"
              }`}
            />
            {storage
              ? `${storage.free_mb} MB DB storage available`
              : "Checking storage..."}
          </span>
        </div>
      </div>

      {/* Real-Time Storage & Database Health Monitor */}
      <div className="bg-[var(--color-bg-secondary)] rounded-2xl p-5 border border-[var(--color-border)] shadow-xs space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-[var(--color-primary)]/10 text-[var(--color-primary)] rounded-xl">
              <Database size={20} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm text-[var(--color-text)]">
                  Database Storage Headroom
                </span>
                <span
                  className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium ${
                    storageUsagePct >= 85
                      ? "bg-[var(--color-danger)]/10 text-[var(--color-danger)]"
                      : storageUsagePct >= 70
                        ? "bg-[var(--color-warning)]/10 text-[var(--color-warning)]"
                        : "bg-[var(--color-success)]/10 text-[var(--color-success)]"
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      storageUsagePct >= 85
                        ? "bg-[var(--color-danger)]"
                        : storageUsagePct >= 70
                          ? "bg-[var(--color-warning)]"
                          : "bg-[var(--color-success)]"
                    }`}
                  />
                  {storageUsagePct >= 85
                    ? "Storage Critical"
                    : storageUsagePct >= 70
                      ? "Storage Warning"
                      : "Healthy Headroom"}
                </span>
              </div>
              <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                Real-time tracking of PostgreSQL & pgvector corpus storage to prevent out-of-memory errors
              </p>
            </div>
          </div>

          <div className="text-right">
            <span className="text-[11px] text-[var(--color-text-tertiary)] uppercase tracking-wider block">
              Free Space Available
            </span>
            <span className="font-semibold text-base text-[var(--color-primary)] font-mono">
              {storage ? `${storage.free_mb} MB free` : "Calculating..."}
            </span>
          </div>
        </div>

        {/* Real-time Storage Progress Bar */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs text-[var(--color-text-secondary)]">
            <span>
              {storage
                ? `${storage.used_mb} MB used of ${storage.max_storage_mb} MB storage quota`
                : "Loading storage stats..."}
            </span>
            <span className="font-mono font-medium">
              {storageUsagePct.toFixed(1)}%
            </span>
          </div>

          <div className="w-full h-3 bg-[var(--color-bg-tertiary)] rounded-full overflow-hidden p-0.5 border border-[var(--color-border)]/60">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                storageUsagePct >= 85
                  ? "bg-[var(--color-danger)]"
                  : storageUsagePct >= 70
                    ? "bg-[var(--color-warning)]"
                    : "bg-[var(--color-primary)]"
              }`}
              style={{
                width: `${Math.min(100, Math.max(storage && storage.used_mb > 0 ? 2 : 0, storageUsagePct))}%`,
              }}
            />
          </div>
        </div>

        {/* Secondary Quota Metrics & Safety Badges */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-[var(--color-border)]/60 text-xs text-[var(--color-text-secondary)]">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <FileText size={13} className="text-[var(--color-text-tertiary)]" />
              <span>
                Document Slots:{" "}
                <strong className="text-[var(--color-text)] font-semibold">
                  {storage ? storage.document_count : docs.length}
                </strong>{" "}
                / {maxDocs} ({docPct}%)
              </span>
            </span>

            {storage?.chunk_count != null && (
              <span className="text-[var(--color-text-tertiary)]">
                • {storage.chunk_count} indexed chunks
              </span>
            )}
          </div>

          <div className="flex items-center gap-1.5 bg-[var(--color-bg)] px-2.5 py-1 rounded-lg border border-[var(--color-border)] text-[11px]">
            <HardDrive size={12} className="text-[var(--color-primary)]" />
            <span>
              Safe File Limit:{" "}
              <strong className="text-[var(--color-text)]">
                ≤ {storage?.max_upload_size_mb ?? 50} MB
              </strong>{" "}
              (prevents dyno OOM)
            </span>
          </div>
        </div>
      </div>

      {/* Client-side memory & storage guard banner */}
      {clientError && (
        <div className="flex items-center justify-between p-4 bg-[var(--color-warning)]/10 text-[var(--color-warning)] rounded-xl text-sm border border-[var(--color-warning)]/20">
          <div className="flex items-center gap-2">
            <AlertTriangle size={18} className="shrink-0" />
            <span>{clientError}</span>
          </div>
          <button
            type="button"
            onClick={() => setClientError(null)}
            className="p-1 hover:bg-[var(--color-warning)]/20 rounded text-[var(--color-warning)]"
            title="Dismiss warning"
          >
            <X size={16} />
          </button>
        </div>
      )}

      {/* Upload zone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
          isDragActive
            ? "border-[var(--color-primary)] bg-[var(--color-primary)]/5"
            : "border-[var(--color-border)] hover:border-[var(--color-border-hover)]"
        }`}
      >
        <input {...getInputProps()} />
        <Upload
          size={32}
          className="mx-auto mb-3 text-[var(--color-text-tertiary)]"
        />
        <p className="text-sm text-[var(--color-text-secondary)]">
          Drop PDF, TXT, MD, or HTML files here
        </p>
        <p className="text-xs text-[var(--color-text-tertiary)] mt-1">
          Max {storage?.max_upload_size_mb ?? 50} MB per file •{" "}
          {storage ? `${storage.free_mb} MB free space in database` : "Checking free space..."}
        </p>
        {uploadMut.isPending && (
          <p className="text-xs text-[var(--color-primary)] mt-2">
            Uploading...
          </p>
        )}
      </div>

      {/* Upload error banner */}
      {uploadMut.isError && (
        <div className="flex items-center justify-between p-4 bg-[var(--color-danger)]/10 text-[var(--color-danger)] rounded-xl text-sm border border-[var(--color-danger)]/20">
          <div className="flex items-center gap-2">
            <AlertCircle size={18} className="shrink-0" />
            <span>{parseApiError(uploadMut.error)}</span>
          </div>
          <button
            type="button"
            onClick={() => uploadMut.reset()}
            className="p-1 hover:bg-[var(--color-danger)]/20 rounded text-[var(--color-danger)]"
            title="Dismiss error"
          >
            <X size={16} />
          </button>
        </div>
      )}

      {/* Deletion error banner */}
      {(deleteMut.isError || batchMut.isError) && (
        <div className="flex items-center justify-between p-4 bg-[var(--color-danger)]/10 text-[var(--color-danger)] rounded-xl text-sm border border-[var(--color-danger)]/20">
          <div className="flex items-center gap-2">
            <AlertCircle size={18} className="shrink-0" />
            <span>{parseApiError(deleteMut.error || batchMut.error)}</span>
          </div>
          <button
            type="button"
            onClick={() => {
              deleteMut.reset();
              batchMut.reset();
            }}
            className="p-1 hover:bg-[var(--color-danger)]/20 rounded text-[var(--color-danger)]"
            title="Dismiss error"
          >
            <X size={16} />
          </button>
        </div>
      )}

      {/* Strategy controls */}
      <div className="bg-[var(--color-bg-secondary)] rounded-xl p-4 border border-[var(--color-border)]">
        <div className="grid grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-medium mb-1">Strategy</label>
            <div className="relative">
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full appearance-none bg-[var(--color-bg)] border border-[var(--color-border)] rounded-lg px-3 py-2 text-sm pr-8"
              >
                {STRATEGIES.map((s) => (
                  <option key={s} value={s}>
                    {STRATEGY_LABELS[s]}
                  </option>
                ))}
              </select>
              <ChevronDown
                size={14}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none text-[var(--color-text-tertiary)]"
              />
            </div>
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">
              Chunk Size: {chunkSize}
            </label>
            <input
              type="range"
              min={100}
              max={1000}
              step={50}
              value={chunkSize}
              onChange={(e) => setChunkSize(+e.target.value)}
              className="w-full accent-[var(--color-primary)]"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">
              Overlap: {overlap}
            </label>
            <input
              type="range"
              min={0}
              max={200}
              step={10}
              value={overlap}
              onChange={(e) => setOverlap(+e.target.value)}
              className="w-full accent-[var(--color-primary)]"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1">
              Similarity: {threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(+e.target.value)}
              className="w-full accent-[var(--color-primary)]"
              disabled={strategy !== "semantic"}
            />
          </div>
        </div>
      </div>

      {/* Bulk actions */}
      {selected.size > 0 && (
        <div className="flex items-center gap-3 text-sm">
          <span className="text-[var(--color-text-secondary)]">
            {selected.size} selected
          </span>
          <button
            onClick={() =>
              navigate("/qa", { state: { selectedDocIds: [...selected] } })
            }
            className="px-3 py-1.5 bg-[var(--color-primary)] text-white rounded-lg text-xs font-medium hover:opacity-90 flex items-center gap-1.5 transition-colors"
          >
            <MessageSquare size={13} />
            Ask Questions ({selected.size})
          </button>
          <button
            onClick={() => batchMut.mutate([...selected])}
            className="px-3 py-1.5 bg-[var(--color-danger)] text-white rounded-lg text-xs font-medium hover:opacity-90"
          >
            Delete Selected
          </button>
          <button
            onClick={() => setWipeoutOpen(true)}
            className="px-3 py-1.5 bg-[var(--color-danger)]/10 text-[var(--color-danger)] rounded-lg text-xs font-medium hover:opacity-90"
          >
            Wipeout All
          </button>
        </div>
      )}

      {/* Document table */}
      {isLoading ? (
        <p className="text-sm text-[var(--color-text-secondary)]">Loading...</p>
      ) : docs.length === 0 ? (
        <div className="text-center py-12 text-[var(--color-text-tertiary)]">
          <FileText size={48} className="mx-auto mb-3 opacity-50" />
          <p>No documents uploaded yet</p>
        </div>
      ) : (
        <div className="border border-[var(--color-border)] rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-[var(--color-bg-secondary)]">
              <tr className="text-left text-[var(--color-text-secondary)]">
                <th className="px-4 py-3 w-8">
                  <input
                    type="checkbox"
                    checked={selected.size === docs.length && docs.length > 0}
                    onChange={toggleAll}
                    className="accent-[var(--color-primary)]"
                  />
                </th>
                <th className="px-4 py-3">Name</th>
                <th className="px-4 py-3">Type</th>
                <th className="px-4 py-3">Chunks</th>
                <th className="px-4 py-3">Uploaded</th>
                <th className="px-4 py-3 w-20">Actions</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((doc: Document) => (
                <tr
                  key={doc.id}
                  className="border-t border-[var(--color-border)] hover:bg-[var(--color-bg-secondary)]/50"
                >
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      checked={selected.has(doc.id)}
                      onChange={() => toggleSelect(doc.id)}
                      className="accent-[var(--color-primary)]"
                    />
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => setInspecting(doc.id)}
                      className="flex items-center gap-2 hover:text-[var(--color-primary)] transition-colors"
                    >
                      <File size={14} />
                      {doc.filename}
                    </button>
                  </td>
                  <td className="px-4 py-3 uppercase text-xs font-mono">
                    {doc.file_type}
                  </td>
                  <td className="px-4 py-3">{doc.chunk_count}</td>
                  <td className="px-4 py-3 text-[var(--color-text-secondary)]">
                    {new Date(doc.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() =>
                          navigate("/qa", { state: { selectedDocIds: [doc.id] } })
                        }
                        className="p-1.5 hover:bg-[var(--color-primary)]/10 rounded-lg text-[var(--color-text-tertiary)] hover:text-[var(--color-primary)] transition-colors"
                        title={`Ask questions about ${doc.filename}`}
                      >
                        <MessageSquare size={14} />
                      </button>
                      <button
                        onClick={() => deleteMut.mutate(doc.id)}
                        className="p-1.5 hover:bg-[var(--color-danger)]/10 rounded-lg text-[var(--color-text-tertiary)] hover:text-[var(--color-danger)] transition-colors"
                        title="Delete document"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Chunk inspector drawer */}
      {inspecting && (
        <ChunkDrawer
          docId={inspecting}
          onClose={() => setInspecting(null)}
        />
      )}

      {/* Wipeout modal */}
      {wipeoutOpen && (
        <WipeoutModal
          count={docs.length}
          onConfirm={() => {
            batchMut.mutate(docs.map((d) => d.id));
            setWipeoutOpen(false);
          }}
          onCancel={() => setWipeoutOpen(false)}
        />
      )}
    </div>
  );
}

function ChunkDrawer({
  docId,
  onClose,
}: {
  docId: string;
  onClose: () => void;
}) {
  const { data: chunks = [], isLoading, isError, error } = useQuery({
    queryKey: ["chunks", docId],
    queryFn: () => fetchDocumentChunks(docId),
  });

  return (
    <div className="fixed inset-0 z-50 flex">
      <div className="flex-1 bg-black/30" onClick={onClose} />
      <div className="w-[480px] bg-[var(--color-bg)] border-l border-[var(--color-border)] overflow-y-auto">
        <div className="flex items-center justify-between p-4 border-b border-[var(--color-border)] sticky top-0 bg-[var(--color-bg)] z-10">
          <h3 className="font-semibold">Chunks ({chunks.length})</h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-[var(--color-bg-tertiary)] rounded"
          >
            <X size={18} />
          </button>
        </div>
        {isLoading ? (
          <p className="p-4 text-sm text-[var(--color-text-secondary)]">
            Loading chunks...
          </p>
        ) : isError ? (
          <div className="p-4 text-sm text-[var(--color-danger)] flex items-center gap-2">
            <AlertCircle size={16} className="shrink-0" />
            <span>Failed to load chunks: {parseApiError(error)}</span>
          </div>
        ) : chunks.length === 0 ? (
          <p className="p-4 text-sm text-[var(--color-text-tertiary)]">
            No chunks found for this document.
          </p>
        ) : (
          <div className="divide-y divide-[var(--color-border)]">
            {chunks.map((c: Chunk, i: number) => (
              <div key={c.chunk_id} className="p-4 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono text-[var(--color-text-tertiary)]">
                    #{i} &middot; {c.token_count} tokens
                  </span>
                  {c.section_heading && (
                    <span className="text-[var(--color-primary)] font-medium">
                      {c.section_heading}
                    </span>
                  )}
                </div>
                <p className="text-sm leading-relaxed whitespace-pre-wrap line-clamp-6">
                  {c.content}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function WipeoutModal({
  count,
  onConfirm,
  onCancel,
}: {
  count: number;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-[var(--color-bg)] rounded-2xl p-6 w-96 shadow-xl border border-[var(--color-border)]">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 bg-[var(--color-danger)]/10 rounded-full">
            <AlertTriangle size={20} className="text-[var(--color-danger)]" />
          </div>
          <h3 className="font-semibold">Delete All Documents</h3>
        </div>
        <p className="text-sm text-[var(--color-text-secondary)] mb-6">
          This will permanently delete all {count} documents and their chunks.
          This action cannot be undone.
        </p>
        <div className="flex gap-3 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm rounded-lg border border-[var(--color-border)] hover:bg-[var(--color-bg-secondary)]"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 text-sm rounded-lg bg-[var(--color-danger)] text-white font-medium hover:opacity-90"
          >
            Delete All
          </button>
        </div>
      </div>
    </div>
  );
}
