import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  AlertTriangle,
  ChevronDown,
  File,
  FileText,
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
  loadDemoDocuments,
  uploadDocument,
  type Chunk,
  type Document,
} from "../lib/api";

const STRATEGIES = ["fixed", "structure", "semantic"] as const;
const STRATEGY_LABELS: Record<string, string> = {
  fixed: "Fixed-size",
  structure: "Structure-aware",
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
    if (status === "413") return "File exceeds maximum upload size (10 MB)";
    if (status === "422") return "Storage quota exceeded. Delete existing documents first.";
    if (status === "409") return "Duplicate file detected";
    if (status === "403") return "Document is protected evaluation data";
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

  const [demoBanner, setDemoBanner] = useState<string | null>(null);

  const demoMut = useMutation({
    mutationFn: loadDemoDocuments,
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ["documents"] });
      qc.invalidateQueries({ queryKey: ["storage"] });
      if (res.loaded_count > 0) {
        setDemoBanner(`Loaded ${res.loaded_count} sample documents`);
      } else {
        setDemoBanner("Sample documents already present");
      }
      setTimeout(() => setDemoBanner(null), 4000);
    },
    onError: (err) => {
      setClientError(parseApiError(err));
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

        if (fileSizeMb > maxUploadMb) {
          setClientError(
            `"${f.name}" (${fileSizeMb.toFixed(1)} MB) exceeds the ${maxUploadMb} MB upload limit`,
          );
          return;
        }

        if (fileSizeMb > freeMb) {
          setClientError(
            `"${f.name}" (${fileSizeMb.toFixed(1)} MB) exceeds remaining storage (${freeMb.toFixed(1)} MB free)`,
          );
          return;
        }

        const maxDocs = storage?.max_documents_count ?? 50;
        const currentDocCount = storage?.document_count ?? docs.length;
        if (currentDocCount >= maxDocs) {
          setClientError(
            `Document limit reached (${currentDocCount}/${maxDocs}). Delete documents first.`,
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
  const storageUsagePct = storage?.usage_percent ?? 0;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Page header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-medium">Documents</h2>
          <p className="text-[13px] text-[var(--text-secondary)] mt-1">
            Upload, chunk, and index source materials
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => demoMut.mutate()}
            disabled={demoMut.isPending}
            className="inline-flex items-center gap-2 px-3 py-2 text-[13px] font-medium rounded-lg border border-[var(--border)] bg-[var(--surface-card)] hover:bg-[var(--surface-raised)] text-[var(--text-primary)] transition-colors disabled:opacity-50 cursor-pointer"
          >
            {demoMut.isPending ? "Loading..." : "Load samples"}
          </button>
          <span className="text-[12px] text-[var(--text-muted)] flex items-center gap-2 pl-3 border-l border-[var(--border)]">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                storageUsagePct >= 85
                  ? "bg-[var(--negative)]"
                  : storageUsagePct >= 70
                    ? "bg-[var(--warning)]"
                    : "bg-[var(--positive)]"
              }`}
            />
            {storage
              ? `${storage.free_mb} MB free`
              : "Checking..."}
          </span>
        </div>
      </div>

      {/* Demo banner */}
      {demoBanner && (
        <div className="flex items-center gap-2 px-4 py-3 bg-[var(--surface-card)] text-[var(--text-secondary)] border border-[var(--border)] rounded-lg text-[13px]">
          <span>{demoBanner}</span>
        </div>
      )}

      {/* Storage summary */}
      <div className="bg-[var(--surface-card)] rounded-xl p-5 border border-[var(--border)] space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span className="font-medium text-[14px] text-[var(--text-primary)]">
              Storage
            </span>
            <span className="ml-3 text-[12px] text-[var(--text-muted)] font-mono">
              {storageUsagePct.toFixed(1)}% used
            </span>
          </div>
          <span className="text-[13px] text-[var(--text-secondary)] font-mono tabular-nums">
            {storage ? `${storage.used_mb} / ${storage.max_storage_mb} MB` : "Loading..."}
          </span>
        </div>

        <div className="w-full h-1.5 bg-[var(--surface-raised)] rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-[width] duration-500 ${
              storageUsagePct >= 85
                ? "bg-[var(--negative)]"
                : storageUsagePct >= 70
                  ? "bg-[var(--warning)]"
                  : "bg-[var(--accent)]"
            }`}
            style={{
              width: `${Math.min(100, Math.max(storage && storage.used_mb > 0 ? 2 : 0, storageUsagePct))}%`,
            }}
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 text-[12px] text-[var(--text-muted)]">
          <div className="flex items-center gap-4">
            <span>
              Documents: <span className="text-[var(--text-primary)] font-medium tabular-nums">{storage ? storage.document_count : docs.length}</span> / {maxDocs}
            </span>
            {storage?.chunk_count != null && (
              <span className="tabular-nums">{storage.chunk_count} chunks indexed</span>
            )}
          </div>
          <span className="font-mono tabular-nums">
            Max upload: {storage?.max_upload_size_mb ?? 50} MB
          </span>
        </div>
      </div>

      {/* Client error banner */}
      {clientError && (
        <div className="flex items-center justify-between px-4 py-3 bg-[var(--surface-card)] border border-[var(--border)] rounded-lg text-[13px]">
          <div className="flex items-center gap-2 text-[var(--warning)]">
            <AlertTriangle size={16} strokeWidth={1.5} className="shrink-0" />
            <span>{clientError}</span>
          </div>
          <button
            type="button"
            onClick={() => setClientError(null)}
            className="p-1 text-[var(--text-muted)] hover:text-[var(--text-primary)] rounded transition-colors"
          >
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>
      )}

      {/* Upload zone */}
      <div
        {...getRootProps()}
        className={`border border-dashed rounded-xl px-8 py-10 text-center cursor-pointer transition-colors ${
          isDragActive
            ? "border-[var(--accent)] bg-[var(--surface-raised)]"
            : "border-[var(--border)] hover:border-[var(--text-muted)]"
        }`}
      >
        <input {...getInputProps()} />
        <Upload
          size={24}
          strokeWidth={1.5}
          className="mx-auto mb-3 text-[var(--text-muted)]"
        />
        <p className="text-[13px] text-[var(--text-secondary)]">
          Drop PDF, TXT, MD, or HTML files here
        </p>
        <p className="text-[12px] text-[var(--text-muted)] mt-1">
          Up to {storage?.max_upload_size_mb ?? 50} MB per file
        </p>
        {uploadMut.isPending && (
          <p className="text-[12px] text-[var(--accent-fg)] mt-2">
            Uploading...
          </p>
        )}
      </div>

      {/* Upload error */}
      {uploadMut.isError && (
        <div className="flex items-center justify-between px-4 py-3 bg-[var(--surface-card)] border border-[var(--border)] rounded-lg text-[13px]">
          <div className="flex items-center gap-2 text-[var(--negative)]">
            <AlertCircle size={16} strokeWidth={1.5} className="shrink-0" />
            <span>{parseApiError(uploadMut.error)}</span>
          </div>
          <button
            type="button"
            onClick={() => uploadMut.reset()}
            className="p-1 text-[var(--text-muted)] hover:text-[var(--text-primary)] rounded transition-colors"
          >
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>
      )}

      {/* Delete error */}
      {(deleteMut.isError || batchMut.isError) && (
        <div className="flex items-center justify-between px-4 py-3 bg-[var(--surface-card)] border border-[var(--border)] rounded-lg text-[13px]">
          <div className="flex items-center gap-2 text-[var(--negative)]">
            <AlertCircle size={16} strokeWidth={1.5} className="shrink-0" />
            <span>{parseApiError(deleteMut.error || batchMut.error)}</span>
          </div>
          <button
            type="button"
            onClick={() => { deleteMut.reset(); batchMut.reset(); }}
            className="p-1 text-[var(--text-muted)] hover:text-[var(--text-primary)] rounded transition-colors"
          >
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>
      )}

      {/* Chunking strategy controls */}
      <div className="bg-[var(--surface-card)] rounded-xl p-4 border border-[var(--border)]">
        <div className="grid grid-cols-4 gap-4">
          <div>
            <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-1.5">Strategy</label>
            <div className="relative">
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full appearance-none bg-[var(--surface)] border border-[var(--border)] rounded-lg px-3 py-2 text-[13px] pr-8 text-[var(--text-primary)]"
              >
                {STRATEGIES.map((s) => (
                  <option key={s} value={s}>
                    {STRATEGY_LABELS[s]}
                  </option>
                ))}
              </select>
              <ChevronDown
                size={14}
                strokeWidth={1.5}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none text-[var(--text-muted)]"
              />
            </div>
          </div>
          <div>
            <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-1.5">
              Chunk size: <span className="font-mono tabular-nums">{chunkSize}</span>
            </label>
            <input
              type="range"
              min={100}
              max={1000}
              step={50}
              value={chunkSize}
              onChange={(e) => setChunkSize(+e.target.value)}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-1.5">
              Overlap: <span className="font-mono tabular-nums">{overlap}</span>
            </label>
            <input
              type="range"
              min={0}
              max={200}
              step={10}
              value={overlap}
              onChange={(e) => setOverlap(+e.target.value)}
              className="w-full"
            />
          </div>
          <div>
            <label className="block text-[12px] font-medium text-[var(--text-secondary)] mb-1.5">
              Similarity: <span className="font-mono tabular-nums">{threshold.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(+e.target.value)}
              className="w-full"
              disabled={strategy !== "semantic"}
            />
          </div>
        </div>
      </div>

      {/* Bulk actions */}
      {selected.size > 0 && (
        <div className="flex items-center gap-3 text-[13px]">
          <span className="text-[var(--text-muted)] tabular-nums">
            {selected.size} selected
          </span>
          <button
            onClick={() =>
              navigate("/qa", { state: { selectedDocIds: [...selected] } })
            }
            className="px-3 py-1.5 bg-[var(--accent)] text-white rounded-lg text-[12px] font-medium hover:opacity-90 flex items-center gap-1.5 transition-colors"
          >
            <MessageSquare size={13} strokeWidth={1.5} />
            Ask ({selected.size})
          </button>
          <button
            onClick={() => batchMut.mutate([...selected])}
            className="px-3 py-1.5 bg-[var(--negative)] text-white rounded-lg text-[12px] font-medium hover:opacity-90 transition-colors"
          >
            Delete selected
          </button>
          <button
            onClick={() => setWipeoutOpen(true)}
            className="px-3 py-1.5 text-[var(--negative)] rounded-lg text-[12px] font-medium border border-[var(--border)] hover:bg-[var(--surface-raised)] transition-colors"
          >
            Delete all
          </button>
        </div>
      )}

      {/* Document table */}
      {isLoading ? (
        <p className="text-[13px] text-[var(--text-muted)]">Loading...</p>
      ) : docs.length === 0 ? (
        <div className="text-center py-16 text-[var(--text-muted)]">
          <FileText size={32} strokeWidth={1.5} className="mx-auto mb-3 opacity-40" />
          <p className="text-[14px]">No documents uploaded</p>
          <p className="text-[12px] mt-1">Drop files above or load sample documents</p>
        </div>
      ) : (
        <div className="border border-[var(--border)] rounded-xl overflow-hidden">
          <table className="w-full text-[13px]">
            <thead className="bg-[var(--surface-card)]">
              <tr className="text-left text-[var(--text-muted)]">
                <th className="px-4 py-3 w-8">
                  <input
                    type="checkbox"
                    checked={selected.size === docs.length && docs.length > 0}
                    onChange={toggleAll}
                  />
                </th>
                <th className="px-4 py-3 font-medium">Name</th>
                <th className="px-4 py-3 font-medium">Type</th>
                <th className="px-4 py-3 font-medium">Chunks</th>
                <th className="px-4 py-3 font-medium">Uploaded</th>
                <th className="px-4 py-3 w-20 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((doc: Document) => (
                <tr
                  key={doc.id}
                  className="border-t border-[var(--border)] hover:bg-[var(--surface-raised)]/50 transition-colors"
                >
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      checked={selected.has(doc.id)}
                      onChange={() => toggleSelect(doc.id)}
                    />
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => setInspecting(doc.id)}
                      className="flex items-center gap-2 text-[var(--text-primary)] hover:text-[var(--accent-fg)] transition-colors"
                    >
                      <File size={14} strokeWidth={1.5} />
                      {doc.filename}
                    </button>
                  </td>
                  <td className="px-4 py-3 text-[12px] font-mono text-[var(--text-muted)]">
                    {doc.file_type}
                  </td>
                  <td className="px-4 py-3 font-mono tabular-nums">{doc.chunk_count}</td>
                  <td className="px-4 py-3 text-[var(--text-secondary)]">
                    {new Date(doc.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() =>
                          navigate("/qa", { state: { selectedDocIds: [doc.id] } })
                        }
                        className="p-1.5 hover:bg-[var(--surface-raised)] rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
                        title={`Ask about ${doc.filename}`}
                      >
                        <MessageSquare size={14} strokeWidth={1.5} />
                      </button>
                      <button
                        onClick={() => deleteMut.mutate(doc.id)}
                        className="p-1.5 hover:bg-[var(--surface-raised)] rounded-lg text-[var(--text-muted)] hover:text-[var(--negative)] transition-colors"
                        title="Delete"
                      >
                        <Trash2 size={14} strokeWidth={1.5} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {inspecting && (
        <ChunkDrawer docId={inspecting} onClose={() => setInspecting(null)} />
      )}

      {wipeoutOpen && (
        <WipeoutModal
          count={docs.filter((d) => d.is_evictable !== false).length}
          protectedCount={docs.filter((d) => d.is_evictable === false).length}
          onConfirm={() => {
            const evictable = docs.filter((d) => d.is_evictable !== false).map((d) => d.id);
            if (evictable.length > 0) batchMut.mutate(evictable);
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
      <div className="w-[480px] bg-[var(--surface)] border-l border-[var(--border)] overflow-y-auto">
        <div className="flex items-center justify-between px-5 h-14 border-b border-[var(--border)] sticky top-0 bg-[var(--surface)] z-10">
          <h3 className="font-medium text-[14px]">
            Chunks <span className="text-[var(--text-muted)] font-mono tabular-nums">({chunks.length})</span>
          </h3>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-[var(--surface-raised)] rounded-lg text-[var(--text-muted)] transition-colors"
          >
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>
        {isLoading ? (
          <p className="p-5 text-[13px] text-[var(--text-muted)]">Loading chunks...</p>
        ) : isError ? (
          <div className="p-5 text-[13px] text-[var(--negative)] flex items-center gap-2">
            <AlertCircle size={16} strokeWidth={1.5} className="shrink-0" />
            <span>Failed to load: {parseApiError(error)}</span>
          </div>
        ) : chunks.length === 0 ? (
          <p className="p-5 text-[13px] text-[var(--text-muted)]">
            No chunks found
          </p>
        ) : (
          <div className="divide-y divide-[var(--border)]">
            {chunks.map((c: Chunk, i: number) => (
              <div key={c.chunk_id} className="px-5 py-4 space-y-2">
                <div className="flex items-center justify-between text-[12px]">
                  <span className="font-mono text-[var(--text-muted)] tabular-nums">
                    #{i} · {c.token_count} tokens
                  </span>
                  {c.section_heading && (
                    <span className="text-[var(--accent-fg)] font-medium truncate max-w-[200px]">
                      {c.section_heading}
                    </span>
                  )}
                </div>
                <p className="text-[13px] leading-relaxed whitespace-pre-wrap line-clamp-6 text-[var(--text-secondary)]">
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
  protectedCount = 0,
  onConfirm,
  onCancel,
}: {
  count: number;
  protectedCount?: number;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-[var(--surface-card)] rounded-xl p-6 w-96 border border-[var(--border)]">
        <div className="flex items-center gap-3 mb-4">
          <AlertTriangle size={20} strokeWidth={1.5} className="text-[var(--negative)]" />
          <h3 className="font-medium text-[15px]">Delete all documents</h3>
        </div>
        <p className="text-[13px] text-[var(--text-secondary)] mb-6">
          This permanently deletes {count} uploaded documents and their chunks.
          {protectedCount > 0 && (
            <span className="block mt-2 text-[12px] text-[var(--text-muted)]">
              {protectedCount} protected evaluation document{protectedCount > 1 ? "s" : ""} will be preserved.
            </span>
          )}
        </p>
        <div className="flex gap-3 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-[13px] rounded-lg border border-[var(--border)] hover:bg-[var(--surface-raised)] text-[var(--text-primary)] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={count === 0}
            className="px-4 py-2 text-[13px] rounded-lg bg-[var(--negative)] text-white font-medium hover:opacity-90 disabled:opacity-50 transition-colors"
          >
            Delete all
          </button>
        </div>
      </div>
    </div>
  );
}
