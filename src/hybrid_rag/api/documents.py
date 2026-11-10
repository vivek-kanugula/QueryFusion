"""Document CRUD endpoints: upload, list, delete, batch-delete, consistency check."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from hybrid_rag.db.engine import get_db
from hybrid_rag.ingest.chunkers import CHUNKERS, chunk_fixed
from hybrid_rag.ingest.index_manager import (
    check_consistency,
    delete_document,
    index_document,
)
from hybrid_rag.ingest.parsers import parse_file
from hybrid_rag.models.document import Document, DocumentChunk
from hybrid_rag.settings import settings

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
)


@router.post("")
def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Query(default="fixed"),
    background: bool = Query(default=False),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
):
    if chunking_strategy not in CHUNKERS:
        valid = list(CHUNKERS.keys())
        raise HTTPException(400, f"Unknown strategy: {chunking_strategy}. Choose: {valid}")

    max_upload_bytes = settings.max_upload_size_mb * 1024 * 1024
    chunk_size = 1024 * 1024  # 1 MB chunk
    chunks_list: list[bytes] = []
    total_bytes = 0

    try:
        while True:
            chunk = file.file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_upload_bytes:
                raise HTTPException(413, f"File exceeds {settings.max_upload_size_mb} MB limit")
            chunks_list.append(chunk)
    finally:
        file.file.close()

    data = b"".join(chunks_list)
    if not data or len(data) == 0:
        raise HTTPException(400, "File is empty")

    max_storage_bytes = settings.max_storage_mb * 1024 * 1024
    current_storage = db.query(func.coalesce(func.sum(Document.size_bytes), 0)).scalar()
    try:
        current_storage_bytes = int(current_storage) if current_storage is not None else 0
    except (TypeError, ValueError):
        current_storage_bytes = 0

    if current_storage_bytes + len(data) > max_storage_bytes:
        raise HTTPException(
            422,
            f"Storage quota exceeded ({settings.max_storage_mb} MB). "
            "Delete existing documents first.",
        )

    try:
        parsed = parse_file(data, file.filename or "unknown")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse document: {e}")

    existing = db.query(Document).filter(Document.file_hash == parsed.content_hash).first()
    if existing:
        raise HTTPException(409, f"Duplicate file (matches document {existing.id})")

    doc_count = db.query(Document).count()
    if doc_count >= settings.max_documents_count:
        raise HTTPException(
            422, f"Document limit reached ({settings.max_documents_count}). Delete documents first."
        )

    doc = Document(
        id=uuid.uuid4(),
        filename=parsed.filename,
        file_type=parsed.file_type,
        file_hash=parsed.content_hash,
        size_bytes=parsed.size_bytes,
        chunking_strategy=chunking_strategy,
        status="processing" if background else "ready",
        is_evictable=True,
    )

    from sqlalchemy.exc import IntegrityError

    if background:
        from hybrid_rag.core.jobs import job_registry
        from hybrid_rag.ingest.index_manager import process_document_background

        if background_tasks is None:
            background_tasks = BackgroundTasks()

        job = job_registry.create_job("document_ingest")

        try:
            db.add(doc)
            db.commit()
            db.refresh(doc)
        except IntegrityError:
            db.rollback()
            raise HTTPException(409, "Duplicate file uploaded concurrently")

        background_tasks.add_task(
            process_document_background,
            doc.id,
            data,
            file.filename or "unknown",
            chunking_strategy,
            job.job_id,
        )

        return {
            "document_id": str(doc.id),
            "filename": doc.filename,
            "file_type": doc.file_type,
            "chunks_created": 0,
            "chunking_strategy": chunking_strategy,
            "status": "processing",
            "job_id": job.job_id,
        }

    chunker = CHUNKERS[chunking_strategy]

    if chunking_strategy == "fixed":
        chunks = chunker(parsed.full_text)
    elif chunking_strategy == "structure":
        section_dicts = [{"text": s.text, "heading": s.heading} for s in parsed.sections]
        chunks = chunker(parsed.full_text, sections=section_dicts)
    elif chunking_strategy == "semantic":
        chunks = chunker(parsed.full_text)
    else:
        chunks = chunk_fixed(parsed.full_text)

    try:
        db_chunks = index_document(db, doc, chunks, chunking_strategy)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Duplicate file uploaded concurrently")

    return {
        "document_id": str(doc.id),
        "filename": doc.filename,
        "file_type": doc.file_type,
        "chunks_created": len(db_chunks),
        "chunking_strategy": chunking_strategy,
        "status": "ready",
        "job_id": None,
    }


@router.post("/demo")
def load_demo_documents(db: Session = Depends(get_db)):
    """Seed pre-packaged demonstration documents for quick testing."""
    from hybrid_rag.data.sample_documents import DEMO_DOCUMENTS

    created_docs: list[dict[str, object]] = []
    already_existing: list[str] = []

    for item in DEMO_DOCUMENTS:
        filename = str(item["filename"])
        content_str = str(item["content"])
        strategy = str(item["chunking_strategy"])
        raw_bytes = content_str.encode("utf-8")
        parsed = parse_file(raw_bytes, filename)

        existing = db.query(Document).filter(
            (Document.filename == filename) | (Document.file_hash == parsed.content_hash)
        ).first()

        if existing:
            already_existing.append(filename)
            continue

        doc = Document(
            filename=parsed.filename,
            file_type=parsed.file_type,
            file_hash=parsed.content_hash,
            size_bytes=parsed.size_bytes,
            chunking_strategy=strategy,
            is_evictable=True,
        )

        chunker = CHUNKERS.get(strategy, chunk_fixed)
        if strategy == "structure" and parsed.sections:
            section_dicts = [{"text": s.text, "heading": s.heading} for s in parsed.sections]
            chunks = chunker(parsed.full_text, sections=section_dicts)
        else:
            chunks = chunker(parsed.full_text)

        db_chunks = index_document(db, doc, chunks, strategy)
        created_docs.append({
            "document_id": str(doc.id),
            "filename": doc.filename,
            "chunks_created": len(db_chunks),
            "chunking_strategy": strategy,
        })

    return {
        "status": "success",
        "loaded_count": len(created_docs),
        "existing_count": len(already_existing),
        "created_documents": created_docs,
        "already_existing": already_existing,
    }


@router.get("")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [
        {
            "id": str(d.id),
            "filename": d.filename,
            "file_type": d.file_type,
            "size_bytes": d.size_bytes,
            "chunking_strategy": d.chunking_strategy,
            "chunk_count": len(d.chunks),
            "status": getattr(d, "status", "ready") or "ready",
            "error_message": getattr(d, "error_message", None),
            "is_evictable": d.is_evictable,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.get("/{document_id}/status")
def get_document_status(document_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieve async ingestion status and chunk count for a document."""
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return {
        "document_id": str(doc.id),
        "filename": doc.filename,
        "status": getattr(doc, "status", "ready") or "ready",
        "error_message": getattr(doc, "error_message", None),
        "chunk_count": len(doc.chunks) if doc.chunks else 0,
        "chunking_strategy": doc.chunking_strategy,
    }


@router.get("/storage")
def get_storage_stats(db: Session = Depends(get_db)):
    """Return real-time database storage metrics, quotas, and headroom."""
    max_storage_bytes = settings.max_storage_mb * 1024 * 1024
    max_upload_bytes = settings.max_upload_size_mb * 1024 * 1024

    current_storage = db.query(func.coalesce(func.sum(Document.size_bytes), 0)).scalar()
    try:
        used_bytes = int(current_storage) if current_storage is not None else 0
    except (TypeError, ValueError):
        used_bytes = 0

    doc_count = db.query(Document).count()
    chunk_count = db.query(DocumentChunk).count()
    free_bytes = max(0, max_storage_bytes - used_bytes)

    used_mb = round(used_bytes / (1024 * 1024), 2)
    free_mb = round(free_bytes / (1024 * 1024), 2)
    usage_percent = (
        round((used_bytes / max_storage_bytes) * 100, 1) if max_storage_bytes > 0 else 0.0
    )
    usage_percent = min(100.0, usage_percent)

    hnsw_ceiling = 35000
    ceiling_percent = round((chunk_count / hnsw_ceiling) * 100, 1) if hnsw_ceiling > 0 else 0.0
    storage_warning = None
    if chunk_count > hnsw_ceiling * 0.8:
        storage_warning = (
            f"Chunk count ({chunk_count}) approaching HNSW index ceiling (~{hnsw_ceiling}). "
            "Consider deleting unused documents."
        )

    return {
        "used_bytes": used_bytes,
        "used_mb": used_mb,
        "free_bytes": free_bytes,
        "free_mb": free_mb,
        "max_storage_mb": settings.max_storage_mb,
        "max_storage_bytes": max_storage_bytes,
        "usage_percent": usage_percent,
        "max_upload_size_mb": settings.max_upload_size_mb,
        "max_upload_size_bytes": max_upload_bytes,
        "document_count": doc_count,
        "max_documents_count": settings.max_documents_count,
        "chunk_count": chunk_count,
        "hnsw_ceiling": hnsw_ceiling,
        "hnsw_usage_percent": ceiling_percent,
        "storage_warning": storage_warning,
    }


@router.get("/chunks/{chunk_id}/context")
def get_chunk_context(
    chunk_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Retrieve detailed context for a chunk including document metadata and adjacent chunks."""
    target_chunk = db.get(DocumentChunk, chunk_id)
    if target_chunk is None:
        target_chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()

    if target_chunk is None:
        raise HTTPException(404, f"Chunk {chunk_id} not found")

    doc = db.get(Document, target_chunk.document_id)
    if doc is None:
        doc = db.query(Document).filter(Document.id == target_chunk.document_id).first()

    total_chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == target_chunk.document_id)
        .count()
    )

    prev_chunk = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id == target_chunk.document_id,
            DocumentChunk.chunk_index == target_chunk.chunk_index - 1,
        )
        .first()
    )

    next_chunk = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id == target_chunk.document_id,
            DocumentChunk.chunk_index == target_chunk.chunk_index + 1,
        )
        .first()
    )

    def _format_chunk(c: DocumentChunk | None) -> dict[str, object] | None:
        if c is None:
            return None
        return {
            "chunk_id": str(c.id),
            "index": c.chunk_index,
            "token_count": c.token_count,
            "content": c.content,
            "section_heading": c.section_heading or "",
        }

    return {
        "document": {
            "id": str(doc.id) if doc else str(target_chunk.document_id),
            "filename": doc.filename if doc else "Unknown",
            "file_type": doc.file_type if doc else "text",
            "chunking_strategy": doc.chunking_strategy if doc else "fixed",
            "size_bytes": doc.size_bytes if doc else 0,
            "total_chunks": total_chunks,
        },
        "target_chunk": _format_chunk(target_chunk),
        "previous_chunk": _format_chunk(prev_chunk),
        "next_chunk": _format_chunk(next_chunk),
    }


@router.get("/{document_id}/chunk-comparison")
def compare_chunking_strategies(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Show how each chunking strategy splits the same document."""
    doc = db.get(Document, document_id)
    if doc is None:
        doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(404, "Document not found")

    full_text = ""
    chunks_in_db = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )
    if chunks_in_db:
        full_text = "\n".join(c.content for c in chunks_in_db)

    if not full_text.strip():
        return {
            "document_id": str(document_id),
            "filename": doc.filename,
            "strategies": {},
            "error": "No text content available for comparison",
        }

    from hybrid_rag.ingest.chunkers import chunk_fixed, chunk_structure_aware

    strategies_output = {}
    for name, chunker_fn in [
        ("fixed", chunk_fixed),
        ("structure", chunk_structure_aware),
    ]:
        try:
            result_chunks = chunker_fn(full_text)
            strategies_output[name] = [
                {
                    "index": c.chunk_index,
                    "token_count": c.token_count,
                    "char_start": c.char_offset_start,
                    "char_end": c.char_offset_end,
                    "section_heading": c.section_heading,
                    "preview": c.text[:200] + ("..." if len(c.text) > 200 else ""),
                }
                for c in result_chunks
            ]
        except Exception as exc:
            strategies_output[name] = {"error": str(exc)}

    return {
        "document_id": str(document_id),
        "filename": doc.filename,
        "current_strategy": doc.chunking_strategy,
        "text_length": len(full_text),
        "strategies": strategies_output,
    }


@router.get("/{document_id}/chunks")
def list_document_chunks(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Retrieve all chunks belonging to a document ordered by chunk_index."""
    doc = db.get(Document, document_id)
    if doc is None:
        doc = db.query(Document).filter(Document.id == document_id).first()

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )

    if not chunks and doc is not None and hasattr(doc, "chunks") and doc.chunks:
        chunks = sorted(doc.chunks, key=lambda x: getattr(x, "chunk_index", 0))

    if not doc and not chunks:
        raise HTTPException(404, "Document not found")

    return [
        {
            "chunk_id": str(c.id),
            "index": c.chunk_index,
            "token_count": c.token_count,
            "content": c.content,
            "section_heading": c.section_heading or "",
        }
        for c in chunks
    ]


@router.delete("/{document_id}")
def delete_single_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    if not doc.is_evictable:
        raise HTTPException(
            403,
            f"Document {document_id} is protected (golden evaluation data) and cannot be deleted",
        )

    delete_document(db, document_id, doc.chunking_strategy)
    return {"deleted": str(document_id)}


class BatchDeleteRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)


@router.post("/batch-delete")
def batch_delete_documents(
    body: BatchDeleteRequest,
    db: Session = Depends(get_db),
):
    if not body.document_ids:
        raise HTTPException(400, "document_ids required")

    try:
        doc_ids = [uuid.UUID(str(i)) for i in body.document_ids]
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, "Invalid UUID in document_ids")

    docs = db.query(Document).filter(Document.id.in_(doc_ids)).all()
    if not docs:
        raise HTTPException(404, "No matching documents found")

    protected = [d for d in docs if not d.is_evictable]
    if protected:
        names = [f"{d.filename} ({d.id})" for d in protected]
        raise HTTPException(
            403,
            f"Cannot delete protected golden documents: {', '.join(names)}",
        )

    for doc in docs:
        delete_document(db, doc.id, doc.chunking_strategy)

    return {"deleted": [str(d.id) for d in docs]}


@router.get("/consistency")
def consistency_check(
    strategy: str = Query(default="fixed"),
    db: Session = Depends(get_db),
):
    result = check_consistency(db, strategy)
    status_code = 200 if result["consistent"] else 503
    from fastapi.responses import JSONResponse

    return JSONResponse(content=result, status_code=status_code)
