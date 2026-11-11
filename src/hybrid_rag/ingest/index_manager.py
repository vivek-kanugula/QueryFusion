"""Index Manager: orchestration for write/delete across PG and BM25.

Write order: Postgres (with pgvector embeddings) -> BM25
Delete order: Postgres (cascade) -> BM25 rebuild
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

from sqlalchemy.orm import Session

from hybrid_rag.core.chunk_id import make_chunk_id
from hybrid_rag.ingest.chunkers import ChunkOutput
from hybrid_rag.ingest.embedder import embed_texts
from hybrid_rag.models.document import Document, DocumentChunk

logger = logging.getLogger(__name__)


def index_document(
    db: Session,
    document: Document,
    chunks: list[ChunkOutput],
    strategy: str,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[DocumentChunk]:
    """Write chunks with pgvector embeddings to Postgres and rebuild BM25.

    Uses compensating rollback if any phase fails.
    """
    if document.id is None:
        document.id = uuid.uuid4()

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts) if texts else []
    if progress_callback:
        progress_callback(35.0, f"Generated embeddings for {len(chunks)} chunks")

    db_chunks: list[DocumentChunk] = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk_id = make_chunk_id(document.id, strategy, chunk.chunk_index)
        db_chunk = DocumentChunk(
            id=chunk_id,
            document_id=document.id,
            chunk_index=chunk.chunk_index,
            content=chunk.text,
            token_count=chunk.token_count,
            page_number=chunk.page_number,
            section_heading=chunk.section_heading,
            char_offset_start=chunk.char_offset_start,
            char_offset_end=chunk.char_offset_end,
            embedding=embedding.tolist() if embedding is not None else None,
        )
        db_chunks.append(db_chunk)

    try:
        db.add(document)
        db.add_all(db_chunks)
        db.flush()

        from hybrid_rag.retrieval.bm25_store import get_bm25_store

        store = get_bm25_store()
        chunk_ids = [str(c.id) for c in db_chunks]
        doc_id_strs = [str(document.id)] * len(db_chunks)
        result = store.add_chunks(texts, chunk_ids, strategy, doc_ids=doc_id_strs)
        if result == -1:
            store.rebuild_from_db(db, strategy)

        if progress_callback:
            progress_callback(80.0, "Updated BM25 sparse index")

        db.commit()

        if progress_callback:
            progress_callback(100.0, "Committed chunks to database")

    except Exception as exc:
        logger.error(
            "Indexing failed for document %s (strategy=%s); rolling back: %s",
            document.id,
            strategy,
            exc,
            exc_info=True,
        )
        try:
            db.rollback()
        except Exception:
            logger.warning("DB rollback failed during compensating rollback", exc_info=True)

        try:
            from hybrid_rag.db.engine import SessionLocal
            from hybrid_rag.retrieval.bm25_store import get_bm25_store

            store = get_bm25_store()
            with SessionLocal() as fresh_db:
                store.rebuild_from_db(fresh_db, strategy)
        except Exception:
            logger.warning(
                "BM25 index rebuild failed during compensating rollback for document %s; "
                "sparse retrieval may be stale until next restart",
                document.id,
                exc_info=True,
            )

        raise

    logger.info(
        "Indexed %d chunks for document %s (strategy=%s)",
        len(db_chunks),
        document.id,
        strategy,
    )
    return db_chunks


def process_document_background(
    document_id: uuid.UUID,
    raw_data: bytes,
    filename: str,
    chunking_strategy: str,
    job_id: str | None = None,
) -> None:
    """Asynchronous background worker to parse, chunk, embed, and index a document.

    Runs decoupled from the HTTP request cycle with its own thread-safe DB session
    and reports live progress to the core JobRegistry.
    """
    from hybrid_rag.core.jobs import job_registry
    from hybrid_rag.db.engine import SessionLocal
    from hybrid_rag.ingest.chunkers import CHUNKERS, chunk_fixed
    from hybrid_rag.ingest.parsers import parse_file

    with SessionLocal() as db:
        doc = db.get(Document, document_id)
        if not doc:
            logger.error("Background task: Document %s not found in DB", document_id)
            if job_id:
                job_registry.fail_job(job_id, f"Document {document_id} not found")
            return

        try:
            if job_id:
                job_registry.update_progress(job_id, 10.0, "Parsing document structure...")

            parsed = parse_file(raw_data, filename)

            if job_id:
                job_registry.update_progress(job_id, 25.0, "Splitting document into chunks...")

            chunker = CHUNKERS.get(chunking_strategy, chunk_fixed)
            if chunking_strategy == "structure" and parsed.sections:
                section_dicts = [{"text": s.text, "heading": s.heading} for s in parsed.sections]
                chunks = chunker(parsed.full_text, sections=section_dicts)
            else:
                chunks = chunker(parsed.full_text)

            def _on_sub_progress(pct: float, msg: str) -> None:
                if job_id:
                    scaled = 30.0 + (pct / 100.0) * 60.0
                    job_registry.update_progress(job_id, scaled, msg)

            db_chunks = index_document(
                db, doc, chunks, chunking_strategy, progress_callback=_on_sub_progress
            )

            doc.status = "ready"
            doc.error_message = None
            db.commit()

            try:
                from hybrid_rag.storage.s3 import is_storage_configured, upload_file

                if is_storage_configured():
                    upload_file(f"documents/{doc.id}/{parsed.filename}", raw_data)
            except Exception as exc:
                logger.warning("Object storage upload failed in background: %s", exc)

            if job_id:
                job_registry.complete_job(
                    job_id,
                    {
                        "document_id": str(doc.id),
                        "filename": doc.filename,
                        "file_type": doc.file_type,
                        "chunks_created": len(db_chunks),
                        "chunking_strategy": chunking_strategy,
                        "status": "ready",
                    },
                )
            logger.info(
                "Background indexing completed for document %s (%d chunks)", doc.id, len(db_chunks)
            )

        except Exception as exc:
            logger.error(
                "Background indexing failed for document %s: %s", document_id, exc, exc_info=True
            )
            try:
                doc.status = "failed"
                doc.error_message = str(exc)
                db.commit()
            except Exception:
                db.rollback()

            if job_id:
                job_registry.fail_job(job_id, str(exc))


def delete_document(
    db: Session,
    document_id: uuid.UUID,
    strategy: str = "fixed",
) -> None:
    """Delete document from Postgres (cascading to chunks/vectors) and rebuild BM25."""
    doc = db.get(Document, document_id)
    if doc:
        filename = doc.filename
        try:
            db.delete(doc)
            db.commit()
            logger.info("Deleted document %s from Postgres (cascade to chunks)", document_id)
            try:
                from hybrid_rag.storage.s3 import delete_file, is_storage_configured

                if is_storage_configured():
                    delete_file(f"documents/{document_id}/{filename}")
            except Exception:
                pass
        except Exception as exc:
            logger.error(
                "Failed to delete document %s from Postgres; rolling back: %s",
                document_id,
                exc,
                exc_info=True,
            )
            try:
                db.rollback()
            except Exception:
                logger.warning("DB rollback failed during delete compensation", exc_info=True)
            raise

    try:
        from hybrid_rag.retrieval.bm25_store import get_bm25_store

        store = get_bm25_store()
        result = store.remove_by_doc_id(strategy, str(document_id))
        if result == -1:
            store.rebuild_from_db(db, strategy)
    except Exception:
        logger.warning(
            "BM25 index rebuild failed after deleting document %s",
            document_id,
            exc_info=True,
        )


def delete_documents_batch(
    db: Session,
    document_ids: list[uuid.UUID],
    strategy: str = "fixed",
) -> None:
    for doc_id in document_ids:
        delete_document(db, doc_id, strategy)


def check_consistency(
    db: Session,
    strategy: str = "fixed",
) -> dict:
    """Assert chunk count parity and pgvector embedding presence."""
    from sqlalchemy import func

    pg_count = (
        db.query(func.count(DocumentChunk.id))
        .join(Document)
        .filter(Document.chunking_strategy == strategy)
        .scalar()
        or 0
    )

    embedded_count = (
        db.query(func.count(DocumentChunk.id))
        .join(Document)
        .filter(
            Document.chunking_strategy == strategy,
            DocumentChunk.embedding.isnot(None),
        )
        .scalar()
        or 0
    )

    return {
        "strategy": strategy,
        "postgres_chunks": pg_count,
        "embedded_chunks": embedded_count,
        "consistent": pg_count == embedded_count,
    }
