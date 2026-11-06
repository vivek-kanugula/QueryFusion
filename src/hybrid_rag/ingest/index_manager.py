"""Index Manager: orchestration for write/delete across PG and BM25.

Write order: Postgres (with pgvector embeddings) -> BM25
Delete order: Postgres (cascade) -> BM25 rebuild
"""

from __future__ import annotations

import logging
import uuid

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
) -> list[DocumentChunk]:
    """Write chunks with pgvector embeddings to Postgres and rebuild BM25.

    Uses compensating rollback if any phase fails.
    """
    if document.id is None:
        document.id = uuid.uuid4()

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts) if texts else []

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

        get_bm25_store().rebuild_from_db(db, strategy)

        db.commit()

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
            from hybrid_rag.retrieval.bm25_store import get_bm25_store

            get_bm25_store().rebuild_from_db(db, strategy)
        except Exception:
            logger.warning(
                "BM25 index rebuild failed during compensating rollback for document %s",
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


def delete_document(
    db: Session,
    document_id: uuid.UUID,
    strategy: str = "fixed",
) -> None:
    """Delete document from Postgres (cascading to chunks/vectors) and rebuild BM25."""
    doc = db.get(Document, document_id)
    if doc:
        try:
            db.delete(doc)
            db.commit()
            logger.info("Deleted document %s from Postgres (cascade to chunks)", document_id)
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

        get_bm25_store().rebuild_from_db(db, strategy)
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
