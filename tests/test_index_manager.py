"""Tests for index manager (PG+BM25 saga pattern, consistency)."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.ingest.chunkers import ChunkOutput
from hybrid_rag.ingest.index_manager import (
    check_consistency,
    delete_document,
    index_document,
)
from hybrid_rag.models.document import Document


def _make_doc(doc_id=None):
    return Document(
        id=doc_id or uuid.uuid4(),
        filename="test.txt",
        file_type="txt",
        file_hash="abc123def456",
        size_bytes=100,
        chunking_strategy="fixed",
    )


def _make_chunks(n=3):
    return [
        ChunkOutput(
            text=f"Chunk {i} content.",
            chunk_index=i,
            token_count=5,
            char_offset_start=i * 20,
            char_offset_end=(i + 1) * 20,
        )
        for i in range(n)
    ]


@patch("hybrid_rag.retrieval.bm25_store.get_bm25_store")
@patch("hybrid_rag.ingest.index_manager.embed_texts")
def test_index_document_writes_pg_and_rebuilds_bm25(mock_embed, mock_get_bm25):
    import numpy as np

    mock_embed.return_value = [np.zeros(384) for _ in range(3)]
    mock_bm25 = MagicMock()
    mock_bm25.add_chunks.return_value = 3
    mock_get_bm25.return_value = mock_bm25

    db = MagicMock()
    doc = _make_doc()
    chunks = _make_chunks(3)

    result = index_document(db, doc, chunks, "fixed")

    assert len(result) == 3
    db.add.assert_called_once_with(doc)
    db.add_all.assert_called_once()
    db.flush.assert_called_once()
    db.commit.assert_called_once()
    mock_bm25.add_chunks.assert_called_once()


def test_delete_document_removes_from_postgres():
    db = MagicMock()
    doc_id = uuid.uuid4()
    mock_doc = _make_doc(doc_id)
    db.get.return_value = mock_doc

    delete_document(db, doc_id, "fixed")

    db.delete.assert_called_once_with(mock_doc)
    db.commit.assert_called_once()


def test_delete_document_not_found():
    db = MagicMock()
    db.get.return_value = None

    delete_document(db, uuid.uuid4(), "fixed")

    db.delete.assert_not_called()


def test_check_consistency_matching():
    db = MagicMock()
    mock_query = MagicMock()
    mock_query.join.return_value.filter.return_value.scalar.side_effect = [10, 10]
    db.query.return_value = mock_query

    result = check_consistency(db, "fixed")
    assert result["consistent"] is True
    assert result["postgres_chunks"] == 10
    assert result["embedded_chunks"] == 10


def test_check_consistency_mismatch():
    db = MagicMock()
    mock_query = MagicMock()
    mock_query.join.return_value.filter.return_value.scalar.side_effect = [10, 8]
    db.query.return_value = mock_query

    result = check_consistency(db, "fixed")
    assert result["consistent"] is False
    assert result["postgres_chunks"] == 10
    assert result["embedded_chunks"] == 8


@patch("hybrid_rag.retrieval.bm25_store.get_bm25_store")
@patch("hybrid_rag.ingest.index_manager.embed_texts")
def test_index_document_pgvector_and_bm25_rebuild(mock_embed, mock_get_bm25):
    import numpy as np

    mock_embed.return_value = [np.ones(384) for _ in range(2)]
    mock_bm25 = MagicMock()
    mock_bm25.add_chunks.return_value = 2
    mock_get_bm25.return_value = mock_bm25

    db = MagicMock()
    doc = _make_doc()
    chunks = _make_chunks(2)

    result = index_document(db, doc, chunks, "fixed")
    assert len(result) == 2
    assert result[0].embedding == [1.0] * 384
    db.commit.assert_called_once()
    mock_bm25.add_chunks.assert_called_once()


def test_check_consistency_pgvector_mode():
    db = MagicMock()
    mock_query = MagicMock()
    mock_query.join.return_value.filter.return_value.scalar.side_effect = [15, 15]
    db.query.return_value = mock_query

    result = check_consistency(db, "fixed")
    assert result["consistent"] is True
    assert result["postgres_chunks"] == 15
    assert result["embedded_chunks"] == 15


@patch("hybrid_rag.ingest.index_manager.embed_texts")
def test_index_document_commit_failure_triggers_rollback(mock_embed):
    """Verify that when db.commit() fails, DB is rolled back and exception is re-raised."""
    import numpy as np

    mock_embed.return_value = [np.zeros(384) for _ in range(3)]
    db = MagicMock()
    db.commit.side_effect = RuntimeError("Database disk full")

    doc = _make_doc()
    chunks = _make_chunks(3)

    with pytest.raises(RuntimeError, match="Database disk full"):
        index_document(db, doc, chunks, "fixed")

    db.rollback.assert_called_once()


@patch("hybrid_rag.retrieval.bm25_store.get_bm25_store")
@patch("hybrid_rag.ingest.index_manager.embed_texts")
def test_index_document_bm25_failure_triggers_rollback(mock_embed, mock_get_bm25):
    """Verify that when BM25 rebuild fails, DB is rolled back and db.commit() is never called."""
    import numpy as np

    mock_embed.return_value = [np.zeros(384) for _ in range(3)]
    mock_bm25 = MagicMock()
    mock_bm25.add_chunks.return_value = -1
    mock_bm25.rebuild_from_db.side_effect = [
        RuntimeError("Stemmer internal memory error"),
        0,
    ]
    mock_get_bm25.return_value = mock_bm25

    db = MagicMock()
    doc = _make_doc()
    chunks = _make_chunks(3)

    with pytest.raises(RuntimeError, match="Stemmer internal memory error"):
        index_document(db, doc, chunks, "fixed")

    db.commit.assert_not_called()
    db.rollback.assert_called_once()


def test_delete_document_db_error_triggers_rollback():
    """Verify that when db.commit() fails during delete_document, rollback is executed."""
    db = MagicMock()
    doc_id = uuid.uuid4()
    mock_doc = _make_doc(doc_id)
    db.get.return_value = mock_doc
    db.commit.side_effect = RuntimeError("Postgres constraint error on delete")

    with pytest.raises(RuntimeError, match="Postgres constraint error on delete"):
        delete_document(db, doc_id, "fixed")

    db.delete.assert_called_once_with(mock_doc)
    db.rollback.assert_called_once()
