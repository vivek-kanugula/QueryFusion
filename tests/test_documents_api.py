"""Tests for document API endpoints."""

import io
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from hybrid_rag.api.documents import upload_document
from hybrid_rag.app import app
from hybrid_rag.db.engine import get_db

client = TestClient(app)


def _override_db(mock_session):
    """Return a FastAPI dependency override that yields mock_session."""

    def _get_db_override():
        yield mock_session

    return _get_db_override


def test_upload_document_oversized():
    mock_session = MagicMock()
    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        big_data = b"x" * (51 * 1024 * 1024)
        response = client.post(
            "/api/v1/documents?chunking_strategy=fixed",
            files={"file": ("big.txt", io.BytesIO(big_data), "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 413


def test_upload_duplicate():
    from hybrid_rag.models.document import Document

    mock_session = MagicMock()
    existing = MagicMock(spec=Document)
    existing.id = uuid.uuid4()
    mock_session.query.return_value.filter.return_value.first.return_value = existing

    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        response = client.post(
            "/api/v1/documents?chunking_strategy=fixed",
            files={"file": ("test.txt", io.BytesIO(b"duplicate content"), "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 409


def test_upload_invalid_strategy():
    mock_session = MagicMock()
    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        response = client.post(
            "/api/v1/documents?chunking_strategy=invalid_strategy",
            files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 400


def test_list_documents():
    mock_session = MagicMock()
    mock_session.query.return_value.order_by.return_value.all.return_value = []
    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        response = client.get("/api/v1/documents")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == []


def test_delete_not_found():
    fake_id = str(uuid.uuid4())
    mock_session = MagicMock()
    mock_session.get.return_value = None
    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        response = client.delete(f"/api/v1/documents/{fake_id}")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 404


def test_batch_delete_empty():
    response = client.post(
        "/api/v1/documents/batch-delete",
        json={"document_ids": []},
    )
    assert response.status_code == 400


def test_consistency_check():
    mock_session = MagicMock()
    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        with patch("hybrid_rag.api.documents.check_consistency") as mock_check:
            mock_check.return_value = {
                "strategy": "fixed",
                "postgres_chunks": 10,
                "embedded_chunks": 10,
                "consistent": True,
            }
            response = client.get("/api/v1/documents/consistency?strategy=fixed")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["consistent"] is True


def test_upload_storage_quota_exceeded():
    """Verify that uploading a document when storage quota is reached returns 422."""
    mock_session = MagicMock()
    # Mock current storage already at 200 MB limit
    mock_session.query.return_value.scalar.return_value = 200 * 1024 * 1024
    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        response = client.post(
            "/api/v1/documents?chunking_strategy=fixed",
            files={"file": ("doc.txt", io.BytesIO(b"new document content"), "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 422
    assert "Storage quota exceeded" in response.json()["detail"]


def test_upload_streaming_aborts_early_and_closes():
    """Verify streaming reads abort early at limit without reading entire stream,
    and guarantee handle closure."""

    class TrackedStream(io.BytesIO):
        def __init__(self, data):
            super().__init__(data)
            self.read_calls = 0
            self.bytes_yielded = 0
            self.was_closed = False

        def read(self, size=-1):
            self.read_calls += 1
            chunk = super().read(size)
            self.bytes_yielded += len(chunk)
            return chunk

        def close(self):
            self.was_closed = True
            super().close()

    # 100 MB data stream (50MB is the limit)
    big_data = b"x" * (100 * 1024 * 1024)
    stream = TrackedStream(big_data)
    upload_file = UploadFile(file=stream, filename="huge.txt")

    mock_db = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        upload_document(file=upload_file, chunking_strategy="fixed", db=mock_db)

    assert exc_info.value.status_code == 413
    assert "File exceeds 50 MB limit" in exc_info.value.detail
    # Must have aborted at 51 MB (51 chunks of 1MB), NOT reading the full 100MB
    assert stream.bytes_yielded == 51 * 1024 * 1024
    assert stream.read_calls == 51
    # File handle must be closed by finally block
    assert stream.was_closed is True


def test_get_storage_stats_empty():
    """Verify GET /api/v1/documents/storage returns accurate metrics for empty DB."""
    mock_session = MagicMock()
    # sum(size_bytes) is None/0
    mock_session.query.return_value.scalar.return_value = 0
    # count() is 0
    mock_session.query.return_value.count.return_value = 0

    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        response = client.get("/api/v1/documents/storage")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["used_bytes"] == 0
    assert data["used_mb"] == 0.0
    assert data["free_mb"] == 200.0
    assert data["max_storage_mb"] == 200
    assert data["usage_percent"] == 0.0
    assert data["max_upload_size_mb"] == 50
    assert data["document_count"] == 0
    assert data["max_documents_count"] == 50
    assert data["chunk_count"] == 0


def test_get_storage_stats_populated():
    """Verify GET /api/v1/documents/storage returns accurate calculated headroom and usage."""
    mock_session = MagicMock()
    # 50 MB used = 52428800 bytes
    mock_session.query.return_value.scalar.return_value = 50 * 1024 * 1024
    # 3 documents, 150 chunks
    mock_session.query.return_value.count.side_effect = [3, 150]

    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        response = client.get("/api/v1/documents/storage")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["used_bytes"] == 50 * 1024 * 1024
    assert data["used_mb"] == 50.0
    assert data["free_mb"] == 150.0
    assert data["usage_percent"] == 25.0
    assert data["document_count"] == 3
    assert data["chunk_count"] == 150

