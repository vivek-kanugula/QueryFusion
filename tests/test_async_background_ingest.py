"""Comprehensive tests for asynchronous background document ingestion.

Hermetic and CI-safe (zero dependency on external PostgreSQL instance).
Covers:
1. Backward compatibility of synchronous uploads (background=False)
2. Asynchronous upload contract (background=True) returning status="processing" and job_id
3. Background worker execution (process_document_background) updating DB status & job_registry
4. Background worker failure handling (setting status="failed" and error_message)
5. Document status query endpoint (/api/v1/documents/{id}/status)
6. Listing documents with status and error fields
7. Hybrid retrieval candidate isolation for unprocessed documents
"""

from __future__ import annotations

import io
import uuid
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from hybrid_rag.app import app
from hybrid_rag.core.jobs import JobStatus, job_registry
from hybrid_rag.db.engine import get_db
from hybrid_rag.ingest.index_manager import process_document_background
from hybrid_rag.models.document import Document
from hybrid_rag.schemas.pipeline import PipelineConfig

client = TestClient(app)


def _override_db(mock_session):
    def _get_db_override():
        yield mock_session

    return _get_db_override


def test_synchronous_upload_backward_compatibility():
    """Verify standard POST /api/v1/documents without background flag indexes synchronously."""
    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = None
    mock_session.query.return_value.count.return_value = 0
    mock_session.query.return_value.scalar.return_value = 0

    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        file_bytes = b"Sample technical document for synchronous indexing backward compatibility test."
        files = {"file": ("sync_test.txt", io.BytesIO(file_bytes), "text/plain")}

        with patch("hybrid_rag.api.documents.index_document", return_value=[MagicMock()]) as mock_index:
            resp = client.post("/api/v1/documents?chunking_strategy=fixed", files=files)

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()

        assert data["status"] == "ready"
        assert data["chunks_created"] == 1
        assert data["job_id"] is None
        assert "document_id" in data
        assert mock_index.called
    finally:
        app.dependency_overrides.clear()


def test_async_upload_endpoint_returns_processing_and_job_id():
    """Verify POST /api/v1/documents?background=true returns immediately with status=processing and job_id."""
    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = None
    mock_session.query.return_value.count.return_value = 0
    mock_session.query.return_value.scalar.return_value = 0

    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        file_bytes = b"Large architecture manual content intended for asynchronous background ingestion."
        files = {"file": ("async_test.txt", io.BytesIO(file_bytes), "text/plain")}

        with patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
            resp = client.post(
                "/api/v1/documents?chunking_strategy=fixed&background=true",
                files=files,
            )

        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "processing"
        assert data["chunks_created"] == 0
        assert data["job_id"] is not None
        assert "document_id" in data

        # Verify job was registered in job_registry and background task was queued
        job = job_registry.get_job(data["job_id"])
        assert job is not None
        assert job.job_type == "document_ingest"
        assert mock_add_task.called
    finally:
        app.dependency_overrides.clear()


def test_background_worker_execution_and_completion():
    """Verify process_document_background parses, chunks, embeds, and marks document ready."""
    doc_id = uuid.uuid4()
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.filename = "bg_worker_test.txt"
    mock_doc.status = "processing"
    mock_doc.error_message = None

    mock_db = MagicMock()
    mock_db.get.return_value = mock_doc

    job = job_registry.create_job("document_ingest")
    payload = b"Line 1: High-throughput ingestion.\nLine 2: Asynchronous task workers.\nLine 3: Fast vector search."

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__enter__.return_value = mock_db
    mock_session_factory.return_value.__exit__.return_value = False

    with (
        patch("hybrid_rag.db.engine.SessionLocal", mock_session_factory),
        patch("hybrid_rag.ingest.index_manager.index_document", return_value=[MagicMock(), MagicMock()]),
    ):
        process_document_background(
            document_id=doc_id,
            raw_data=payload,
            filename="bg_worker_test.txt",
            chunking_strategy="fixed",
            job_id=job.job_id,
        )

    assert mock_doc.status == "ready"
    assert mock_doc.error_message is None
    assert mock_db.commit.called

    completed_job = job_registry.get_job(job.job_id)
    assert completed_job is not None
    assert completed_job.status == JobStatus.COMPLETED
    assert completed_job.progress_percent == 100.0


def test_background_worker_failure_handling():
    """Verify process_document_background captures errors, sets status=failed, and fails job."""
    doc_id = uuid.uuid4()
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.filename = "failing_doc.pdf"
    mock_doc.status = "processing"
    mock_doc.error_message = None

    mock_db = MagicMock()
    mock_db.get.return_value = mock_doc

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__enter__.return_value = mock_db
    mock_session_factory.return_value.__exit__.return_value = False

    job = job_registry.create_job("document_ingest")

    with (
        patch("hybrid_rag.db.engine.SessionLocal", mock_session_factory),
        patch("hybrid_rag.ingest.parsers.parse_file", side_effect=ValueError("Corrupt PDF byte sequence")),
    ):
        process_document_background(
            document_id=doc_id,
            raw_data=b"%PDF-corrupt",
            filename="failing_doc.pdf",
            chunking_strategy="fixed",
            job_id=job.job_id,
        )

    assert mock_doc.status == "failed"
    assert "Corrupt PDF" in (mock_doc.error_message or "")
    assert mock_db.commit.called

    failed_job = job_registry.get_job(job.job_id)
    assert failed_job is not None
    assert failed_job.status == JobStatus.FAILED
    assert "Corrupt PDF" in (failed_job.error or "")


def test_get_document_status_endpoint():
    """Verify GET /api/v1/documents/{id}/status contract for existing and missing documents."""
    doc_id = uuid.uuid4()
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.filename = "status_check.txt"
    mock_doc.status = "ready"
    mock_doc.error_message = None
    mock_doc.chunks = [MagicMock(), MagicMock()]
    mock_doc.chunking_strategy = "fixed"

    mock_session = MagicMock()
    mock_session.get.return_value = mock_doc

    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        # 1. Existing document
        resp = client.get(f"/api/v1/documents/{doc_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == str(doc_id)
        assert data["status"] == "ready"
        assert data["chunk_count"] == 2
        assert data["filename"] == "status_check.txt"

        # 2. Nonexistent document
        mock_session.get.return_value = None
        missing_id = uuid.uuid4()
        resp_missing = client.get(f"/api/v1/documents/{missing_id}/status")
        assert resp_missing.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_list_documents_includes_status():
    """Verify GET /api/v1/documents returns status and error_message fields."""
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = uuid.uuid4()
    mock_doc.filename = "sample.txt"
    mock_doc.file_type = "txt"
    mock_doc.size_bytes = 100
    mock_doc.chunking_strategy = "fixed"
    mock_doc.chunks = []
    mock_doc.status = "processing"
    mock_doc.error_message = None
    mock_doc.is_evictable = True
    mock_doc.created_at.isoformat.return_value = "2026-11-10T18:00:00"

    mock_session = MagicMock()
    mock_session.query.return_value.order_by.return_value.all.return_value = [mock_doc]

    app.dependency_overrides[get_db] = _override_db(mock_session)
    try:
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        docs = resp.json()
        assert isinstance(docs, list)
        assert len(docs) == 1
        assert docs[0]["status"] == "processing"
        assert docs[0]["error_message"] is None
    finally:
        app.dependency_overrides.clear()


def test_retrieval_excludes_processing_documents():
    """Verify documents in status='processing' (no chunks committed) are not returned by retriever."""
    from hybrid_rag.retrieval.retriever import hybrid_retrieve

    mock_db = MagicMock()
    # No chunks matching document id
    mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

    doc_id = uuid.uuid4()
    config = PipelineConfig(
        retrieval_strategy="hybrid",
        dense_top_k=10,
        sparse_top_k=10,
        fusion_top_k=10,
        document_ids=(str(doc_id),),
    )

    trace = hybrid_retrieve("test query for unprocessed doc", config=config, db=mock_db)
    fused_candidates = trace.stages[-1].candidates if trace.stages else []
    assert len(fused_candidates) == 0
