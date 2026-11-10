"""Comprehensive tests for asynchronous background document ingestion.

Covers:
1. Backward compatibility of synchronous uploads (background=False)
2. Asynchronous upload contract (background=True) returning status="processing" and job_id
3. Background worker execution (process_document_background) updating DB status & job_registry
4. Background worker failure handling (setting status="failed" and error_message)
5. Document status query endpoint (/api/v1/documents/{id}/status)
6. Listing documents with status and error fields
7. End-to-end integration via FastAPI TestClient
"""

import io
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from hybrid_rag.app import app
from hybrid_rag.core.jobs import JobStatus, job_registry
from hybrid_rag.db.engine import SessionLocal
from hybrid_rag.ingest.index_manager import process_document_background
from hybrid_rag.models.document import Document

client = TestClient(app)


def test_synchronous_upload_backward_compatibility():
    """Verify standard POST /api/v1/documents without background flag indexes synchronously."""
    unique_id = uuid.uuid4()
    file_bytes = f"Sample technical document {unique_id} for synchronous indexing backward compatibility test.".encode()
    files = {"file": (f"sync_test_{unique_id}.txt", io.BytesIO(file_bytes), "text/plain")}

    resp = client.post("/api/v1/documents?chunking_strategy=fixed", files=files)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    assert data["status"] == "ready"
    assert data["chunks_created"] > 0
    assert data["job_id"] is None
    assert "document_id" in data

    doc_id = uuid.UUID(data["document_id"])
    with SessionLocal() as db:
        doc = db.get(Document, doc_id)
        assert doc is not None
        assert doc.status == "ready"
        assert doc.error_message is None
        assert len(doc.chunks) > 0


def test_async_upload_endpoint_returns_processing_and_job_id():
    """Verify POST /api/v1/documents?background=true returns immediately with status=processing and job_id."""
    unique_id = uuid.uuid4()
    file_bytes = f"Large architecture manual content {unique_id} intended for asynchronous background ingestion.".encode()
    files = {"file": (f"async_test_{unique_id}.txt", io.BytesIO(file_bytes), "text/plain")}

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

    # Verify job was registered in job_registry
    job = job_registry.get_job(data["job_id"])
    assert job is not None
    assert job.job_type == "document_ingest"


def test_background_worker_execution_and_completion():
    """Verify process_document_background parses, chunks, embeds, and marks document ready."""
    with SessionLocal() as db:
        doc = Document(
            id=uuid.uuid4(),
            filename="bg_worker_test.txt",
            file_type="txt",
            file_hash=f"hash_{uuid.uuid4()}",
            size_bytes=120,
            chunking_strategy="fixed",
            status="processing",
            is_evictable=True,
        )
        db.add(doc)
        db.commit()
        doc_id = doc.id

    job = job_registry.create_job("document_ingest")
    payload = b"Line 1: High-throughput ingestion.\nLine 2: Asynchronous task workers.\nLine 3: Fast vector search."

    process_document_background(
        document_id=doc_id,
        raw_data=payload,
        filename="bg_worker_test.txt",
        chunking_strategy="fixed",
        job_id=job.job_id,
    )

    with SessionLocal() as db:
        updated_doc = db.get(Document, doc_id)
        assert updated_doc is not None
        assert updated_doc.status == "ready"
        assert updated_doc.error_message is None
        assert len(updated_doc.chunks) > 0

    completed_job = job_registry.get_job(job.job_id)
    assert completed_job is not None
    assert completed_job.status == JobStatus.COMPLETED
    assert completed_job.progress_percent == 100.0


def test_background_worker_failure_handling():
    """Verify process_document_background captures errors, sets status=failed, and fails job."""
    with SessionLocal() as db:
        doc = Document(
            id=uuid.uuid4(),
            filename="failing_doc.pdf",
            file_type="pdf",
            file_hash=f"hash_{uuid.uuid4()}",
            size_bytes=100,
            chunking_strategy="fixed",
            status="processing",
            is_evictable=True,
        )
        db.add(doc)
        db.commit()
        doc_id = doc.id

    job = job_registry.create_job("document_ingest")

    with patch("hybrid_rag.ingest.parsers.parse_file", side_effect=ValueError("Corrupt PDF byte sequence")):
        process_document_background(
            document_id=doc_id,
            raw_data=b"%PDF-corrupt",
            filename="failing_doc.pdf",
            chunking_strategy="fixed",
            job_id=job.job_id,
        )

    with SessionLocal() as db:
        failed_doc = db.get(Document, doc_id)
        assert failed_doc is not None
        assert failed_doc.status == "failed"
        assert "Corrupt PDF" in (failed_doc.error_message or "")

    failed_job = job_registry.get_job(job.job_id)
    assert failed_job is not None
    assert failed_job.status == JobStatus.FAILED
    assert "Corrupt PDF" in (failed_job.error or "")


def test_get_document_status_endpoint():
    """Verify GET /api/v1/documents/{id}/status contract for existing and missing documents."""
    with SessionLocal() as db:
        doc = Document(
            id=uuid.uuid4(),
            filename="status_check.txt",
            file_type="txt",
            file_hash=f"hash_{uuid.uuid4()}",
            size_bytes=50,
            chunking_strategy="fixed",
            status="ready",
            is_evictable=True,
        )
        db.add(doc)
        db.commit()
        doc_id = doc.id

    # 1. Existing document
    resp = client.get(f"/api/v1/documents/{doc_id}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == str(doc_id)
    assert data["status"] == "ready"
    assert data["filename"] == "status_check.txt"

    # 2. Nonexistent document
    missing_id = uuid.uuid4()
    resp_missing = client.get(f"/api/v1/documents/{missing_id}/status")
    assert resp_missing.status_code == 404


def test_list_documents_includes_status():
    """Verify GET /api/v1/documents returns status and error_message fields."""
    resp = client.get("/api/v1/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list)
    if len(docs) > 0:
        first = docs[0]
        assert "status" in first
        assert "error_message" in first


def test_retrieval_excludes_processing_documents():
    """Verify documents in status='processing' (no chunks committed) are not returned by retriever."""
    from hybrid_rag.retrieval.retriever import hybrid_retrieve
    from hybrid_rag.schemas.pipeline import PipelineConfig

    with SessionLocal() as db:
        doc = Document(
            id=uuid.uuid4(),
            filename="unprocessed.txt",
            file_type="txt",
            file_hash=f"hash_{uuid.uuid4()}",
            size_bytes=50,
            chunking_strategy="fixed",
            status="processing",
            is_evictable=True,
        )
        db.add(doc)
        db.commit()
        doc_id = doc.id

    config = PipelineConfig(
        retrieval_strategy="hybrid",
        dense_top_k=10,
        sparse_top_k=10,
        fusion_top_k=10,
        document_ids=(str(doc_id),),
    )

    trace = hybrid_retrieve("test query for unprocessed doc", config=config)
    fused_candidates = trace.stages[-1].candidates if trace.stages else []
    assert len(fused_candidates) == 0


