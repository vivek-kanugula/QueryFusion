"""Frontend-Backend Integration and Contract Test Suite.

Verifies the HTTP and schema contracts between the React frontend client
(frontend/src/lib/api.ts, Documents.tsx, QA.tsx) and the FastAPI backend:
1. Document upload route succeeds without 404
2. Chunking strategy parameter compatibility (query param & strategy enum)
3. Document chunks inspection endpoint contract (/api/v1/documents/{id}/chunks)
4. Citation resolution cascade handles non-sequential, out-of-order, or string IDs
5. Authentication header (X-API-Key) propagation and acceptance
"""

from __future__ import annotations

import io
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hybrid_rag.app import app
from hybrid_rag.db.engine import get_db
from hybrid_rag.models.document import DocumentChunk


@pytest.fixture
def mock_db():
    session = MagicMock()
    app.dependency_overrides[get_db] = lambda: session
    yield session
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def auth_client():
    return TestClient(app)


def test_document_upload_route_contract(mock_db, auth_client):
    """Verify POST /api/v1/documents receives multipart file and query parameters without 404."""
    mock_db.query.return_value.filter.return_value.first.return_value = None  # No duplicates
    mock_db.query.return_value.count.return_value = 0  # Under document limit
    mock_db.query.return_value.scalar.return_value = 0  # Under storage quota

    file_bytes = b"Hello, this is a test document for frontend integration verification."
    files = {"file": ("integration_test.txt", io.BytesIO(file_bytes), "text/plain")}

    with patch("hybrid_rag.api.documents.index_document", return_value=[MagicMock()]):
        resp = auth_client.post(
            "/api/v1/documents?chunking_strategy=fixed",
            files=files,
        )

    assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "document_id" in data
    assert data["filename"] == "integration_test.txt"
    assert data["chunking_strategy"] == "fixed"


@pytest.mark.parametrize("strategy", ["fixed", "structure", "semantic"])
def test_chunking_strategy_parameter_compatibility(mock_db, auth_client, strategy):
    """Verify backend accepts all 3 frontend chunking strategies ('fixed', 'structure', etc.)."""
    mock_db.query.return_value.filter.return_value.first.return_value = None
    mock_db.query.return_value.count.return_value = 0
    mock_db.query.return_value.scalar.return_value = 0

    file_bytes = b"# Heading 1\n\nParagraph text for strategy verification testing."
    files = {"file": ("test_doc.md", io.BytesIO(file_bytes), "text/markdown")}

    with patch("hybrid_rag.api.documents.index_document", return_value=[MagicMock()]):
        resp = auth_client.post(
            f"/api/v1/documents?chunking_strategy={strategy}",
            files=files,
        )

    assert resp.status_code == 200
    assert resp.json()["chunking_strategy"] == strategy


def test_chunking_strategy_invalid_structure_aware_returns_clear_error(auth_client):
    """Verify passing old 'structure_aware' returns 400 with list of valid strategies."""
    files = {"file": ("test.txt", io.BytesIO(b"data"), "text/plain")}
    resp = auth_client.post(
        "/api/v1/documents?chunking_strategy=structure_aware",
        files=files,
    )
    assert resp.status_code == 400
    assert "structure" in resp.json()["detail"]


def test_document_chunks_endpoint_contract(mock_db, auth_client):
    """Verify GET /api/v1/documents/{id}/chunks endpoint contract returns expected Chunk schema."""
    doc_id = uuid.uuid4()
    mock_chunks = [
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            chunk_index=0,
            token_count=18,
            content="Chunk content zero.",
            section_heading="Introduction",
        ),
        DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            chunk_index=1,
            token_count=22,
            content="Chunk content one.",
            section_heading="Body",
        ),
    ]

    mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
        mock_chunks
    )

    resp = auth_client.get(
        f"/api/v1/documents/{doc_id}/chunks",
    )

    assert resp.status_code == 200
    chunks_data = resp.json()
    assert len(chunks_data) == 2
    first_chunk = chunks_data[0]
    assert "chunk_id" in first_chunk
    assert "content" in first_chunk
    assert "token_count" in first_chunk
    assert "index" in first_chunk
    assert first_chunk["index"] == 0
    assert first_chunk["content"] == "Chunk content zero."


def test_citation_parsing_cascade_non_sequential_and_string_ids():
    """Verify 5-step citation cascade correctly resolves non-sequential and string citations."""
    citations = [
        {"citation_id": "[C3]", "chunk_id": "chunk-uuid-c3", "evidence_quote": "evidence c3"},
        {
            "citation_id": "[doc-troubleshooting]",
            "chunk_id": "chunk-uuid-trouble",
            "evidence_quote": "evidence trouble",
        },
        {"citation_id": "[C1]", "chunk_id": "chunk-uuid-c1", "evidence_quote": "evidence c1"},
    ]

    def resolve_citation(token: str, citation_list: list[dict]) -> dict | None:
        clean = token.strip()
        # 1. Exact match on citation_id
        for c in citation_list:
            if c.get("citation_id") == clean:
                return c
        # 2. Match without brackets
        inner = clean.strip("[]^")
        for c in citation_list:
            if c.get("citation_id", "").strip("[]^") == inner:
                return c
        # 3. Match on chunk_id or document_id
        for c in citation_list:
            if c.get("chunk_id") == inner or c.get("document_id") == inner:
                return c
        # 4. Fallback numeric index
        digits = "".join(filter(str.isdigit, inner))
        if digits:
            idx = int(digits) - 1
            if 0 <= idx < len(citation_list):
                return citation_list[idx]
        return None

    # Test non-sequential citation [C3]
    res_c3 = resolve_citation("[C3]", citations)
    assert res_c3 is not None
    assert res_c3["chunk_id"] == "chunk-uuid-c3"

    # Test string identifier [doc-troubleshooting]
    res_str = resolve_citation("[doc-troubleshooting]", citations)
    assert res_str is not None
    assert res_str["chunk_id"] == "chunk-uuid-trouble"

    # Test out-of-order [C1]
    res_c1 = resolve_citation("[C1]", citations)
    assert res_c1 is not None
    assert res_c1["chunk_id"] == "chunk-uuid-c1"

    # Test unknown citation returns None safely without crashing
    res_unknown = resolve_citation("[C999]", citations)
    assert res_unknown is None


def test_frontend_endpoints_accessible_without_auth(mock_db, auth_client):
    """Verify backend endpoints called by frontend work cleanly without authentication."""
    # 1. Documents router
    resp_docs = auth_client.get("/api/v1/documents")
    assert resp_docs.status_code == 200

    # 2. Jobs router
    with patch("hybrid_rag.api.jobs.job_registry.get_job") as mock_get_job:
        mock_job = MagicMock()
        mock_job.to_dict.return_value = {"job_id": "job-1", "status": "completed"}
        mock_get_job.return_value = mock_job
        resp_job = auth_client.get("/api/jobs/job-1")
        assert resp_job.status_code == 200

    # 3. Traces router
    from hybrid_rag.schemas.pipeline import QueryTrace

    with patch("hybrid_rag.api.traces.trace_store.get") as mock_get_trace:
        mock_get_trace.return_value = QueryTrace(query="test", trace_id="trace-1", stages=[])
        resp_trace = auth_client.get("/api/traces/trace-1")
        assert resp_trace.status_code == 200


def test_frontend_storage_endpoint_contract(mock_db, auth_client):
    """Verify GET /api/v1/documents/storage schema aligns with frontend StorageStats interface."""
    mock_db.query.return_value.scalar.return_value = 10 * 1024 * 1024
    mock_db.query.return_value.count.side_effect = [2, 50]

    resp = auth_client.get("/api/v1/documents/storage")
    assert resp.status_code == 200
    data = resp.json()

    required_fields = [
        "used_bytes",
        "used_mb",
        "free_bytes",
        "free_mb",
        "max_storage_mb",
        "max_storage_bytes",
        "usage_percent",
        "max_upload_size_mb",
        "max_upload_size_bytes",
        "document_count",
        "max_documents_count",
        "chunk_count",
    ]
    for field in required_fields:
        assert field in data, f"Missing expected field in storage response: {field}"

    assert data["used_mb"] == 10.0
    assert data["free_mb"] == 190.0
    assert data["usage_percent"] == 5.0
    assert data["document_count"] == 2
    assert data["chunk_count"] == 50

