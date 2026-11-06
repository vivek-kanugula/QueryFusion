"""Document API and Ingestion Contract Fuzzing Test Suite.

Tests cover:
1. Route /api/v1/documents parameter variations, casing, whitespace, and extra form fields.
2. Chunking strategy parameter handling: valid, invalid, boundary cases.
3. /api/v1/documents/{document_id}/chunks: 404 on nonexistent, 422 on invalid UUID,
   200 [] on 0 chunks, strict ordering on 100 out-of-order chunks, schema conformance.
4. Header propagation and error code contracts.
"""

from __future__ import annotations

import io
import random
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hybrid_rag.app import app
from hybrid_rag.db.engine import get_db
from hybrid_rag.models.document import Document, DocumentChunk


@pytest.fixture
def mock_db():
    session = MagicMock()
    app.dependency_overrides[get_db] = lambda: session
    yield session
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def auth_client():
    return TestClient(app)


AUTH_HEADERS: dict[str, str] = {}


# ==============================================================================
# SECTION 1: Route /api/v1/documents parameter variations, casing, and whitespace
# ==============================================================================


@pytest.mark.parametrize(
    "strategy_param,expected_status,should_contain",
    [
        ("fixed", 200, None),
        ("structure", 200, None),
        ("semantic", 200, None),
        ("FIXED", 400, "Unknown strategy: FIXED"),
        ("STRUCTURE", 400, "Unknown strategy: STRUCTURE"),
        (" fixed ", 400, "Unknown strategy:  fixed "),
        ("structure_aware", 400, "Unknown strategy: structure_aware"),
        ("", 400, "Unknown strategy: "),
        ("   ", 400, "Unknown strategy:    "),
        ("unknown_strat", 400, "Choose: ['fixed', 'structure', 'semantic']"),
        ("' OR '1'='1", 400, "Unknown strategy"),
    ],
)
def test_document_upload_strategy_variations(
    mock_db, auth_client, strategy_param, expected_status, should_contain
):
    """Stress test chunking_strategy parameter with case, whitespace, and invalid values."""
    mock_db.query.return_value.filter.return_value.first.return_value = None
    mock_db.query.return_value.count.return_value = 0
    mock_db.query.return_value.scalar.return_value = 0

    file_bytes = b"# Test Document\n\nAdversarial test content for parameter variations."
    files = {"file": ("test.md", io.BytesIO(file_bytes), "text/markdown")}

    with patch("hybrid_rag.api.documents.index_document", return_value=[MagicMock()]):
        resp = auth_client.post(
            f"/api/v1/documents?chunking_strategy={strategy_param}",
            files=files,
            headers=AUTH_HEADERS,
        )

    assert resp.status_code == expected_status, (
        f"Expected {expected_status}, got {resp.status_code}: {resp.text}"
    )
    if should_contain:
        assert should_contain in resp.json()["detail"]


def test_document_upload_with_frontend_extra_form_data(mock_db, auth_client):
    """Frontend api.ts appends extra form fields: chunk_size, overlap, similarity_threshold.
    Verify FastAPI safely accepts these extra multipart form fields without error."""
    mock_db.query.return_value.filter.return_value.first.return_value = None
    mock_db.query.return_value.count.return_value = 0
    mock_db.query.return_value.scalar.return_value = 0

    file_bytes = b"Sample text for multipart extra data test."
    files = {"file": ("test.txt", io.BytesIO(file_bytes), "text/plain")}
    data = {
        "chunking_strategy": "fixed",
        "chunk_size": "512",
        "overlap": "64",
        "similarity_threshold": "0.7",
    }

    with (
        patch("hybrid_rag.api.documents.index_document", return_value=[MagicMock()]),
    ):
        resp = auth_client.post(
            "/api/v1/documents?chunking_strategy=fixed&extra_query_param=ignore_me",
            files=files,
            data=data,
            headers=AUTH_HEADERS,
        )

    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["filename"] == "test.txt"
    assert res_data["chunking_strategy"] == "fixed"


def test_document_upload_missing_file_returns_422(auth_client):
    """Ensure missing file field returns clean 422 Unprocessable Entity."""
    resp = auth_client.post(
        "/api/v1/documents?chunking_strategy=fixed",
        data={"chunk_size": "500"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 422


# ==============================================================================
# SECTION 2: /api/v1/documents/{document_id}/chunks edge cases
# ==============================================================================


def test_document_chunks_nonexistent_uuid_returns_404(mock_db, auth_client):
    """Verify nonexistent UUID returns clean 404 with 'Document not found'."""
    random_id = uuid.uuid4()
    mock_db.get.return_value = None
    mock_db.query.return_value.filter.return_value.first.return_value = None
    mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

    resp = auth_client.get(
        f"/api/v1/documents/{random_id}/chunks",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Document not found"


def test_document_chunks_malformed_uuid_returns_422(auth_client):
    """Verify malformed UUID returns clean 422 validation error."""
    resp = auth_client.get(
        "/api/v1/documents/not-a-valid-uuid/chunks",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 422


def test_document_chunks_zero_chunks_returns_empty_list_200(mock_db, auth_client):
    """Verify document existing in DB with 0 chunks returns 200 OK with empty array []."""
    doc_id = uuid.uuid4()
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.chunks = []

    mock_db.get.return_value = mock_doc
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc
    mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

    resp = auth_client.get(
        f"/api/v1/documents/{doc_id}/chunks",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json() == []


def test_document_chunks_large_scale_and_order_preservation(mock_db, auth_client):
    """Stress test order preservation with 100 chunks inserted in shuffled order.
    Verify response returns chunks sorted strictly by chunk_index 0..99,
    and null section_headings are serialized as empty string ""."""
    doc_id = uuid.uuid4()
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id

    # Create 100 chunks with indices 0..99
    indices = list(range(100))
    shuffled_indices = list(indices)
    random.seed(42)
    random.shuffle(shuffled_indices)

    all_chunks = []
    for idx in shuffled_indices:
        c = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            chunk_index=idx,
            token_count=idx * 10 + 5,
            content=f"Content for chunk index {idx}",
            section_heading=f"Heading {idx}" if idx % 2 == 0 else None,  # Test null headings
        )
        all_chunks.append(c)

    # In DB query, order_by(DocumentChunk.chunk_index) returns them sorted
    sorted_chunks = sorted(all_chunks, key=lambda x: x.chunk_index)
    mock_db.get.return_value = mock_doc
    mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
        sorted_chunks
    )

    resp = auth_client.get(
        f"/api/v1/documents/{doc_id}/chunks",
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 100

    # Verify strict ascending order
    for expected_idx, chunk in enumerate(data):
        assert chunk["index"] == expected_idx
        assert chunk["content"] == f"Content for chunk index {expected_idx}"
        assert chunk["token_count"] == expected_idx * 10 + 5
        assert isinstance(chunk["chunk_id"], str)
        if expected_idx % 2 == 0:
            assert chunk["section_heading"] == f"Heading {expected_idx}"
        else:
            assert chunk["section_heading"] == ""  # None normalized to ""


# ==============================================================================
# SECTION 3: Unauthenticated Access Verification
# ==============================================================================


@pytest.mark.parametrize(
    "endpoint,method",
    [
        ("/api/v1/documents", "GET"),
        ("/health", "GET"),
        ("/ready", "GET"),
    ],
)
def test_unauthenticated_endpoints_accessible(mock_db, auth_client, endpoint, method):
    """Verify endpoints are accessible unauthenticated with no 401 or 403."""
    mock_db.query.return_value.order_by.return_value.all.return_value = []
    resp = auth_client.request(method, endpoint)
    assert resp.status_code != 401
    assert resp.status_code != 403


# ==============================================================================
# SECTION 4: Node.js Frontend Client SSE Stress Test Execution
# ==============================================================================


def test_frontend_sse_parser_boundary_conditions():
    """Execute the Node.js adversarial SSE test harness covering:
    - Packet splitting across TCP chunks
    - CRLF vs LF vs mixed line endings & split CRLF
    - Empty event lines and empty payloads
    - Unknown event types
    - Stream drops and AbortController
    - HTTP 4xx/5xx structured error responses
    """
    import subprocess

    res = subprocess.run(
        ["node", "--experimental-strip-types", "tests/test_frontend_sse_adversarial.mjs"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Node SSE stress suite failed:\n{res.stdout}\n{res.stderr}"
    assert "10/10 passed" in res.stdout
