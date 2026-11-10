"""End-to-end tests for recent platform additions:
- Query difficulty classification endpoint (/api/query/classify)
- Adaptive alpha streaming integration (/api/query/stream)
- Chunking strategy comparison endpoint (/api/v1/documents/{id}/chunk-comparison)
- Incremental BM25 store chunk addition and document removal
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hybrid_rag.api.documents import router as documents_router
from hybrid_rag.api.query import router as query_router
from hybrid_rag.db.engine import get_db
from hybrid_rag.models.document import Document, DocumentChunk
from hybrid_rag.retrieval.bm25_store import Bm25Store
from hybrid_rag.schemas.generation import (
    CompositeConfidence,
    GenerationResponse,
    GroundedAnswer,
    VerificationResult,
)
from hybrid_rag.schemas.pipeline import PipelineConfig, QueryTrace, StageTrace


@pytest.fixture()
def app_client():
    test_app = FastAPI()
    test_app.include_router(query_router)
    test_app.include_router(documents_router)
    return TestClient(test_app)


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = "message"
        data = {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        events.append((event_type, data))
    return events


def test_classify_query_endpoint(app_client):
    """Test POST /api/query/classify correctly categorizes queries."""
    res_exact = app_client.post(
        "/api/query/classify",
        json={"question": 'Where is "TimeoutException" defined in v1.2.0?'},
    )
    assert res_exact.status_code == 200
    data_exact = res_exact.json()
    assert data_exact["category"] == "exact_match"
    assert data_exact["recommended_alpha"] == 0.2
    assert data_exact["confidence"] > 0

    res_semantic = app_client.post(
        "/api/query/classify",
        json={"question": "Explain the architectural advantages and trade-offs of microservices"},
    )
    assert res_semantic.status_code == 200
    data_semantic = res_semantic.json()
    assert data_semantic["category"] == "semantic"
    assert data_semantic["recommended_alpha"] == 0.7


def test_stream_query_adaptive_alpha_enabled(app_client):
    """When adaptive_alpha=True, stream emits classification in started event and uses tuned alpha."""
    mock_trace = QueryTrace(
        query='Explain "ZeroDivisionError" in detail',
        config=PipelineConfig(alpha=0.2),
        stages=[StageTrace(stage="dense", candidates=[], duration_ms=1.0)],
    )

    mock_gen = GenerationResponse(
        query='Explain "ZeroDivisionError" in detail',
        answer=GroundedAnswer(answer="A division error occurs when dividing by zero.", citations=[]),
        verification=VerificationResult(citation_coverage=1.0, all_supported=True, claims=[]),
        confidence=CompositeConfidence(
            overall_score=0.9,
            retrieval_score=0.9,
            citation_coverage=1.0,
            completeness_score=0.9,
        ),
        trace_id="test-trace",
        model="gemini",
    )

    with (
        patch("hybrid_rag.retrieval.retriever.full_retrieve", return_value=mock_trace) as mock_retrieve,
        patch("hybrid_rag.generation.pipeline.generate_grounded_answer", return_value=mock_gen),
    ):
        resp = app_client.post(
            "/api/query/stream",
            json={
                "question": 'Find "ZeroDivisionError" code 0x1234',
                "adaptive_alpha": True,
                "alpha": 0.8,  # Should be overridden by classifier to 0.2
            },
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)

        started_events = [data for et, data in events if et == "stage" and data.get("name") == "started"]
        assert len(started_events) == 1
        classification = started_events[0].get("classification")
        assert classification is not None
        assert classification["category"] == "exact_match"
        assert classification["alpha"] == 0.2

        # Verify full_retrieve was invoked with alpha=0.2
        assert mock_retrieve.called
        config_passed = mock_retrieve.call_args[0][1]
        assert config_passed.alpha == 0.2


def test_stream_query_adaptive_alpha_disabled(app_client):
    """When adaptive_alpha=False, the requested alpha is respected and classification is omitted."""
    mock_trace = QueryTrace(
        query="What is python?",
        config=PipelineConfig(alpha=0.75),
        stages=[StageTrace(stage="dense", candidates=[], duration_ms=1.0)],
    )

    mock_gen = GenerationResponse(
        query="What is python?",
        answer=GroundedAnswer(answer="A programming language.", citations=[]),
        verification=VerificationResult(citation_coverage=1.0, all_supported=True, claims=[]),
        confidence=CompositeConfidence(
            overall_score=0.9,
            retrieval_score=0.9,
            citation_coverage=1.0,
            completeness_score=0.9,
        ),
        trace_id="test-trace",
        model="gemini",
    )

    with (
        patch("hybrid_rag.retrieval.retriever.full_retrieve", return_value=mock_trace) as mock_retrieve,
        patch("hybrid_rag.generation.pipeline.generate_grounded_answer", return_value=mock_gen),
    ):
        resp = app_client.post(
            "/api/query/stream",
            json={
                "question": "What is python?",
                "adaptive_alpha": False,
                "alpha": 0.75,
            },
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)

        started_events = [data for et, data in events if et == "stage" and data.get("name") == "started"]
        assert len(started_events) == 1
        assert "classification" not in started_events[0]

        assert mock_retrieve.called
        config_passed = mock_retrieve.call_args[0][1]
        assert config_passed.alpha == 0.75


def test_chunk_comparison_endpoint_not_found(app_client):
    """GET /chunk-comparison for nonexistent document returns 404."""
    mock_db = MagicMock()
    mock_db.get.return_value = None
    mock_db.query.return_value.filter.return_value.first.return_value = None

    test_app = app_client.app
    test_app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = app_client.get(f"/api/v1/documents/{uuid.uuid4()}/chunk-comparison")
        assert resp.status_code == 404
    finally:
        test_app.dependency_overrides.clear()


def test_chunk_comparison_endpoint_success(app_client):
    """GET /chunk-comparison returns multi-strategy breakdown with preview and token counts."""
    doc_id = uuid.uuid4()
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.filename = "architecture.md"
    mock_doc.chunking_strategy = "fixed"

    chunk1 = MagicMock(spec=DocumentChunk)
    chunk1.document_id = doc_id
    chunk1.chunk_index = 0
    chunk1.content = "# Overview\nThis document describes the hybrid RAG architecture in detail.\n"

    chunk2 = MagicMock(spec=DocumentChunk)
    chunk2.document_id = doc_id
    chunk2.chunk_index = 1
    chunk2.content = "## Retrieval\nWe combine dense vector embeddings with BM25 sparse lexical search.\n"

    mock_db = MagicMock()
    mock_db.get.return_value = mock_doc
    mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [chunk1, chunk2]

    test_app = app_client.app
    test_app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = app_client.get(f"/api/v1/documents/{doc_id}/chunk-comparison")
        assert resp.status_code == 200
        data = resp.json()

        assert data["document_id"] == str(doc_id)
        assert data["filename"] == "architecture.md"
        assert data["current_strategy"] == "fixed"
        assert "fixed" in data["strategies"]
        assert "structure" in data["strategies"]

        fixed_chunks = data["strategies"]["fixed"]
        assert len(fixed_chunks) >= 1
        assert "token_count" in fixed_chunks[0]
        assert "preview" in fixed_chunks[0]
        assert "char_start" in fixed_chunks[0]
        assert "char_end" in fixed_chunks[0]
    finally:
        test_app.dependency_overrides.clear()


def test_bm25_store_incremental_add_and_remove():
    """Verify Bm25Store.add_chunks and remove_by_doc_id update in-memory corpus and search results."""
    store = Bm25Store()
    strategy = "test_strat"

    # 1. Add chunks for Doc 1
    doc1_id = "doc-1"
    doc1_texts = [
        "Kubernetes container orchestration simplifies deployment.",
        "Pods and services communicate across a virtual network mesh.",
    ]
    doc1_cids = ["c1-1", "c1-2"]

    count1 = store.add_chunks(doc1_texts, doc1_cids, strategy, doc_ids=[doc1_id, doc1_id])
    assert count1 == 2
    assert store.has_index(strategy)

    # Search finds Doc 1
    res1 = store.search("Kubernetes orchestration", strategy)
    assert any(cid == "c1-1" for cid, _ in res1)

    # 2. Incrementally add chunks for Doc 2
    doc2_id = "doc-2"
    doc2_texts = [
        "PostgreSQL supports relational queries and JSONB documents.",
        "pgvector extends PostgreSQL with cosine distance vector indexing.",
    ]
    doc2_cids = ["c2-1", "c2-2"]

    count2 = store.add_chunks(doc2_texts, doc2_cids, strategy, doc_ids=[doc2_id, doc2_id])
    assert count2 == 4  # Total chunks now 4

    # Search finds both Doc 1 and Doc 2
    res_k8s = store.search("Kubernetes", strategy)
    res_pg = store.search("PostgreSQL", strategy)
    assert any(cid == "c1-1" for cid, _ in res_k8s)
    assert any(cid == "c2-1" for cid, _ in res_pg)

    # 3. Remove Doc 1 without affecting Doc 2
    rem_count = store.remove_by_doc_id(strategy, doc1_id)
    assert rem_count == 2  # 2 chunks remaining

    # Search for Doc 1 keywords now returns empty
    res_k8s_after = store.search("Kubernetes orchestration", strategy)
    assert not any(cid.startswith("c1-") for cid, _ in res_k8s_after)

    # Doc 2 is still found
    res_pg_after = store.search("pgvector cosine", strategy)
    assert any(cid == "c2-2" for cid, _ in res_pg_after)

    # 4. Remove Doc 2 -> store becomes empty and cleared
    rem_count_final = store.remove_by_doc_id(strategy, doc2_id)
    assert rem_count_final == 0
    assert not store.has_index(strategy)
