"""Tests for streaming query API: stage transitions, trace ID, error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from hybrid_rag.api.query import QueryRequest, _sse_event, router


@pytest.fixture()
def client():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


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


def test_sse_event_format():
    result = _sse_event("stage", {"name": "dense"})
    assert result.startswith("event: stage\n")
    assert result.endswith("\n\n")
    lines = result.strip().split("\n")
    assert lines[0] == "event: stage"
    data = json.loads(lines[1][len("data: ") :])
    assert data["name"] == "dense"


def test_query_request_defaults():
    req = QueryRequest(question="What is RAG?")
    assert req.retrieval_strategy == "hybrid_rerank"
    assert req.chunking_strategy == "fixed"
    assert req.alpha == 0.5
    assert req.rerank_enabled is True
    assert req.verify is True
    assert req.temperature == 0.0


def test_query_request_custom():
    req = QueryRequest(
        question="test",
        retrieval_strategy="dense",
        alpha=0.7,
        rerank_enabled=False,
        verify=False,
    )
    assert req.retrieval_strategy == "dense"
    assert req.alpha == 0.7
    assert req.rerank_enabled is False


def _make_mock_trace():
    from hybrid_rag.schemas.pipeline import (
        CandidateResult,
        PipelineConfig,
        QueryTrace,
        StageTrace,
    )

    config = PipelineConfig(
        retrieval_strategy="hybrid_rerank",
        chunking_strategy="fixed",
    )
    return QueryTrace(
        trace_id="test-trace-id",
        query="What is RAG?",
        config=config,
        stages=[
            StageTrace(
                stage="dense",
                candidates=[
                    CandidateResult(
                        chunk_id="c1",
                        score=0.95,
                        rank=1,
                        content="RAG is...",
                        metadata={},
                    ),
                ],
                duration_ms=50.0,
            ),
        ],
    )


def _make_mock_gen_response():
    from hybrid_rag.schemas.generation import (
        Citation,
        ClaimVerification,
        CompositeConfidence,
        GenerationResponse,
        GroundedAnswer,
        VerificationResult,
        VerificationVerdict,
    )

    return GenerationResponse(
        query="What is RAG?",
        answer=GroundedAnswer(
            answer="RAG combines retrieval with generation. [C1]",
            citations=[
                Citation(
                    citation_id="[C1]",
                    chunk_id="c1",
                    evidence_quote="RAG is...",
                ),
            ],
            has_sufficient_context=True,
            fallback_reason=None,
        ),
        verification=VerificationResult(
            citation_coverage=1.0,
            all_supported=True,
            claims=[
                ClaimVerification(
                    claim_text="RAG combines retrieval with generation",
                    citation_id="[C1]",
                    verdict=VerificationVerdict.SUPPORTED,
                    explanation="Matches source",
                ),
            ],
        ),
        confidence=CompositeConfidence(
            overall_score=0.88,
            retrieval_score=0.9,
            citation_coverage=1.0,
            completeness_score=0.8,
        ),
        trace_id="test-trace-id",
        model="gemini-3.5-flash-lite",
        cached=False,
    )


@patch("hybrid_rag.api.query.trace_store")
def test_stream_query_full_pipeline(mock_trace_store, client):
    mock_trace = _make_mock_trace()
    mock_gen = _make_mock_gen_response()

    with (
        patch(
            "hybrid_rag.retrieval.retriever.full_retrieve",
            return_value=mock_trace,
        ),
        patch(
            "hybrid_rag.generation.pipeline.generate_grounded_answer",
            return_value=mock_gen,
        ),
        patch(
            "hybrid_rag.retrieval.bm25_store.get_bm25_store",
            return_value=MagicMock(),
        ),
    ):
        resp = client.post(
            "/api/query/stream",
            json={"question": "What is RAG?"},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)

    event_types = [e[0] for e in events]
    assert event_types[0] == "stage"
    assert events[0][1]["name"] == "started"
    assert "trace_id" in events[0][1]
    assert "stage" in event_types
    assert "answer" in event_types
    assert "completed" in event_types


@patch("hybrid_rag.api.query.trace_store")
def test_stream_query_stage_order(mock_trace_store, client):
    mock_trace = _make_mock_trace()
    mock_gen = _make_mock_gen_response()

    with (
        patch(
            "hybrid_rag.retrieval.retriever.full_retrieve",
            return_value=mock_trace,
        ),
        patch(
            "hybrid_rag.generation.pipeline.generate_grounded_answer",
            return_value=mock_gen,
        ),
        patch(
            "hybrid_rag.retrieval.bm25_store.get_bm25_store",
            return_value=MagicMock(),
        ),
    ):
        resp = client.post(
            "/api/query/stream",
            json={"question": "What is RAG?"},
        )

    events = _parse_sse(resp.text)
    stage_names = [e[1].get("name") for e in events if e[0] == "stage"]

    assert stage_names[0] == "started"
    assert "dense" in stage_names
    assert "sparse" in stage_names
    assert "rrf" in stage_names
    assert "retrieval_complete" in stage_names
    assert "generation" in stage_names
    assert "verification" in stage_names

    started_idx = stage_names.index("started")
    complete_idx = stage_names.index("retrieval_complete")
    gen_idx = stage_names.index("generation")
    assert started_idx < complete_idx < gen_idx


@patch("hybrid_rag.api.query.trace_store")
def test_stream_query_trace_id_in_completed(mock_trace_store, client):
    mock_trace = _make_mock_trace()
    mock_gen = _make_mock_gen_response()

    with (
        patch(
            "hybrid_rag.retrieval.retriever.full_retrieve",
            return_value=mock_trace,
        ),
        patch(
            "hybrid_rag.generation.pipeline.generate_grounded_answer",
            return_value=mock_gen,
        ),
        patch(
            "hybrid_rag.retrieval.bm25_store.get_bm25_store",
            return_value=MagicMock(),
        ),
    ):
        resp = client.post(
            "/api/query/stream",
            json={"question": "What is RAG?"},
        )

    events = _parse_sse(resp.text)
    completed_events = [e for e in events if e[0] == "completed"]
    assert len(completed_events) == 1
    completed = completed_events[0][1]
    assert "trace_id" in completed
    assert "model" in completed
    assert "retrieval_ms" in completed
    assert "generation_ms" in completed


def test_stream_query_error_handling(client):
    with (
        patch(
            "hybrid_rag.retrieval.bm25_store.get_bm25_store",
            side_effect=RuntimeError("DB connection failed"),
        ),
    ):
        resp = client.post(
            "/api/query/stream",
            json={"question": "What is RAG?"},
        )

    events = _parse_sse(resp.text)
    event_types = [e[0] for e in events]
    assert "error" in event_types
    error_data = next(e[1] for e in events if e[0] == "error")
    assert "DB connection failed" in error_data["message"]


def test_stream_response_headers(client):
    with patch(
        "hybrid_rag.retrieval.bm25_store.get_bm25_store",
        side_effect=RuntimeError("fail"),
    ):
        resp = client.post(
            "/api/query/stream",
            json={"question": "test"},
        )

    assert resp.headers.get("cache-control") == "no-cache"
    assert resp.headers.get("x-accel-buffering") == "no"


@patch("hybrid_rag.api.query.trace_store")
def test_stream_query_saves_trace(mock_trace_store, client):
    mock_trace = _make_mock_trace()
    mock_gen = _make_mock_gen_response()

    with (
        patch(
            "hybrid_rag.retrieval.retriever.full_retrieve",
            return_value=mock_trace,
        ),
        patch(
            "hybrid_rag.generation.pipeline.generate_grounded_answer",
            return_value=mock_gen,
        ),
        patch(
            "hybrid_rag.retrieval.bm25_store.get_bm25_store",
            return_value=MagicMock(),
        ),
    ):
        client.post(
            "/api/query/stream",
            json={"question": "What is RAG?"},
        )

    mock_trace_store.save.assert_called_once_with(mock_trace)


@patch("hybrid_rag.api.query.trace_store")
def test_stream_without_rerank(mock_trace_store, client):
    mock_trace = _make_mock_trace()
    mock_gen = _make_mock_gen_response()

    with (
        patch(
            "hybrid_rag.retrieval.retriever.full_retrieve",
            return_value=mock_trace,
        ),
        patch(
            "hybrid_rag.generation.pipeline.generate_grounded_answer",
            return_value=mock_gen,
        ),
        patch(
            "hybrid_rag.retrieval.bm25_store.get_bm25_store",
            return_value=MagicMock(),
        ),
    ):
        resp = client.post(
            "/api/query/stream",
            json={
                "question": "What is RAG?",
                "rerank_enabled": False,
            },
        )

    events = _parse_sse(resp.text)
    stage_names = [e[1].get("name") for e in events if e[0] == "stage"]
    assert "rerank" not in stage_names


@patch("hybrid_rag.api.query.trace_store")
def test_stream_query_passes_db_session_to_full_retrieve(mock_trace_store, client):
    from hybrid_rag.db.engine import get_db

    mock_trace = _make_mock_trace()
    mock_gen = _make_mock_gen_response()
    mock_db = MagicMock()

    client.app.dependency_overrides[get_db] = lambda: mock_db

    try:
        with (
            patch(
                "hybrid_rag.retrieval.retriever.full_retrieve",
                return_value=mock_trace,
            ) as mock_full_retrieve,
            patch(
                "hybrid_rag.generation.pipeline.generate_grounded_answer",
                return_value=mock_gen,
            ),
            patch(
                "hybrid_rag.retrieval.bm25_store.get_bm25_store",
                return_value=MagicMock(),
            ),
        ):
            resp = client.post(
                "/api/query/stream",
                json={"question": "What is RAG?"},
            )
            assert resp.status_code == 200
            assert mock_full_retrieve.called
            call_kwargs = mock_full_retrieve.call_args.kwargs
            assert call_kwargs.get("db") is mock_db
    finally:
        client.app.dependency_overrides.clear()
