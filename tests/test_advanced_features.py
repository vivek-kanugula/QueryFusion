"""Tests for chunk context inspector, HyDE query expansion, and Corrective RAG (CRAG)."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hybrid_rag.api.documents import router as documents_router
from hybrid_rag.api.query import router as query_router
from hybrid_rag.generation.crag import (
    execute_corrective_fallback,
    should_trigger_crag,
)
from hybrid_rag.retrieval.expansion import expand_query
from hybrid_rag.retrieval.retriever import multi_query_retrieve
from hybrid_rag.schemas.generation import (
    Citation,
    ClaimVerification,
    CompositeConfidence,
    GenerationResponse,
    GroundedAnswer,
    VerificationResult,
    VerificationVerdict,
)
from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, QueryTrace, StageTrace

# ── Chunk Context Endpoint Tests ──


def test_get_chunk_context_not_found():
    app = FastAPI()
    app.include_router(documents_router)
    client = TestClient(app)

    mock_db = MagicMock()
    mock_db.get.return_value = None
    mock_db.query.return_value.filter.return_value.first.return_value = None

    from hybrid_rag.db.engine import get_db

    client.app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.get(f"/api/v1/documents/chunks/{uuid.uuid4()}/context")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
    finally:
        client.app.dependency_overrides.clear()


def test_get_chunk_context_success():
    app = FastAPI()
    app.include_router(documents_router)
    client = TestClient(app)

    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    mock_doc = MagicMock()
    mock_doc.id = doc_id
    mock_doc.filename = "architecture.pdf"
    mock_doc.file_type = "pdf"
    mock_doc.chunking_strategy = "fixed"
    mock_doc.size_bytes = 10240

    mock_target_chunk = MagicMock()
    mock_target_chunk.id = chunk_id
    mock_target_chunk.document_id = doc_id
    mock_target_chunk.chunk_index = 1
    mock_target_chunk.token_count = 120
    mock_target_chunk.content = "Target chunk content."
    mock_target_chunk.section_heading = "Overview"

    mock_prev_chunk = MagicMock()
    mock_prev_chunk.id = uuid.uuid4()
    mock_prev_chunk.chunk_index = 0
    mock_prev_chunk.token_count = 110
    mock_prev_chunk.content = "Previous chunk content."
    mock_prev_chunk.section_heading = "Intro"

    mock_db = MagicMock()

    def mock_get(model, obj_id):
        if str(obj_id) == str(chunk_id):
            return mock_target_chunk
        if str(obj_id) == str(doc_id):
            return mock_doc
        return None

    mock_db.get.side_effect = mock_get

    query_mock = MagicMock()
    filter_mock = MagicMock()
    filter_mock.count.return_value = 3

    # Consecutive first() calls for prev_chunk and next_chunk
    filter_mock.first.side_effect = [mock_prev_chunk, None]
    query_mock.filter.return_value = filter_mock
    mock_db.query.return_value = query_mock

    from hybrid_rag.db.engine import get_db

    client.app.dependency_overrides[get_db] = lambda: mock_db
    try:
        resp = client.get(f"/api/v1/documents/chunks/{chunk_id}/context")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document"]["filename"] == "architecture.pdf"
        assert data["target_chunk"]["content"] == "Target chunk content."
        assert data["previous_chunk"]["content"] == "Previous chunk content."
        assert data["next_chunk"] is None
    finally:
        client.app.dependency_overrides.clear()


# ── Query Expansion Tests ──


@patch("hybrid_rag.retrieval.expansion.call_llm")
def test_expand_query_success(mock_call):
    mock_call.return_value = json.dumps(
        {
            "queries": ["sparse BM25 scoring algorithm", "BM25 inverted index token matching"],
            "hypothetical_passage": (
                "BM25 scores documents based on term frequency and document length."
            ),
        }
    )
    result = expand_query("How does BM25 work?")
    assert len(result) >= 3
    assert result[0] == "How does BM25 work?"
    assert "sparse BM25 scoring algorithm" in result
    assert "BM25 inverted index token matching" in result


@patch("hybrid_rag.retrieval.expansion.call_llm")
def test_expand_query_fallback_on_error(mock_call):
    mock_call.side_effect = RuntimeError("API failed")
    result = expand_query("What is pgvector?")
    assert result == ["What is pgvector?"]


# ── Multi-Query Retrieval Fusion Tests ──


def test_multi_query_retrieve_single_query():
    config = PipelineConfig(retrieval_strategy="hybrid", chunking_strategy="fixed")
    mock_trace = QueryTrace(query="q1", config=config)

    with patch(
        "hybrid_rag.retrieval.retriever.full_retrieve", return_value=mock_trace
    ) as mock_full:
        trace = multi_query_retrieve(["q1"], config)
        assert trace == mock_trace
        assert mock_full.called


def test_multi_query_retrieve_multiple_queries_fuses_rrf():
    config = PipelineConfig(retrieval_strategy="hybrid", chunking_strategy="fixed", fusion_top_k=5)
    c1 = CandidateResult(chunk_id="c1", score=0.9, rank=1, content="Doc 1", metadata={})
    c2 = CandidateResult(chunk_id="c2", score=0.8, rank=2, content="Doc 2", metadata={})
    c3 = CandidateResult(chunk_id="c3", score=0.7, rank=1, content="Doc 3", metadata={})

    trace1 = QueryTrace(
        query="q1", config=config, stages=[StageTrace(stage="rrf", candidates=[c1, c2])]
    )
    trace2 = QueryTrace(
        query="q2", config=config, stages=[StageTrace(stage="rrf", candidates=[c3, c1])]
    )

    with (
        patch(
            "hybrid_rag.retrieval.retriever.hybrid_retrieve",
            side_effect=[trace1, trace2],
        ),
        patch("hybrid_rag.retrieval.retriever.hydrate_candidates"),
    ):
        merged = multi_query_retrieve(["q1", "q2"], config)
        assert len(merged.stages) >= 1
        top_candidates = merged.stages[0].candidates
        # c1 appeared in both traces, so its RRF score is highest
        assert top_candidates[0].chunk_id == "c1"
        assert top_candidates[0].metadata.get("multi_query_fused") is True


# ── Corrective RAG (CRAG) Tests ──


def test_should_trigger_crag_insufficient_context():
    resp = GenerationResponse(
        query="q",
        answer=GroundedAnswer(answer="fallback", has_sufficient_context=False),
        confidence=CompositeConfidence(
            overall_score=0.9, retrieval_score=0.9, citation_coverage=1.0, completeness_score=0.9
        ),
    )
    trigger, reason = should_trigger_crag(resp)
    assert trigger is True
    assert "Insufficient context" in reason


def test_should_trigger_crag_unsupported_claim():
    resp = GenerationResponse(
        query="q",
        answer=GroundedAnswer(
            answer="claimed fact [C1]",
            citations=[Citation(citation_id="[C1]", chunk_id="c1", evidence_quote="q")],
            has_sufficient_context=True,
        ),
        verification=VerificationResult(
            citation_coverage=0.0,
            all_supported=False,
            claims=[
                ClaimVerification(
                    claim_text="claimed fact",
                    citation_id="[C1]",
                    verdict=VerificationVerdict.UNSUPPORTED,
                )
            ],
        ),
        confidence=CompositeConfidence(
            overall_score=0.8, retrieval_score=0.8, citation_coverage=0.0, completeness_score=0.8
        ),
    )
    trigger, reason = should_trigger_crag(resp)
    assert trigger is True
    assert "ungrounded claim" in reason


def test_should_trigger_crag_low_confidence():
    resp = GenerationResponse(
        query="q",
        answer=GroundedAnswer(answer="low confidence answer", has_sufficient_context=True),
        confidence=CompositeConfidence(
            overall_score=0.35, retrieval_score=0.4, citation_coverage=0.5, completeness_score=0.3
        ),
    )
    trigger, reason = should_trigger_crag(resp, min_confidence=0.5)
    assert trigger is True
    assert "below threshold" in reason


def test_should_trigger_crag_high_confidence_no_trigger():
    resp = GenerationResponse(
        query="q",
        answer=GroundedAnswer(answer="high confidence answer", has_sufficient_context=True),
        verification=VerificationResult(
            citation_coverage=1.0,
            all_supported=True,
            claims=[
                ClaimVerification(
                    claim_text="fact",
                    citation_id="[C1]",
                    verdict=VerificationVerdict.SUPPORTED,
                )
            ],
        ),
        confidence=CompositeConfidence(
            overall_score=0.88, retrieval_score=0.9, citation_coverage=1.0, completeness_score=0.8
        ),
    )
    trigger, reason = should_trigger_crag(resp, min_confidence=0.5)
    assert trigger is False
    assert reason == ""


def test_execute_corrective_fallback():
    config = PipelineConfig(retrieval_strategy="dense", chunking_strategy="fixed")
    mock_orig_trace = QueryTrace(query="q", config=config)
    mock_fallback_trace = QueryTrace(query="q", config=config)

    initial_resp = GenerationResponse(
        query="q",
        answer=GroundedAnswer(answer="bad", has_sufficient_context=False),
    )
    corrected_gen = GenerationResponse(
        query="q",
        answer=GroundedAnswer(answer="good corrected", has_sufficient_context=True),
        confidence=CompositeConfidence(
            overall_score=0.85, retrieval_score=0.9, citation_coverage=1.0, completeness_score=0.8
        ),
    )

    with (
        patch(
            "hybrid_rag.retrieval.retriever.full_retrieve", return_value=mock_fallback_trace
        ) as mock_ret,
        patch(
            "hybrid_rag.generation.pipeline.generate_grounded_answer", return_value=corrected_gen
        ) as mock_gen,
    ):
        result = execute_corrective_fallback(
            query="q",
            original_trace=mock_orig_trace,
            initial_response=initial_resp,
            reason="Low confidence",
            config=config,
        )

        assert mock_ret.called
        called_config = mock_ret.call_args[0][1]
        assert called_config.dense_top_k > config.dense_top_k
        assert called_config.sparse_top_k > config.sparse_top_k
        assert called_config.fusion_top_k >= config.fusion_top_k
        assert called_config.rerank_top_k >= config.rerank_top_k
        assert mock_gen.called
        assert result.was_corrected is True
        assert result.correction_reason == "Low confidence"
        assert result.answer.answer == "good corrected"


# ── Streaming API Query Expansion & CRAG Integration Tests ──


@patch("hybrid_rag.api.query.trace_store")
def test_stream_query_with_query_expansion_and_crag(mock_trace_store):
    app = FastAPI()
    app.include_router(query_router)
    client = TestClient(app)

    config = PipelineConfig(retrieval_strategy="hybrid", chunking_strategy="fixed")
    mock_trace = QueryTrace(
        trace_id="exp-trace",
        query="What is BM25?",
        config=config,
        stages=[StageTrace(stage="dense", candidates=[], duration_ms=10.0)],
    )

    initial_gen = GenerationResponse(
        query="What is BM25?",
        answer=GroundedAnswer(answer="Insufficient context", has_sufficient_context=False),
        confidence=CompositeConfidence(
            overall_score=0.3, retrieval_score=0.4, citation_coverage=0.0, completeness_score=0.3
        ),
        trace_id="exp-trace",
    )
    corrected_gen = GenerationResponse(
        query="What is BM25?",
        answer=GroundedAnswer(
            answer="BM25 is a ranking algorithm [C1].",
            citations=[Citation(citation_id="[C1]", chunk_id="c1", evidence_quote="quote")],
            has_sufficient_context=True,
        ),
        confidence=CompositeConfidence(
            overall_score=0.88, retrieval_score=0.9, citation_coverage=1.0, completeness_score=0.8
        ),
        trace_id="exp-trace",
        was_corrected=True,
        correction_reason="Insufficient context reported in initial retrieval",
    )

    with (
        patch(
            "hybrid_rag.retrieval.expansion.expand_query",
            return_value=["What is BM25?", "BM25 scoring formula"],
        ) as mock_expand,
        patch(
            "hybrid_rag.retrieval.retriever.multi_query_retrieve",
            return_value=mock_trace,
        ) as mock_mq,
        patch(
            "hybrid_rag.generation.pipeline.generate_grounded_answer",
            return_value=initial_gen,
        ),
        patch(
            "hybrid_rag.generation.crag.execute_corrective_fallback",
            return_value=corrected_gen,
        ) as mock_crag,
        patch("hybrid_rag.retrieval.bm25_store.get_bm25_store"),
    ):
        resp = client.post(
            "/api/query/stream",
            json={
                "question": "What is BM25?",
                "query_expansion": True,
                "enable_crag": True,
            },
        )

    assert resp.status_code == 200
    assert mock_expand.called
    assert mock_mq.called
    assert mock_crag.called

    events = []
    for block in resp.text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        evt_type = "message"
        data = {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                evt_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        events.append((evt_type, data))

    stage_names = [d.get("name") for t, d in events if t == "stage"]
    assert "expansion" in stage_names
    assert "corrective_fallback" in stage_names

    completed_events = [d for t, d in events if t == "completed"]
    assert len(completed_events) == 1
    assert completed_events[0]["was_corrected"] is True
    assert "Insufficient context" in completed_events[0]["correction_reason"]
    assert len(completed_events[0]["expanded_queries"]) == 2
