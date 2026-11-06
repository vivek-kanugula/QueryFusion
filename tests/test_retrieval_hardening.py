"""Retrieval and Candidate Passage Hydration Stress Test Suite.

Verifies failure modes, boundary conditions, and algorithmic guarantees
for RRF Fusion and Candidate Passage Hydration.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from hybrid_rag.api.query import router
from hybrid_rag.db.engine import get_db
from hybrid_rag.models.document import DocumentChunk
from hybrid_rag.retrieval.retriever import full_retrieve, hydrate_candidates, rrf_fuse
from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, QueryTrace, StageTrace


def _make_candidate(
    chunk_id: str,
    score: float = 1.0,
    rank: int = 1,
    content: str = "",
    metadata: dict | None = None,
) -> CandidateResult:
    return CandidateResult(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        content=content,
        metadata=dict(metadata or {}),
    )


# ==============================================================================
# SECTION 1: RRF Fusion Algorithmic Stress Vectors
# ==============================================================================


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("k", [1, 10, 60, 1000])
def test_rrf_empty_dense_strictly_no_phantom_dense_reciprocal(alpha: float, k: int):
    """When dense candidates are empty, sparse candidates must NEVER receive any dense score.

    Score must strictly equal (1 - alpha) / (k + s_rank).
    """
    sparse_candidates = [
        _make_candidate(f"sparse_{i}", score=10.0 - i, rank=i) for i in range(1, 11)
    ]
    stage = rrf_fuse([], sparse_candidates, alpha=alpha, k=k, fusion_top_k=15)

    assert len(stage.candidates) == len(sparse_candidates)
    for c in stage.candidates:
        s_rank = c.metadata["sparse_rank"]
        assert s_rank is not None
        assert c.metadata["dense_rank"] is None
        assert c.metadata["rank_delta"] is None

        expected_score = (1.0 - alpha) / (k + s_rank)
        phantom_score = alpha / (k + 1) + expected_score

        # Strict equality to expected single-arm score
        assert pytest.approx(c.score, rel=1e-9) == expected_score
        # Must never equal phantom score
        assert abs(c.score - phantom_score) > 1e-7


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("k", [1, 10, 60, 1000])
def test_rrf_empty_sparse_strictly_no_phantom_sparse_reciprocal(alpha: float, k: int):
    """When sparse candidates are empty, dense candidates must NEVER receive any sparse score.

    Score must strictly equal alpha / (k + d_rank).
    """
    dense_candidates = [
        _make_candidate(f"dense_{i}", score=1.0 - i * 0.05, rank=i) for i in range(1, 11)
    ]
    stage = rrf_fuse(dense_candidates, [], alpha=alpha, k=k, fusion_top_k=15)

    assert len(stage.candidates) == len(dense_candidates)
    for c in stage.candidates:
        d_rank = c.metadata["dense_rank"]
        assert d_rank is not None
        assert c.metadata["sparse_rank"] is None
        assert c.metadata["rank_delta"] is None

        expected_score = alpha / (k + d_rank)
        phantom_score = expected_score + (1.0 - alpha) / (k + 1)

        # Strict equality to expected single-arm score
        assert pytest.approx(c.score, rel=1e-9) == expected_score
        # Must never equal phantom score
        assert abs(c.score - phantom_score) > 1e-7


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("k", [1, 60, 1000])
def test_rrf_both_lists_empty(alpha: float, k: int):
    """When both candidate lists are empty, rrf_fuse must return empty candidates without error."""
    stage = rrf_fuse([], [], alpha=alpha, k=k, fusion_top_k=10)
    assert stage.stage == "rrf"
    assert stage.candidates == []
    assert stage.duration_ms >= 0


def test_rrf_extreme_parameters():
    """Verify edge values: alpha=0.0 (pure sparse), alpha=1.0 (pure dense), k=1, k=1000."""
    dense = [_make_candidate("d1", score=0.9, rank=1, content="dense text")]
    sparse = [_make_candidate("s1", score=8.5, rank=1, content="sparse text")]

    # Pure sparse: alpha = 0.0
    stage_sparse = rrf_fuse(dense, sparse, alpha=0.0, k=60, fusion_top_k=5)
    assert len(stage_sparse.candidates) == 1
    assert stage_sparse.candidates[0].chunk_id == "s1"
    assert stage_sparse.candidates[0].score == 8.5
    assert stage_sparse.candidates[0].content == "sparse text"
    assert stage_sparse.candidates[0].metadata["sparse_rank"] == 1
    assert stage_sparse.candidates[0].metadata["dense_rank"] is None

    # Pure dense: alpha = 1.0
    stage_dense = rrf_fuse(dense, sparse, alpha=1.0, k=60, fusion_top_k=5)
    assert len(stage_dense.candidates) == 1
    assert stage_dense.candidates[0].chunk_id == "d1"
    assert stage_dense.candidates[0].score == 0.9
    assert stage_dense.candidates[0].content == "dense text"
    assert stage_dense.candidates[0].metadata["dense_rank"] == 1
    assert stage_dense.candidates[0].metadata["sparse_rank"] is None

    # Minimum k = 1
    stage_k1 = rrf_fuse(dense, sparse, alpha=0.5, k=1, fusion_top_k=5)
    # Both have rank 1: score = 0.5 / (1 + 1) = 0.25
    assert pytest.approx(stage_k1.candidates[0].score, rel=1e-6) == 0.25
    assert pytest.approx(stage_k1.candidates[1].score, rel=1e-6) == 0.25

    # Large k = 1000
    stage_k1000 = rrf_fuse(dense, sparse, alpha=0.5, k=1000, fusion_top_k=5)
    assert pytest.approx(stage_k1000.candidates[0].score, rel=1e-6) == 0.5 / 1001

    # fusion_top_k = 0
    stage_top0 = rrf_fuse(dense, sparse, alpha=0.5, k=60, fusion_top_k=0)
    assert stage_top0.candidates == []


def test_rrf_deterministic_ordering_stress_tie_break():
    """Verify deterministic candidate order across multiple runs with identical scores."""
    # Build 10 candidates in dense and 10 in sparse with equal ranks -> identical scores
    dense = [_make_candidate(f"chunk_dense_{i}", score=0.5, rank=i) for i in range(1, 11)]
    sparse = [_make_candidate(f"chunk_sparse_{i}", score=5.0, rank=i) for i in range(1, 11)]

    # Collect ordering over 20 repeated executions
    first_order = None
    for _ in range(20):
        stage = rrf_fuse(dense, sparse, alpha=0.5, k=60, fusion_top_k=20)
        current_order = [c.chunk_id for c in stage.candidates]
        if first_order is None:
            first_order = current_order
        else:
            assert current_order == first_order, "Non-deterministic order detected across runs!"

    # Verify dense elements precede sparse elements when tied at the same rank
    for i in range(1, 11):
        d_idx = first_order.index(f"chunk_dense_{i}")
        s_idx = first_order.index(f"chunk_sparse_{i}")
        assert d_idx < s_idx, f"Dense chunk {i} should precede sparse chunk {i} on equal score"


def test_rrf_content_and_metadata_preservation_edge_cases():
    """When candidates overlap or vary in content availability, rrf_fuse must preserve content."""
    # Case A: sparse has no content, dense has content
    dense_a = [_make_candidate("c1", 0.9, 1, content="Real Dense Text", metadata={"doc_id": "d1"})]
    sparse_a = [_make_candidate("c1", 5.0, 1, content="", metadata={"sparse_extra": True})]
    stage_a = rrf_fuse(dense_a, sparse_a, alpha=0.5, k=60, fusion_top_k=5)
    assert stage_a.candidates[0].content == "Real Dense Text"

    # Case B: sparse has content, dense has no content
    dense_b = [_make_candidate("c2", 0.9, 1, content="", metadata={"dense_extra": True})]
    sparse_b = [
        _make_candidate("c2", 5.0, 1, content="Real Sparse Text", metadata={"doc_id": "d2"})
    ]
    stage_b = rrf_fuse(dense_b, sparse_b, alpha=0.5, k=60, fusion_top_k=5)
    assert stage_b.candidates[0].content == "Real Sparse Text"

    # Case C: disjoint candidates preserve respective metadata
    dense_c = [_make_candidate("c3", 0.9, 1, content="Dense Only", metadata={"author": "Alice"})]
    sparse_c = [_make_candidate("c4", 5.0, 1, content="Sparse Only", metadata={"author": "Bob"})]
    stage_c = rrf_fuse(dense_c, sparse_c, alpha=0.5, k=60, fusion_top_k=5)
    c3 = next(c for c in stage_c.candidates if c.chunk_id == "c3")
    c4 = next(c for c in stage_c.candidates if c.chunk_id == "c4")
    assert c3.metadata["author"] == "Alice"
    assert c4.metadata["author"] == "Bob"


# ==============================================================================
# SECTION 2: Candidate Passage Hydration Stress Vectors
# ==============================================================================


def test_full_retrieve_hydrates_content_and_document_id_when_rerank_disabled():
    """When rerank_enabled=False, all candidates must have genuine passage text and document_id."""
    chunk_uuid_1 = uuid.uuid4()
    chunk_uuid_2 = uuid.uuid4()
    doc_uuid_1 = uuid.uuid4()
    doc_uuid_2 = uuid.uuid4()

    mock_chunks = [
        DocumentChunk(
            id=chunk_uuid_1,
            document_id=doc_uuid_1,
            chunk_index=0,
            content="Genuine passage text for first chunk from database.",
        ),
        DocumentChunk(
            id=chunk_uuid_2,
            document_id=doc_uuid_2,
            chunk_index=1,
            content="Genuine passage text for second chunk from database.",
        ),
    ]

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = mock_chunks

    config = PipelineConfig(rerank_enabled=False)
    fake_trace = QueryTrace(query="test query", config=config)
    fake_trace.add_stage(
        StageTrace(
            stage="rrf",
            candidates=[
                CandidateResult(
                    chunk_id=str(chunk_uuid_1), score=0.03, rank=1, content="", metadata={}
                ),
                CandidateResult(
                    chunk_id=str(chunk_uuid_2), score=0.02, rank=2, content="", metadata={}
                ),
            ],
            duration_ms=5.0,
        )
    )

    with patch("hybrid_rag.retrieval.retriever.hybrid_retrieve", return_value=fake_trace):
        result_trace = full_retrieve("test query", config, db=mock_db)

    assert len(result_trace.stages) == 1
    assert result_trace.stages[-1].stage == "rrf"
    candidates = result_trace.stages[-1].candidates
    assert len(candidates) == 2

    assert candidates[0].content == "Genuine passage text for first chunk from database."
    assert candidates[0].metadata["document_id"] == str(doc_uuid_1)
    assert candidates[1].content == "Genuine passage text for second chunk from database."
    assert candidates[1].metadata["document_id"] == str(doc_uuid_2)


def test_full_retrieve_hydrates_before_cross_encoder_reranker():
    """When rerank_enabled=True, cross-encoder must receive hydrated passage text, NOT raw UUIDs."""
    chunk_uuid = uuid.uuid4()
    doc_uuid = uuid.uuid4()

    mock_chunk = DocumentChunk(
        id=chunk_uuid,
        document_id=doc_uuid,
        chunk_index=0,
        content="Important technical documentation text.",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_chunk]

    config = PipelineConfig(rerank_enabled=True)
    fake_trace = QueryTrace(query="test query", config=config)
    fake_trace.add_stage(
        StageTrace(
            stage="rrf",
            candidates=[
                CandidateResult(
                    chunk_id=str(chunk_uuid), score=0.016, rank=1, content="", metadata={}
                ),
            ],
            duration_ms=5.0,
        )
    )

    with (
        patch("hybrid_rag.retrieval.retriever.hybrid_retrieve", return_value=fake_trace),
        patch("hybrid_rag.retrieval.reranker.get_reranker") as mock_get_reranker,
    ):
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [2.5]
        mock_get_reranker.return_value = mock_reranker

        result_trace = full_retrieve("test query", config, db=mock_db)

        # Inspect the exact arguments passed to cross-encoder rerank
        assert mock_reranker.rerank.called
        call_query, call_documents = mock_reranker.rerank.call_args[0]
        assert call_query == "test query"
        assert len(call_documents) == 1
        # The document text must be the genuine passage, NEVER the raw chunk_uuid
        assert call_documents[0] == "Important technical documentation text."
        assert call_documents[0] != str(chunk_uuid)

    rerank_candidates = result_trace.stages[-1].candidates
    assert rerank_candidates[0].content == "Important technical documentation text."
    assert rerank_candidates[0].metadata["document_id"] == str(doc_uuid)


def test_stream_query_passes_hydrated_passages_to_generator():
    """Test POST /api/query/stream via TestClient to ensure generator receives hydrated text."""
    chunk_uuid = uuid.uuid4()
    doc_uuid = uuid.uuid4()

    mock_chunk = DocumentChunk(
        id=chunk_uuid,
        document_id=doc_uuid,
        chunk_index=0,
        content="Grounded passage evidence for LLM generation.",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_chunk]

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = lambda: mock_db
    client = TestClient(app)

    captured_generator_candidates = []

    def mock_generate_grounded_answer(query, trace, **kwargs):
        # Capture candidates passed into generator
        for c in trace.stages[-1].candidates:
            captured_generator_candidates.append(c)

        from hybrid_rag.schemas.generation import (
            Citation,
            CompositeConfidence,
            GenerationResponse,
            GroundedAnswer,
        )

        return GenerationResponse(
            query=query,
            answer=GroundedAnswer(
                answer="Answer with evidence [C1].",
                citations=[
                    Citation(
                        citation_id="[C1]", chunk_id=str(chunk_uuid), evidence_quote="evidence"
                    )
                ],
                has_sufficient_context=True,
                fallback_reason=None,
            ),
            verification=None,
            confidence=CompositeConfidence(overall_score=0.9),
            trace_id=trace.trace_id,
            model="gemini-test",
            cached=False,
        )

    # Fake retrieval returning empty content to see if full_retrieve hydrates it before generation
    config = PipelineConfig(rerank_enabled=False)
    fake_retrieval_trace = QueryTrace(query="What is testing?", config=config)
    fake_retrieval_trace.add_stage(
        StageTrace(
            stage="rrf",
            candidates=[
                CandidateResult(
                    chunk_id=str(chunk_uuid), score=0.016, rank=1, content="", metadata={}
                ),
            ],
            duration_ms=1.0,
        )
    )

    with (
        patch("hybrid_rag.retrieval.retriever.hybrid_retrieve", return_value=fake_retrieval_trace),
        patch(
            "hybrid_rag.generation.pipeline.generate_grounded_answer",
            side_effect=mock_generate_grounded_answer,
        ),
        patch("hybrid_rag.retrieval.bm25_store.get_bm25_store", return_value=MagicMock()),
        patch("hybrid_rag.api.query.trace_store"),
    ):
        resp = client.post(
            "/api/query/stream", json={"question": "What is testing?", "rerank_enabled": False}
        )

    assert resp.status_code == 200
    assert len(captured_generator_candidates) == 1
    # Verify the candidate delivered to the generator has actual content and document_id
    candidate = captured_generator_candidates[0]
    assert candidate.content == "Grounded passage evidence for LLM generation."
    assert candidate.metadata["document_id"] == str(doc_uuid)


def test_hydration_graceful_on_db_exception():
    """Verify that when DB raises operational error, hydration fails gracefully without crash."""
    chunk_uuid = uuid.uuid4()
    candidates = [
        CandidateResult(chunk_id=str(chunk_uuid), score=0.8, rank=1, content="", metadata={})
    ]

    mock_failing_db = MagicMock()
    mock_failing_db.query.side_effect = OperationalError("connection refused", {}, None)

    # hydrate_candidates must catch the exception and return gracefully without raising
    hydrate_candidates(candidates, mock_failing_db)

    # Candidate remains unhydrated, but process survives
    assert candidates[0].content == ""


def test_hydration_graceful_on_missing_chunks():
    """Verify that when a chunk does not exist in the DB, hydration proceeds for existing ones."""
    existing_uuid = uuid.uuid4()
    missing_uuid = uuid.uuid4()
    doc_uuid = uuid.uuid4()

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [
        DocumentChunk(
            id=existing_uuid,
            document_id=doc_uuid,
            chunk_index=0,
            content="Existing chunk content.",
        )
    ]

    candidates = [
        CandidateResult(chunk_id=str(existing_uuid), score=0.9, rank=1, content="", metadata={}),
        CandidateResult(chunk_id=str(missing_uuid), score=0.8, rank=2, content="", metadata={}),
    ]

    hydrate_candidates(candidates, mock_db)

    assert candidates[0].content == "Existing chunk content."
    assert candidates[0].metadata["document_id"] == str(doc_uuid)
    # Missing chunk remains empty without raising KeyError
    assert candidates[1].content == ""
    assert "document_id" not in candidates[1].metadata


def test_hydration_graceful_on_non_uuid_chunks():
    """Verify that synthetic string IDs (non-UUID) are safely skipped without ValueError."""
    candidates = [
        CandidateResult(chunk_id="synthetic-chunk-123", score=0.5, rank=1, content="", metadata={}),
        CandidateResult(
            chunk_id="not-a-uuid", score=0.4, rank=2, content="Existing text", metadata={}
        ),
    ]

    mock_db = MagicMock()
    hydrate_candidates(candidates, mock_db)

    # db.query should not even be called when no valid UUIDs exist
    assert not mock_db.query.called
    assert candidates[0].content == ""
    assert candidates[1].content == "Existing text"


def test_hydration_empty_candidate_list():
    """Verify that an empty candidate list is handled cleanly with zero queries."""
    mock_db = MagicMock()
    hydrate_candidates([], mock_db)
    assert not mock_db.query.called
