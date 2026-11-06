"""Tests for cross-encoder reranker, sigmoid normalization, and Lost-in-the-Middle."""

import math
from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.retrieval.reranker import (
    lost_in_middle_reorder,
    rerank_candidates,
    sigmoid,
)
from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig

# ── Sigmoid normalization ──


def test_sigmoid_zero():
    assert sigmoid(0.0) == 0.5


def test_sigmoid_positive():
    assert sigmoid(1.0) == pytest.approx(1 / (1 + math.exp(-1)))


def test_sigmoid_negative():
    assert sigmoid(-1.0) == pytest.approx(1 / (1 + math.exp(1)))


def test_sigmoid_large_positive():
    assert sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)


def test_sigmoid_large_negative():
    assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)


def test_sigmoid_symmetry():
    assert sigmoid(2.0) + sigmoid(-2.0) == pytest.approx(1.0)


def test_sigmoid_monotonic():
    values = [-10, -5, -1, 0, 1, 5, 10]
    results = [sigmoid(x) for x in values]
    assert results == sorted(results)


def test_sigmoid_bounds():
    for x in [-100, -10, -1, 0, 1, 10, 100]:
        s = sigmoid(x)
        assert 0.0 <= s <= 1.0


# ── Lost-in-the-Middle reordering ──


def _make(chunk_id: str, rank: int, score: float = 1.0) -> CandidateResult:
    return CandidateResult(chunk_id=chunk_id, score=score, rank=rank)


def test_litm_5_items():
    """[R1, R2, R3, R4, R5] -> [R1, R3, R5, R4, R2]"""
    candidates = [_make(f"c{i}", i) for i in range(1, 6)]
    result = lost_in_middle_reorder(candidates)
    assert [c.chunk_id for c in result] == ["c1", "c3", "c5", "c4", "c2"]


def test_litm_4_items():
    """[R1, R2, R3, R4] -> [R1, R3, R4, R2]"""
    candidates = [_make(f"c{i}", i) for i in range(1, 5)]
    result = lost_in_middle_reorder(candidates)
    assert [c.chunk_id for c in result] == ["c1", "c3", "c4", "c2"]


def test_litm_3_items():
    candidates = [_make(f"c{i}", i) for i in range(1, 4)]
    result = lost_in_middle_reorder(candidates)
    assert [c.chunk_id for c in result] == ["c1", "c3", "c2"]


def test_litm_2_items():
    candidates = [_make("a", 1), _make("b", 2)]
    result = lost_in_middle_reorder(candidates)
    assert [c.chunk_id for c in result] == ["a", "b"]


def test_litm_1_item():
    result = lost_in_middle_reorder([_make("a", 1)])
    assert [c.chunk_id for c in result] == ["a"]


def test_litm_empty():
    assert lost_in_middle_reorder([]) == []


def test_litm_7_items():
    """[1,2,3,4,5,6,7] -> [1,3,5,7,6,4,2]"""
    candidates = [_make(f"c{i}", i) for i in range(1, 8)]
    result = lost_in_middle_reorder(candidates)
    assert [c.chunk_id for c in result] == ["c1", "c3", "c5", "c7", "c6", "c4", "c2"]


def test_litm_6_items():
    """[1,2,3,4,5,6] -> [1,3,5,6,4,2]"""
    candidates = [_make(f"c{i}", i) for i in range(1, 7)]
    result = lost_in_middle_reorder(candidates)
    assert [c.chunk_id for c in result] == ["c1", "c3", "c5", "c6", "c4", "c2"]


def test_litm_reassigns_ranks():
    candidates = [_make(f"c{i}", i) for i in range(1, 6)]
    result = lost_in_middle_reorder(candidates)
    assert [c.rank for c in result] == [1, 2, 3, 4, 5]


def test_litm_preserves_metadata():
    candidates = [
        CandidateResult(chunk_id="a", score=0.9, rank=1, metadata={"key": "val"}),
        CandidateResult(chunk_id="b", score=0.8, rank=2, metadata={"key": "other"}),
        CandidateResult(chunk_id="c", score=0.7, rank=3, metadata={"key": "third"}),
    ]
    result = lost_in_middle_reorder(candidates)
    meta_map = {c.chunk_id: c.metadata["key"] for c in result}
    assert meta_map == {"a": "val", "c": "third", "b": "other"}


def test_litm_preserves_content():
    candidates = [
        CandidateResult(chunk_id="a", score=0.9, rank=1, content="alpha"),
        CandidateResult(chunk_id="b", score=0.8, rank=2, content="beta"),
        CandidateResult(chunk_id="c", score=0.7, rank=3, content="gamma"),
    ]
    result = lost_in_middle_reorder(candidates)
    content_map = {c.chunk_id: c.content for c in result}
    assert content_map == {"a": "alpha", "c": "gamma", "b": "beta"}


def test_litm_does_not_mutate_input():
    candidates = [_make(f"c{i}", i) for i in range(1, 4)]
    original_ids = [c.chunk_id for c in candidates]
    lost_in_middle_reorder(candidates)
    assert [c.chunk_id for c in candidates] == original_ids


# ── Reranker with mock cross-encoder ──


def test_rerank_empty_candidates():
    stage = rerank_candidates("query", [], top_k=5)
    assert stage.stage == "rerank"
    assert len(stage.candidates) == 0
    assert stage.duration_ms >= 0


@patch("hybrid_rag.retrieval.reranker.get_reranker")
def test_rerank_scores_sigmoid_normalized(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_get_reranker.return_value = mock_reranker
    mock_reranker.rerank.return_value = [2.0, -1.0, 1.0]

    candidates = [
        CandidateResult(chunk_id="a", score=0.5, rank=1, content="doc A"),
        CandidateResult(chunk_id="b", score=0.4, rank=2, content="doc B"),
        CandidateResult(chunk_id="c", score=0.3, rank=3, content="doc C"),
    ]

    stage = rerank_candidates("test query", candidates, top_k=3, apply_lost_in_middle=False)

    assert stage.candidates[0].chunk_id == "a"
    assert stage.candidates[1].chunk_id == "c"
    assert stage.candidates[2].chunk_id == "b"
    assert stage.candidates[0].score == pytest.approx(sigmoid(2.0))
    assert stage.candidates[1].score == pytest.approx(sigmoid(1.0))
    assert stage.candidates[2].score == pytest.approx(sigmoid(-1.0))


@patch("hybrid_rag.retrieval.reranker.get_reranker")
def test_rerank_records_pre_rerank_rank(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_get_reranker.return_value = mock_reranker
    mock_reranker.rerank.return_value = [1.0, 0.5, 3.0]

    candidates = [
        CandidateResult(chunk_id="a", score=0.5, rank=1, content="doc A"),
        CandidateResult(chunk_id="b", score=0.4, rank=2, content="doc B"),
        CandidateResult(chunk_id="c", score=0.3, rank=3, content="doc C"),
    ]

    stage = rerank_candidates("q", candidates, top_k=3, apply_lost_in_middle=False)

    assert stage.candidates[0].metadata["pre_rerank_rank"] == 3
    assert stage.candidates[0].metadata["raw_rerank_score"] == 3.0


@patch("hybrid_rag.retrieval.reranker.get_reranker")
def test_rerank_top_k_limits(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_get_reranker.return_value = mock_reranker
    mock_reranker.rerank.return_value = [3.0, 2.0, 1.0]

    candidates = [
        CandidateResult(chunk_id=f"c{i}", score=0.0, rank=i, content=f"doc {i}") for i in range(3)
    ]

    stage = rerank_candidates("q", candidates, top_k=2, apply_lost_in_middle=False)
    assert len(stage.candidates) == 2


@patch("hybrid_rag.retrieval.reranker.get_reranker")
def test_rerank_with_litm_applied(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_get_reranker.return_value = mock_reranker
    mock_reranker.rerank.return_value = [5.0, 4.0, 3.0, 2.0, 1.0]

    candidates = [
        CandidateResult(chunk_id=f"c{i}", score=0.0, rank=i + 1, content=f"doc {i}")
        for i in range(5)
    ]

    stage = rerank_candidates("q", candidates, top_k=5, apply_lost_in_middle=True)
    assert [c.chunk_id for c in stage.candidates] == ["c0", "c2", "c4", "c3", "c1"]
    assert stage.metadata["lost_in_middle"] is True


@patch("hybrid_rag.retrieval.reranker.get_reranker")
def test_rerank_without_litm(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_get_reranker.return_value = mock_reranker
    mock_reranker.rerank.return_value = [3.0, 2.0, 1.0]

    candidates = [
        CandidateResult(chunk_id=f"c{i}", score=0.0, rank=i + 1, content=f"doc {i}")
        for i in range(3)
    ]

    stage = rerank_candidates("q", candidates, top_k=3, apply_lost_in_middle=False)
    assert [c.chunk_id for c in stage.candidates] == ["c0", "c1", "c2"]
    assert stage.metadata["lost_in_middle"] is False


@patch("hybrid_rag.retrieval.reranker.get_reranker")
def test_rerank_stage_metadata(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_get_reranker.return_value = mock_reranker
    mock_reranker.rerank.return_value = [1.0]

    candidates = [CandidateResult(chunk_id="c0", score=0.5, rank=1, content="text")]

    stage = rerank_candidates("q", candidates, top_k=1, apply_lost_in_middle=False)

    assert stage.stage == "rerank"
    assert stage.duration_ms > 0
    assert stage.metadata["rerank_latency_ms"] > 0
    assert stage.metadata["candidates_in"] == 1
    assert stage.metadata["candidates_out"] == 1


@patch("hybrid_rag.retrieval.reranker.get_reranker")
def test_rerank_uses_content_not_chunk_id(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_get_reranker.return_value = mock_reranker
    mock_reranker.rerank.return_value = [1.0]

    candidates = [CandidateResult(chunk_id="abc-123", score=0.5, rank=1, content="actual doc text")]

    rerank_candidates("q", candidates, top_k=1, apply_lost_in_middle=False)

    call_args = mock_reranker.rerank.call_args
    documents = call_args[0][1]
    assert documents == ["actual doc text"]


@patch("hybrid_rag.retrieval.reranker.get_reranker")
def test_rerank_falls_back_to_chunk_id_without_content(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_get_reranker.return_value = mock_reranker
    mock_reranker.rerank.return_value = [1.0]

    candidates = [CandidateResult(chunk_id="abc-123", score=0.5, rank=1)]

    rerank_candidates("q", candidates, top_k=1, apply_lost_in_middle=False)

    call_args = mock_reranker.rerank.call_args
    documents = call_args[0][1]
    assert documents == ["abc-123"]


# ── Integration: full_retrieve end-to-end ──


@patch("hybrid_rag.retrieval.reranker.get_reranker")
@patch("hybrid_rag.ingest.embedder.embed_query")
def test_full_retrieve_with_reranking(mock_embed, mock_get_reranker):
    import numpy as np

    from hybrid_rag.retrieval.bm25_store import Bm25Store
    from hybrid_rag.retrieval.retriever import full_retrieve

    mock_embed.return_value = np.zeros(384)

    mock_db = MagicMock()
    c1 = MagicMock()
    c1.id = "c1"
    c1.document_id = "d1"
    c1.content = "machine learning content"
    c2 = MagicMock()
    c2.id = "c2"
    c2.document_id = "d1"
    c2.content = "database systems content"
    chain = mock_db.query.return_value.join.return_value
    chain.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
        (c1, 0.1),
        (c2, 0.3),
    ]
    mock_db.query.return_value.filter.return_value.all.return_value = [c1, c2]

    store = Bm25Store()
    store.rebuild(["machine learning", "database systems"], ["c1", "c3"], "fixed")

    mock_reranker = MagicMock()
    mock_get_reranker.return_value = mock_reranker

    def mock_rerank_fn(query, documents, top_k=None):
        n = len(documents)
        return [float(n - i) for i in range(n)]

    mock_reranker.rerank.side_effect = mock_rerank_fn

    config = PipelineConfig(
        chunking_strategy="fixed",
        dense_top_k=3,
        sparse_top_k=3,
        fusion_top_k=5,
        rerank_top_k=3,
        rerank_enabled=True,
        lost_in_middle=True,
    )

    trace = full_retrieve("machine learning", config, store, db=mock_db)

    assert len(trace.stages) == 4
    assert trace.stages[0].stage == "dense"
    assert trace.stages[1].stage == "sparse"
    assert trace.stages[2].stage == "rrf"
    assert trace.stages[3].stage == "rerank"
    assert trace.stages[3].metadata["lost_in_middle"] is True
    assert trace.total_duration_ms > 0


@patch("hybrid_rag.ingest.embedder.embed_query")
def test_full_retrieve_rerank_disabled(mock_embed):
    import numpy as np

    from hybrid_rag.retrieval.bm25_store import Bm25Store
    from hybrid_rag.retrieval.retriever import full_retrieve

    mock_embed.return_value = np.zeros(384)

    mock_db = MagicMock()
    chain = mock_db.query.return_value.join.return_value
    chain.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

    store = Bm25Store()

    config = PipelineConfig(
        chunking_strategy="fixed",
        rerank_enabled=False,
    )

    trace = full_retrieve("query", config, store, db=mock_db)

    assert len(trace.stages) == 3
    assert trace.stages[-1].stage == "rrf"


# ── Real cross-encoder (slow) ──


@pytest.mark.slow
def test_reranker_real_text_pairs():
    """Real cross-encoder scores relevant passage higher."""
    candidates = [
        CandidateResult(
            chunk_id="relevant",
            score=0.5,
            rank=1,
            content="Python is a popular programming language used for web development",
        ),
        CandidateResult(
            chunk_id="irrelevant",
            score=0.5,
            rank=2,
            content="The weather in Antarctica is extremely cold year round",
        ),
    ]

    stage = rerank_candidates(
        "What programming language is used for web development?",
        candidates,
        top_k=2,
        apply_lost_in_middle=False,
    )

    assert stage.candidates[0].chunk_id == "relevant"
    assert stage.candidates[0].score > stage.candidates[1].score
    assert 0.0 < stage.candidates[0].score < 1.0
    assert 0.0 < stage.candidates[1].score < 1.0
