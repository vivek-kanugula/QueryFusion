import pytest

from hybrid_rag.retrieval.retriever import rrf_fuse
from hybrid_rag.schemas.pipeline import CandidateResult


def _make(chunk_id: str, score: float, rank: int) -> CandidateResult:
    return CandidateResult(chunk_id=chunk_id, score=score, rank=rank)


def test_pure_dense_alpha_1():
    dense = [_make("a", 0.9, 1), _make("b", 0.8, 2)]
    sparse = [_make("c", 5.0, 1), _make("d", 4.0, 2)]
    stage = rrf_fuse(dense, sparse, alpha=1.0, k=60, fusion_top_k=5)
    assert len(stage.candidates) == 2
    assert stage.candidates[0].chunk_id == "a"
    assert stage.candidates[1].chunk_id == "b"
    assert stage.candidates[0].metadata["sparse_rank"] is None


def test_pure_sparse_alpha_0():
    dense = [_make("a", 0.9, 1), _make("b", 0.8, 2)]
    sparse = [_make("c", 5.0, 1), _make("d", 4.0, 2)]
    stage = rrf_fuse(dense, sparse, alpha=0.0, k=60, fusion_top_k=5)
    assert len(stage.candidates) == 2
    assert stage.candidates[0].chunk_id == "c"
    assert stage.candidates[1].chunk_id == "d"
    assert stage.candidates[0].metadata["dense_rank"] is None


def test_equal_weight_both_lists():
    dense = [_make("a", 0.9, 1), _make("b", 0.7, 2)]
    sparse = [_make("a", 5.0, 1), _make("c", 3.0, 2)]
    stage = rrf_fuse(dense, sparse, alpha=0.5, k=60, fusion_top_k=10)
    ids = [c.chunk_id for c in stage.candidates]
    assert "a" in ids
    assert "b" in ids
    assert "c" in ids
    assert stage.candidates[0].chunk_id == "a"


def test_disjoint_candidate_sets():
    dense = [_make("a", 0.9, 1), _make("b", 0.8, 2)]
    sparse = [_make("c", 5.0, 1), _make("d", 4.0, 2)]
    stage = rrf_fuse(dense, sparse, alpha=0.5, k=60, fusion_top_k=10)
    assert len(stage.candidates) == 4
    for c in stage.candidates:
        if c.chunk_id in ("a", "b"):
            assert c.metadata["sparse_rank"] is None
        else:
            assert c.metadata["dense_rank"] is None


def test_identical_candidate_sets():
    dense = [_make("a", 0.9, 1), _make("b", 0.7, 2)]
    sparse = [_make("a", 5.0, 1), _make("b", 3.0, 2)]
    stage = rrf_fuse(dense, sparse, alpha=0.5, k=60, fusion_top_k=10)
    assert len(stage.candidates) == 2
    for c in stage.candidates:
        assert c.metadata["dense_rank"] is not None
        assert c.metadata["sparse_rank"] is not None
        assert c.metadata["rank_delta"] is not None


def test_single_candidate_each():
    dense = [_make("a", 0.9, 1)]
    sparse = [_make("b", 5.0, 1)]
    stage = rrf_fuse(dense, sparse, alpha=0.5, k=60, fusion_top_k=5)
    assert len(stage.candidates) == 2


def test_empty_dense():
    sparse = [_make("c", 5.0, 1)]
    stage = rrf_fuse([], sparse, alpha=0.5, k=60, fusion_top_k=5)
    assert len(stage.candidates) == 1
    assert stage.candidates[0].chunk_id == "c"


def test_empty_sparse():
    dense = [_make("a", 0.9, 1)]
    stage = rrf_fuse(dense, [], alpha=0.5, k=60, fusion_top_k=5)
    assert len(stage.candidates) == 1
    assert stage.candidates[0].chunk_id == "a"


def test_both_empty():
    stage = rrf_fuse([], [], alpha=0.5, k=60, fusion_top_k=5)
    assert len(stage.candidates) == 0


def test_scores_descending():
    dense = [_make("a", 0.9, 1), _make("b", 0.8, 2), _make("c", 0.7, 3)]
    sparse = [_make("d", 5.0, 1), _make("e", 4.0, 2), _make("f", 3.0, 3)]
    stage = rrf_fuse(dense, sparse, alpha=0.5, k=60, fusion_top_k=10)
    scores = [c.score for c in stage.candidates]
    assert scores == sorted(scores, reverse=True)


def test_rank_delta_correct():
    dense = [_make("a", 0.9, 1)]
    sparse = [_make("a", 5.0, 3)]
    stage = rrf_fuse(dense, sparse, alpha=0.5, k=60, fusion_top_k=5)
    assert stage.candidates[0].metadata["rank_delta"] == 2


def test_fusion_top_k_limit():
    dense = [_make(f"d{i}", 0.9 - i * 0.01, i) for i in range(1, 11)]
    sparse = [_make(f"s{i}", 5.0 - i * 0.1, i) for i in range(1, 11)]
    stage = rrf_fuse(dense, sparse, alpha=0.5, k=60, fusion_top_k=5)
    assert len(stage.candidates) == 5


def test_ranks_sequential():
    dense = [_make("a", 0.9, 1), _make("b", 0.7, 2)]
    sparse = [_make("c", 5.0, 1)]
    stage = rrf_fuse(dense, sparse, alpha=0.5, k=60, fusion_top_k=10)
    ranks = [c.rank for c in stage.candidates]
    assert ranks == list(range(1, len(ranks) + 1))


def test_rrf_stage_has_latency():
    stage = rrf_fuse([], [], alpha=0.5, k=60, fusion_top_k=5)
    assert stage.duration_ms >= 0
    assert stage.stage == "rrf"


def test_rrf_empty_dense_no_phantom_rank1():
    stage = rrf_fuse([], [_make("c1", 5.0, 1)], alpha=0.5, k=60, fusion_top_k=5)
    assert len(stage.candidates) == 1
    assert pytest.approx(stage.candidates[0].score, rel=1e-6) == 0.5 / 61
    assert stage.candidates[0].score != pytest.approx(1.0 / 61, rel=1e-6)
    assert stage.candidates[0].metadata["dense_rank"] is None
    assert stage.candidates[0].metadata["sparse_rank"] == 1
    assert stage.candidates[0].metadata["rank_delta"] is None


def test_rrf_empty_sparse_no_phantom_rank1():
    stage = rrf_fuse([_make("d1", 0.9, 1)], [], alpha=0.5, k=60, fusion_top_k=5)
    assert len(stage.candidates) == 1
    assert pytest.approx(stage.candidates[0].score, rel=1e-6) == 0.5 / 61
    assert stage.candidates[0].score != pytest.approx(1.0 / 61, rel=1e-6)
    assert stage.candidates[0].metadata["dense_rank"] == 1
    assert stage.candidates[0].metadata["sparse_rank"] is None
    assert stage.candidates[0].metadata["rank_delta"] is None


def test_rrf_preserves_candidate_content_and_metadata():
    dense = [
        CandidateResult(
            chunk_id="d1",
            score=0.9,
            rank=1,
            content="Sample passage text content",
            metadata={"document_id": "doc-uuid-1234"},
        )
    ]
    stage = rrf_fuse(dense, [], alpha=0.5, k=60, fusion_top_k=5)
    assert stage.candidates[0].content == "Sample passage text content"
    assert stage.candidates[0].metadata["document_id"] == "doc-uuid-1234"
    assert stage.candidates[0].metadata["dense_rank"] == 1


def test_rrf_deterministic_ordering_tie_break():
    dense = [_make("d1", 0.9, 1)]
    sparse = [_make("s1", 5.0, 1)]
    # With equal weight alpha=0.5, both have identical score 0.5/61
    stage = rrf_fuse(dense, sparse, alpha=0.5, k=60, fusion_top_k=5)
    assert len(stage.candidates) == 2
    assert stage.candidates[0].chunk_id == "d1"
    assert stage.candidates[1].chunk_id == "s1"
