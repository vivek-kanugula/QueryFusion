"""Unit tests for Azure optimizations: LRU cache, adaptive rerank bypass, and BM25 mmap persistence."""

from unittest.mock import MagicMock, patch

import numpy as np

from hybrid_rag.ingest.embedder import _embed_query_cached, embed_query, reset_embedder
from hybrid_rag.retrieval.bm25_store import Bm25Store
from hybrid_rag.retrieval.retriever import should_skip_reranker
from hybrid_rag.schemas.pipeline import CandidateResult


def test_embed_query_lru_cache():
    reset_embedder()
    _embed_query_cached.cache_clear()

    with patch("hybrid_rag.ingest.embedder.get_embedder") as mock_get_embedder:
        mock_model = MagicMock()
        mock_model.embed.return_value = [np.array([0.1, 0.2, 0.3], dtype=np.float32)]
        mock_get_embedder.return_value = mock_model

        v1 = embed_query("test query")
        assert len(v1) == 3
        assert mock_model.embed.call_count == 1

        # Second call with identical query should hit cache (0 additional model.embed calls)
        v2 = embed_query("test query")
        assert np.allclose(v1, v2)
        assert mock_model.embed.call_count == 1

        # Cache info check
        info = _embed_query_cached.cache_info()
        assert info.hits >= 1


def test_adaptive_reranker_bypass_dominant_winner():
    # Candidate 1 is clear winner: score 0.030 vs 0.015 (ratio = 2.0 > 1.35)
    candidates = [
        CandidateResult(chunk_id="c1", score=0.030, rank=1, content="best chunk"),
        CandidateResult(chunk_id="c2", score=0.015, rank=2, content="second chunk"),
    ]
    skip, reason = should_skip_reranker(candidates, ratio_threshold=1.35)
    assert skip is True
    assert "high_confidence" in reason


def test_adaptive_reranker_bypass_close_tie_does_not_skip():
    # Candidate 1 and 2 are in a dead heat: 0.0165 vs 0.0160 (ratio = 1.03 < 1.35)
    candidates = [
        CandidateResult(chunk_id="c1", score=0.0165, rank=1, content="chunk A"),
        CandidateResult(chunk_id="c2", score=0.0160, rank=2, content="chunk B"),
    ]
    skip, reason = should_skip_reranker(candidates, ratio_threshold=1.35)
    assert skip is False
    assert "ambiguous" in reason


def test_bm25_store_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("BM25_CACHE_DIR", str(tmp_path / "bm25_local"))
    monkeypatch.setenv("AZURE_PERSISTENT_PATH", str(tmp_path / "bm25_persistent"))

    store = Bm25Store()
    docs = ["quantum computing algorithms", "relational database design"]
    ids = ["c1", "c2"]

    count = store.rebuild(docs, ids, "fixed")
    assert count == 2
    assert store.has_index("fixed")

    results = store.search("quantum", "fixed", top_k=1)
    assert len(results) == 1
    assert results[0][0] == "c1"

    # Verify persistent files were saved
    local_dir = tmp_path / "bm25_local" / "fixed"
    assert (local_dir / "meta.json").exists()

    # Create fresh store and verify loading from disk
    new_store = Bm25Store()
    assert new_store.has_index("fixed") is True
    res2 = new_store.search("quantum", "fixed", top_k=1)
    assert len(res2) == 1
    assert res2[0][0] == "c1"
