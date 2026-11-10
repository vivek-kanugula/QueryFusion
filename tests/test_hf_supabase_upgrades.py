"""Tests for Hugging Face Spaces + Supabase deployment upgrades and model optimizations."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine

from hybrid_rag.db.engine import connect_args
from hybrid_rag.generation.verifier import verify_citations
from hybrid_rag.retrieval.retriever import full_retrieve
from hybrid_rag.schemas.generation import Citation, GroundedAnswer
from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig
from hybrid_rag.settings import Settings, settings


def test_upgraded_settings_defaults():
    s = Settings()
    assert s.reranker_model == "BAAI/bge-reranker-base"
    assert s.gemini_verifier_model == "gemini-3.5-flash"
    assert s.gemini_model == "gemini-3.5-flash-lite"
    assert s.uvicorn_port == 7860
    assert s.max_storage_mb == 200
    assert s.max_documents_count == 50


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.fusion_top_k == 20
    assert cfg.rerank_top_k == 6
    assert cfg.expand_queries is False


def test_supabase_pgbouncer_connect_args():
    # Verify prepare_threshold=0 is set for postgres / psycopg
    assert "prepare_threshold" in connect_args
    assert connect_args["prepare_threshold"] == 0


@patch("hybrid_rag.generation.verifier._dispatch_verification")
def test_verifier_uses_gemini_verifier_model_by_default(mock_dispatch):
    mock_dispatch.return_value = '[]'
    answer = GroundedAnswer(
        answer="Test answer [C1]",
        citations=[Citation(citation_id="[C1]", chunk_id="chunk1", evidence_quote="Test quote")],
        has_sufficient_context=True,
    )
    candidates = [CandidateResult(chunk_id="chunk1", score=0.9, rank=1, content="Test quote in doc")]

    verify_citations(answer, candidates)

    mock_dispatch.assert_called_once()
    called_model = mock_dispatch.call_args[0][0]
    assert called_model == "gemini-3.5-flash"


def test_full_retrieve_with_expand_queries():
    cfg = PipelineConfig(
        retrieval_strategy="dense_only",
        expand_queries=True,
    )

    mock_db = MagicMock()
    mock_db.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

    with patch("hybrid_rag.retrieval.expansion.expand_query", return_value=["main query", "alt query 1"]) as mock_exp, \
         patch("hybrid_rag.ingest.embedder.embed_query") as mock_embed:
        import numpy as np
        mock_embed.return_value = np.zeros(384)
        trace = full_retrieve("main query", cfg, db=mock_db)

    assert trace is not None
    # Verify that expand_query was triggered
    assert mock_exp.called


def test_cr04_thread_pool_session_isolation():
    from sqlalchemy.orm import Session
    from hybrid_rag.retrieval.retriever import hybrid_retrieve

    # If a real Session (or custom object) is passed, worker_db passed to dense_retrieve must be None
    dummy_real_session = Session()

    with patch("hybrid_rag.retrieval.retriever.dense_retrieve") as mock_dense, \
         patch("hybrid_rag.retrieval.retriever.sparse_retrieve") as mock_sparse:
        mock_dense.return_value = MagicMock(candidates=[])
        mock_sparse.return_value = MagicMock(candidates=[])

        cfg = PipelineConfig()
        hybrid_retrieve("test", cfg, db=dummy_real_session)

        # Confirm dense_retrieve received worker_db=None, NOT the dummy_real_session
        assert mock_dense.called
        passed_db = mock_dense.call_args[0][2]
        assert passed_db is None


def test_cors_origins_hf_space_defaults(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    monkeypatch.delenv("FRONTEND_URL", raising=False)

    s = Settings()
    assert "https://huggingface.co" in s.cors_origins


def test_expansion_uses_cache():
    from hybrid_rag.retrieval.expansion import expand_query
    from hybrid_rag.generation.cache import LLMCache

    mock_cache = MagicMock()
    mock_cache.get.return_value = '{"queries": ["cached query 1"], "hypothetical_passage": "cached passage"}'

    with patch("hybrid_rag.retrieval.expansion.call_llm") as mock_call_llm:
        res = expand_query("what is rag", llm_cache=mock_cache)
        assert mock_call_llm.call_count == 0
        assert "what is rag" in res
        assert "cached query 1" in res
        assert "cached passage" in res


def test_expansion_respects_rate_limiter():
    from hybrid_rag.retrieval.expansion import expand_query

    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    mock_limiter = MagicMock()
    mock_limiter.acquire.return_value = False

    with patch("hybrid_rag.retrieval.expansion.call_llm") as mock_call_llm:
        res = expand_query("what is rag", llm_cache=mock_cache, rate_limiter=mock_limiter)
        assert mock_call_llm.call_count == 0
        assert res == ["what is rag"]


def test_cache_skips_sqlite_fallback_on_db_mode(tmp_path):
    from hybrid_rag.generation.cache import LLMCache

    # In use_db=True mode with no cache_dir, failing PG should NOT create an SQLite file
    cache = LLMCache(use_db=True, cache_dir=None)
    with patch("hybrid_rag.db.engine.SessionLocal", side_effect=Exception("PG Down")):
        cache.put("test_key", "test_val")
        val = cache.get("test_key")
        assert val is None
        assert cache._db_path is None


def test_crag_bounds_fusion_and_rerank_caps():
    from hybrid_rag.generation.crag import execute_corrective_fallback
    from hybrid_rag.schemas.generation import GenerationResponse, GroundedAnswer
    from hybrid_rag.schemas.pipeline import QueryTrace

    init_resp = GenerationResponse(
        query="test",
        answer=GroundedAnswer(answer="unsure", has_sufficient_context=False),
    )
    orig_trace = QueryTrace()
    cfg = PipelineConfig(fusion_top_k=20, rerank_top_k=6)

    with patch("hybrid_rag.retrieval.retriever.full_retrieve") as mock_full_retrieve, \
         patch("hybrid_rag.generation.pipeline.generate_grounded_answer") as mock_gen:
        mock_full_retrieve.return_value = QueryTrace()
        mock_gen.return_value = GenerationResponse(
            query="test",
            answer=GroundedAnswer(answer="fixed", has_sufficient_context=True),
        )

        execute_corrective_fallback(
            query="test",
            original_trace=orig_trace,
            initial_response=init_resp,
            reason="insufficient context",
            config=cfg,
        )

        assert mock_full_retrieve.called
        corrected_cfg = mock_full_retrieve.call_args[0][1]
        assert corrected_cfg.fusion_top_k <= 30
        assert corrected_cfg.rerank_top_k <= 8


