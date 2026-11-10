"""Unit, integration, and regression tests for Milestone 1: Retrieval & Core Engine.

Covers:
- CR-03: BM25 Index Crash on Empty / Stopword-Only Vocabulary (bm25_store.py)
- CR-04: User-Selected retrieval_strategy Ignored by Retrieval Engine (retriever.py & api/query.py)
- CR-05: Unbounded In-Memory Accumulation in JobRegistry (jobs.py)
- LO-02: Non-ASCII Characters Stripped by BM25 Technical Tokenizer (tokenizer.py)
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hybrid_rag.api.query import QueryRequest, router as query_router
from hybrid_rag.core.jobs import JobRegistry, JobStatus
from hybrid_rag.retrieval.bm25_store import Bm25Store
from hybrid_rag.retrieval.retriever import (
    dense_retrieve,
    full_retrieve,
    multi_query_retrieve,
    sparse_retrieve,
)
from hybrid_rag.retrieval.tokenizer import (
    expand_technical_tokens,
    tokenize_technical,
)
from hybrid_rag.schemas.pipeline import (
    CandidateResult,
    PipelineConfig,
    QueryTrace,
    StageTrace,
)


# =====================================================================
# CR-03 Tests: BM25 Index Crash on Empty / Stopword-Only Vocabulary
# =====================================================================

def test_bm25_rebuild_stopword_only_vocab():
    """Verify rebuilding BM25 index with stopword-only texts does not crash."""
    store = Bm25Store()
    stopwords_text = ["the and is", "at by for", "of on with"]
    chunk_ids = ["c1", "c2", "c3"]

    indexed_count = store.rebuild(stopwords_text, chunk_ids, "fixed")
    assert indexed_count == 0
    assert not store.has_index("fixed")
    assert store.search("the", "fixed") == []


def test_bm25_rebuild_whitespace_and_punctuation_only():
    """Verify empty, whitespace, and punctuation texts return 0 without ValueError."""
    store = Bm25Store()
    blank_texts = ["", "   ", "\n\t", "!!! ??? ..."]
    chunk_ids = [f"c{i}" for i in range(len(blank_texts))]

    indexed_count = store.rebuild(blank_texts, chunk_ids, "fixed")
    assert indexed_count == 0
    assert not store.has_index("fixed")
    assert store.search("anything", "fixed") == []


def test_bm25_rebuild_clears_previous_valid_index():
    """Verify an existing valid index is cleanly removed when rebuilt with stopword-only text."""
    store = Bm25Store()
    # 1. First index valid terms
    valid_texts = ["PostgreSQL pgvector extension", "BM25 keyword search index"]
    cids = ["v1", "v2"]
    store.rebuild(valid_texts, cids, "fixed")
    assert store.has_index("fixed")
    assert len(store.search("pgvector", "fixed")) > 0

    # 2. Rebuild same strategy with stopword-only text
    store.rebuild(["the is at", "and"], ["s1", "s2"], "fixed")
    assert not store.has_index("fixed")
    assert store.search("pgvector", "fixed") == []


def test_bm25_rebuild_mixed_valid_and_stopword_texts():
    """Verify mixed valid and stopword texts succeed and index valid terms."""
    store = Bm25Store()
    texts = ["the and is", "Distributed database transactions and consensus algorithms"]
    cids = ["m1", "m2"]

    indexed_count = store.rebuild(texts, cids, "hierarchical")
    assert indexed_count == 2
    assert store.has_index("hierarchical")
    res = store.search("consensus", "hierarchical")
    assert len(res) == 1
    assert res[0][0] == "m2"


def test_bm25_rebuild_defensive_value_error_handling():
    """Verify defensive try/except catches unexpected ValueError during indexing."""
    store = Bm25Store()
    with patch("bm25s.BM25.index", side_effect=ValueError("max() iterable argument is empty")):
        count = store.rebuild(["valid content text"], ["c1"], "fixed")
        assert count == 0
        assert not store.has_index("fixed")


# =====================================================================
# CR-04 Tests: User-Selected retrieval_strategy
# =====================================================================

def _make_candidate(cid: str, score: float, rank: int, source: str) -> CandidateResult:
    return CandidateResult(
        chunk_id=cid,
        score=score,
        rank=rank,
        content=f"Content for {cid}",
        metadata={"document_id": "doc-1", "source": source},
    )


def test_full_retrieve_dense_only_bypasses_sparse_and_rrf():
    """Verify 'dense_only' retrieval strategy executes only dense retrieval."""
    config = PipelineConfig(retrieval_strategy="dense_only", rerank_enabled=False)
    dense_candidate = _make_candidate("c_dense", 0.9, 1, "dense")
    mock_dense_stage = StageTrace(
        stage="dense",
        candidates=[dense_candidate],
        duration_ms=10.0,
    )

    with (
        patch("hybrid_rag.retrieval.retriever.dense_retrieve", return_value=mock_dense_stage) as mock_dense,
        patch("hybrid_rag.retrieval.retriever.sparse_retrieve") as mock_sparse,
        patch("hybrid_rag.retrieval.retriever.rrf_fuse") as mock_rrf,
        patch("hybrid_rag.retrieval.retriever.hydrate_candidates"),
    ):
        trace = full_retrieve("test query", config)

        mock_dense.assert_called_once()
        mock_sparse.assert_not_called()
        mock_rrf.assert_not_called()

        assert len(trace.stages) == 1
        assert trace.stages[0].stage == "dense"
        assert trace.stages[0].candidates[0].chunk_id == "c_dense"


def test_full_retrieve_dense_alias_bypasses_sparse_and_rrf():
    """Verify 'dense' alias also bypasses sparse retrieval and RRF."""
    config = PipelineConfig(retrieval_strategy="dense", rerank_enabled=False)
    mock_stage = StageTrace(
        stage="dense",
        candidates=[_make_candidate("c1", 0.85, 1, "dense")],
        duration_ms=8.0,
    )

    with (
        patch("hybrid_rag.retrieval.retriever.dense_retrieve", return_value=mock_stage) as mock_dense,
        patch("hybrid_rag.retrieval.retriever.sparse_retrieve") as mock_sparse,
        patch("hybrid_rag.retrieval.retriever.rrf_fuse") as mock_rrf,
        patch("hybrid_rag.retrieval.retriever.hydrate_candidates"),
    ):
        trace = full_retrieve("query", config)
        mock_dense.assert_called_once()
        mock_sparse.assert_not_called()
        mock_rrf.assert_not_called()
        assert [s.stage for s in trace.stages] == ["dense"]


def test_full_retrieve_bm25_only_bypasses_dense_and_rrf():
    """Verify 'bm25_only' retrieval strategy executes only sparse retrieval."""
    config = PipelineConfig(retrieval_strategy="bm25_only", rerank_enabled=False)
    sparse_candidate = _make_candidate("c_sparse", 5.2, 1, "sparse")
    mock_sparse_stage = StageTrace(
        stage="sparse",
        candidates=[sparse_candidate],
        duration_ms=5.0,
    )

    with (
        patch("hybrid_rag.retrieval.retriever.sparse_retrieve", return_value=mock_sparse_stage) as mock_sparse,
        patch("hybrid_rag.retrieval.retriever.dense_retrieve") as mock_dense,
        patch("hybrid_rag.retrieval.retriever.rrf_fuse") as mock_rrf,
        patch("hybrid_rag.retrieval.retriever.hydrate_candidates") as mock_hydrate,
    ):
        trace = full_retrieve("test query", config)

        mock_sparse.assert_called_once()
        mock_dense.assert_not_called()
        mock_rrf.assert_not_called()
        mock_hydrate.assert_called_once()

        assert len(trace.stages) == 1
        assert trace.stages[0].stage == "sparse"
        assert trace.stages[0].candidates[0].chunk_id == "c_sparse"


def test_full_retrieve_sparse_alias_bypasses_dense_and_rrf():
    """Verify 'sparse' alias also bypasses dense retrieval and RRF."""
    config = PipelineConfig(retrieval_strategy="sparse", rerank_enabled=False)
    mock_stage = StageTrace(
        stage="sparse",
        candidates=[_make_candidate("c1", 3.0, 1, "sparse")],
        duration_ms=4.0,
    )

    with (
        patch("hybrid_rag.retrieval.retriever.sparse_retrieve", return_value=mock_stage) as mock_sparse,
        patch("hybrid_rag.retrieval.retriever.dense_retrieve") as mock_dense,
        patch("hybrid_rag.retrieval.retriever.rrf_fuse") as mock_rrf,
        patch("hybrid_rag.retrieval.retriever.hydrate_candidates"),
    ):
        trace = full_retrieve("query", config)
        mock_sparse.assert_called_once()
        mock_dense.assert_not_called()
        mock_rrf.assert_not_called()
        assert [s.stage for s in trace.stages] == ["sparse"]


def test_full_retrieve_hybrid_executes_hybrid_retrieve():
    """Verify 'hybrid' and 'hybrid_rerank' run hybrid_retrieve and RRF."""
    config = PipelineConfig(retrieval_strategy="hybrid_rerank", rerank_enabled=False)
    mock_hybrid_trace = QueryTrace(
        query="query",
        config=config,
        stages=[
            StageTrace(stage="dense", candidates=[], duration_ms=5.0),
            StageTrace(stage="sparse", candidates=[], duration_ms=5.0),
            StageTrace(
                stage="rrf",
                candidates=[_make_candidate("fused1", 0.03, 1, "fused")],
                duration_ms=2.0,
            ),
        ],
    )

    with (
        patch("hybrid_rag.retrieval.retriever.hybrid_retrieve", return_value=mock_hybrid_trace) as mock_hybrid,
        patch("hybrid_rag.retrieval.retriever.hydrate_candidates"),
    ):
        trace = full_retrieve("query", config)
        mock_hybrid.assert_called_once()
        assert [s.stage for s in trace.stages] == ["dense", "sparse", "rrf"]


def test_full_retrieve_strategy_with_rerank():
    """Verify dense_only and bm25_only candidates are reranked when rerank_enabled=True."""
    config = PipelineConfig(retrieval_strategy="dense_only", rerank_enabled=True, rerank_top_k=5)
    mock_dense_stage = StageTrace(
        stage="dense",
        candidates=[_make_candidate("c1", 0.9, 1, "dense")],
        duration_ms=10.0,
    )
    mock_rerank_stage = StageTrace(
        stage="rerank",
        candidates=[_make_candidate("c1", 0.99, 1, "rerank")],
        duration_ms=15.0,
    )

    with (
        patch("hybrid_rag.retrieval.retriever.dense_retrieve", return_value=mock_dense_stage),
        patch("hybrid_rag.retrieval.reranker.rerank_candidates", return_value=mock_rerank_stage) as mock_rerank,
        patch("hybrid_rag.retrieval.retriever.hydrate_candidates"),
    ):
        trace = full_retrieve("query", config)
        mock_rerank.assert_called_once()
        assert [s.stage for s in trace.stages] == ["dense", "rerank"]


def test_multi_query_retrieve_respects_strategy():
    """Verify multi_query_retrieve routes sub-queries by active strategy."""
    dense_config = PipelineConfig(retrieval_strategy="dense_only", rerank_enabled=False)
    mock_dense_stage = StageTrace(
        stage="dense",
        candidates=[_make_candidate("c1", 0.9, 1, "dense")],
        duration_ms=5.0,
    )

    with (
        patch("hybrid_rag.retrieval.retriever.dense_retrieve", return_value=mock_dense_stage) as mock_dense,
        patch("hybrid_rag.retrieval.retriever.sparse_retrieve") as mock_sparse,
        patch("hybrid_rag.retrieval.retriever.hydrate_candidates"),
    ):
        trace = multi_query_retrieve(["q1", "q2"], dense_config)
        assert mock_dense.call_count == 2
        mock_sparse.assert_not_called()
        assert trace.stages[-1].stage == "multi_query_rrf"


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


def test_query_stream_api_dense_only_stages():
    """Verify /api/query/stream emits only 'dense' stage for dense_only strategy."""
    app = FastAPI()
    app.include_router(query_router)
    client = TestClient(app)

    mock_trace = QueryTrace(
        query="test",
        config=PipelineConfig(retrieval_strategy="dense_only", rerank_enabled=False),
        stages=[StageTrace(stage="dense", candidates=[_make_candidate("c1", 0.9, 1, "dense")], duration_ms=5.0)],
    )

    with (
        patch("hybrid_rag.api.query.trace_store"),
        patch("hybrid_rag.retrieval.retriever.full_retrieve", return_value=mock_trace),
        patch("hybrid_rag.retrieval.bm25_store.get_bm25_store", return_value=MagicMock()),
        patch("hybrid_rag.generation.pipeline.generate_grounded_answer") as mock_gen,
    ):
        from hybrid_rag.schemas.generation import (
            Citation,
            GenerationResponse,
            GroundedAnswer,
            VerificationResult,
            CompositeConfidence,
        )
        mock_gen.return_value = GenerationResponse(
            query="test",
            answer=GroundedAnswer(answer="Ans", citations=[]),
            verification=VerificationResult(citation_coverage=1.0, all_supported=True, claims=[]),
            confidence=CompositeConfidence(overall_score=0.9, retrieval_score=0.9, citation_coverage=1.0, completeness_score=0.9),
            trace_id="test",
            model="gemini",
            cached=False,
        )

        resp = client.post("/api/query/stream", json={"question": "test", "retrieval_strategy": "dense_only", "rerank_enabled": False})
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        stage_names = [e[1].get("name") for e in events if e[0] == "stage"]

        assert "dense" in stage_names
        assert "sparse" not in stage_names
        assert "rrf" not in stage_names


def test_query_stream_api_bm25_only_stages():
    """Verify /api/query/stream emits only 'sparse' stage for bm25_only strategy."""
    app = FastAPI()
    app.include_router(query_router)
    client = TestClient(app)

    mock_trace = QueryTrace(
        query="test",
        config=PipelineConfig(retrieval_strategy="bm25_only", rerank_enabled=False),
        stages=[StageTrace(stage="sparse", candidates=[_make_candidate("c1", 3.0, 1, "sparse")], duration_ms=5.0)],
    )

    with (
        patch("hybrid_rag.api.query.trace_store"),
        patch("hybrid_rag.retrieval.retriever.full_retrieve", return_value=mock_trace),
        patch("hybrid_rag.retrieval.bm25_store.get_bm25_store", return_value=MagicMock()),
        patch("hybrid_rag.generation.pipeline.generate_grounded_answer") as mock_gen,
    ):
        from hybrid_rag.schemas.generation import (
            Citation,
            GenerationResponse,
            GroundedAnswer,
            VerificationResult,
            CompositeConfidence,
        )
        mock_gen.return_value = GenerationResponse(
            query="test",
            answer=GroundedAnswer(answer="Ans", citations=[]),
            verification=VerificationResult(citation_coverage=1.0, all_supported=True, claims=[]),
            confidence=CompositeConfidence(overall_score=0.9, retrieval_score=0.9, citation_coverage=1.0, completeness_score=0.9),
            trace_id="test",
            model="gemini",
            cached=False,
        )

        resp = client.post("/api/query/stream", json={"question": "test", "retrieval_strategy": "bm25_only", "rerank_enabled": False})
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        stage_names = [e[1].get("name") for e in events if e[0] == "stage"]

        assert "sparse" in stage_names
        assert "dense" not in stage_names
        assert "rrf" not in stage_names


# =====================================================================
# CR-05 Tests: Bounded Capacity & Eviction in JobRegistry
# =====================================================================

def test_job_registry_bounded_capacity_evicts_terminal_jobs():
    """Verify JobRegistry enforces max_jobs and evicts completed jobs."""
    reg = JobRegistry(max_jobs=4)
    jobs = [reg.create_job("ingest") for _ in range(4)]
    assert reg.count() == 4

    # Complete the first two jobs
    reg.complete_job(jobs[0].job_id)
    reg.complete_job(jobs[1].job_id)

    # Creating a 5th job must evict the oldest terminal job (jobs[0])
    job5 = reg.create_job("eval")
    assert reg.count() == 4
    assert reg.get_job(jobs[0].job_id) is None
    assert reg.get_job(jobs[1].job_id) is not None
    assert reg.get_job(job5.job_id) is not None

    # Check subscribers map does not retain evicted job
    with reg._lock:
        assert jobs[0].job_id not in reg._subscribers


def test_job_registry_evicts_terminal_before_active():
    """Verify JobRegistry preserves active RUNNING/PENDING jobs over terminal jobs."""
    reg = JobRegistry(max_jobs=3)

    j_running = reg.create_job("ingest")
    reg.update_progress(j_running.job_id, 25.0)

    j_failed = reg.create_job("eval")
    reg.fail_job(j_failed.job_id, "error")

    j_pending = reg.create_job("ingest")

    # Capacity is 3 (j_running, j_failed, j_pending).
    assert reg.count() == 3

    # Add 4th job -> j_failed is the only terminal job, so it MUST be evicted first
    j_new = reg.create_job("query")
    assert reg.count() == 3
    assert reg.get_job(j_failed.job_id) is None
    assert reg.get_job(j_running.job_id) is not None
    assert reg.get_job(j_pending.job_id) is not None
    assert reg.get_job(j_new.job_id) is not None


def test_job_registry_fifo_eviction_when_all_active():
    """Verify JobRegistry uses FIFO eviction when all jobs are non-terminal."""
    reg = JobRegistry(max_jobs=3)
    j1 = reg.create_job("t1")
    j2 = reg.create_job("t2")
    j3 = reg.create_job("t3")
    assert reg.count() == 3

    # All 3 are PENDING. Create 4th job. Oldest (j1) should be evicted.
    j4 = reg.create_job("t4")
    assert reg.count() == 3
    assert reg.get_job(j1.job_id) is None
    assert reg.get_job(j2.job_id) is not None
    assert reg.get_job(j3.job_id) is not None
    assert reg.get_job(j4.job_id) is not None


def test_job_registry_clear():
    """Verify clear() empties jobs and subscribers collections."""
    reg = JobRegistry(max_jobs=10)
    j1 = reg.create_job("j1")
    reg.complete_job(j1.job_id)
    reg.clear()

    assert reg.count() == 0
    with reg._lock:
        assert len(reg._jobs) == 0
        assert len(reg._subscribers) == 0


@pytest.mark.asyncio
async def test_job_registry_subscribe_evicted_job_clean_exit():
    """Verify subscribing to an evicted job terminates cleanly with zero events."""
    reg = JobRegistry(max_jobs=2)
    j1 = reg.create_job("test")
    reg.complete_job(j1.job_id)
    reg.create_job("test2")
    # j3 causes j1 to be evicted
    reg.create_job("test3")

    assert reg.get_job(j1.job_id) is None

    events = []
    async for event in reg.subscribe(j1.job_id):
        events.append(event)

    assert len(events) == 0


# =====================================================================
# LO-02 Tests: Non-ASCII Characters in Tokenizer
# =====================================================================

def test_tokenize_technical_accented_latin_characters():
    """Verify Latin words with accents/diacritics are preserved intact."""
    text = "café München naïve résumé façade año"
    tokens = tokenize_technical(text)

    assert "café" in tokens
    assert "münchen" in tokens
    assert "résumé" in tokens
    assert "año" in tokens
    # Diacritic characters must be preserved in stemmed forms (naïv, façad)
    assert any("ï" in t for t in tokens)
    assert any("ç" in t for t in tokens)
    # Must not contain fragmented single letters from stripped diacritics
    assert "m" not in tokens
    assert "nchen" not in tokens
    assert "sum" not in tokens


def test_tokenize_technical_non_latin_scripts():
    """Verify Cyrillic, Greek, and other non-Latin alphabets are preserved."""
    cyrillic_tokens = tokenize_technical("привет мир")
    assert "привет" in cyrillic_tokens
    assert "мир" in cyrillic_tokens

    greek_tokens = tokenize_technical("καλημέρα κόσμε")
    assert "καλημέρα" in greek_tokens
    assert "κόσμε" in greek_tokens


def test_expand_technical_tokens_non_ascii_identifiers():
    """Verify expand_technical_tokens supports non-ASCII technical tokens."""
    expanded1 = expand_technical_tokens("München-2024.log")
    assert "München2024log" in expanded1
    assert "München" in expanded1

    expanded2 = expand_technical_tokens("café_api_v1")
    assert "caféapiv1" in expanded2
    assert "café" in expanded2

    expanded3 = expand_technical_tokens("кириллица-модуль_v2")
    assert "кириллицамодульv2" in expanded3
    assert "кириллица" in expanded3


def test_tokenize_technical_composite_non_ascii():
    """Verify full technical tokenization handles composite non-ASCII terms."""
    tokens = tokenize_technical("Check München-2024.log error in café_api_v1")
    assert "münchen2024log" in tokens
    assert "münchen" in tokens
    assert "caféapiv1" in tokens
    assert "café" in tokens
