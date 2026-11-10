"""Adversarial stress harness for Milestone 1: Retrieval & Core Engine.

Focuses specifically on:
- CR-04: retrieval_strategy routing & SSE filtering
- CR-05: JobRegistry bounded capacity, concurrency, eviction & subscriber safety
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hybrid_rag.api.query import router as query_router
from hybrid_rag.core.jobs import JobRegistry
from hybrid_rag.retrieval.bm25_store import Bm25Store
from hybrid_rag.retrieval.retriever import (
    full_retrieve,
    multi_query_retrieve,
)
from hybrid_rag.schemas.generation import (
    CompositeConfidence,
    GenerationResponse,
    GroundedAnswer,
    VerificationResult,
)
from hybrid_rag.schemas.pipeline import (
    CandidateResult,
    PipelineConfig,
    QueryTrace,
    StageTrace,
)


def _make_candidate(cid: str, score: float, rank: int, source: str) -> CandidateResult:
    return CandidateResult(
        chunk_id=cid,
        score=score,
        rank=rank,
        content=f"Content for {cid}",
        metadata={"document_id": str(uuid.uuid4()), "source": source},
    )


# =====================================================================
# CR-04 Adversarial Suite
# =====================================================================

class TestCR04RetrievalStrategyRouting:
    """Stress-test retrieval_strategy routing, isolation, and candidate hydration."""

    @pytest.mark.parametrize("strategy", ["dense_only", "dense", "DENSE_ONLY", "DENSE"])
    def test_full_retrieve_dense_only_never_invokes_sparse_or_rrf(self, strategy: str):
        """CR-04: 'dense_only' strategy must strictly bypass sparse retrieval and RRF."""
        config = PipelineConfig(retrieval_strategy=strategy, rerank_enabled=False)
        mock_stage = StageTrace(
            stage="dense",
            candidates=[_make_candidate("c_dense", 0.95, 1, "dense")],
            duration_ms=5.0,
        )

        mock_bm25 = MagicMock(spec=Bm25Store)
        mock_bm25.search.side_effect = RuntimeError("BM25Store.search must NEVER be invoked in dense_only mode!")

        with (
            patch("hybrid_rag.retrieval.retriever.dense_retrieve", return_value=mock_stage) as mock_dense,
            patch("hybrid_rag.retrieval.retriever.sparse_retrieve", side_effect=RuntimeError("sparse_retrieve must NEVER be called!")),
            patch("hybrid_rag.retrieval.retriever.rrf_fuse", side_effect=RuntimeError("rrf_fuse must NEVER be called!")),
            patch("hybrid_rag.retrieval.retriever.hydrate_candidates"),
        ):
            trace = full_retrieve("test query", config, bm25_store=mock_bm25)

            mock_dense.assert_called_once()
            mock_bm25.search.assert_not_called()
            assert [s.stage for s in trace.stages] == ["dense"]
            assert trace.stages[0].candidates[0].chunk_id == "c_dense"

    @pytest.mark.parametrize("strategy", ["bm25_only", "bm25", "sparse", "BM25_ONLY", "SPARSE"])
    def test_full_retrieve_bm25_only_never_invokes_dense_or_rrf(self, strategy: str):
        """CR-04: 'bm25_only' strategy must strictly bypass dense retrieval, embedder, and RRF."""
        config = PipelineConfig(retrieval_strategy=strategy, rerank_enabled=False)
        mock_stage = StageTrace(
            stage="sparse",
            candidates=[_make_candidate("c_sparse", 4.5, 1, "sparse")],
            duration_ms=3.0,
        )

        with (
            patch("hybrid_rag.retrieval.retriever.sparse_retrieve", return_value=mock_stage) as mock_sparse,
            patch("hybrid_rag.retrieval.retriever.dense_retrieve", side_effect=RuntimeError("dense_retrieve must NEVER be called!")),
            patch("hybrid_rag.retrieval.retriever.rrf_fuse", side_effect=RuntimeError("rrf_fuse must NEVER be called!")),
            patch("hybrid_rag.ingest.embedder.embed_query", side_effect=RuntimeError("embed_query must NEVER be called!")),
            patch("hybrid_rag.retrieval.retriever.hydrate_candidates") as mock_hydrate,
        ):
            trace = full_retrieve("test query", config)

            mock_sparse.assert_called_once()
            mock_hydrate.assert_called_once()
            assert [s.stage for s in trace.stages] == ["sparse"]
            assert trace.stages[0].candidates[0].chunk_id == "c_sparse"

    def test_full_retrieve_bm25_only_hydrates_candidates_from_db(self):
        """CR-04: BM25 candidates initially lacking content MUST be hydrated with full content from PostgreSQL."""
        chunk_uuid = uuid.uuid4()
        chunk_id_str = str(chunk_uuid)
        doc_uuid = uuid.uuid4()

        # BM25 initially yields CandidateResult with empty content and no document_id in metadata
        raw_bm25_candidate = CandidateResult(
            chunk_id=chunk_id_str,
            score=7.8,
            rank=1,
            content="",
            metadata={"source": "sparse"},
        )
        mock_sparse_stage = StageTrace(
            stage="sparse",
            candidates=[raw_bm25_candidate],
            duration_ms=4.0,
        )

        # Mock DB chunk
        mock_db_chunk = MagicMock()
        mock_db_chunk.id = chunk_uuid
        mock_db_chunk.content = "Complete text retrieved directly from PostgreSQL DocumentChunk table"
        mock_db_chunk.document_id = doc_uuid

        mock_db_session = MagicMock()
        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_db_chunk]

        config = PipelineConfig(retrieval_strategy="bm25_only", rerank_enabled=False)

        with (
            patch("hybrid_rag.retrieval.retriever.sparse_retrieve", return_value=mock_sparse_stage),
            patch("hybrid_rag.retrieval.retriever.dense_retrieve", side_effect=RuntimeError("dense_retrieve called in bm25_only")),
            patch("hybrid_rag.retrieval.retriever.rrf_fuse", side_effect=RuntimeError("rrf_fuse called in bm25_only")),
        ):
            trace = full_retrieve("quantum computing", config, db=mock_db_session)

            assert len(trace.stages) == 1
            cand = trace.stages[0].candidates[0]
            assert cand.chunk_id == chunk_id_str
            assert cand.content == "Complete text retrieved directly from PostgreSQL DocumentChunk table"
            assert cand.metadata.get("document_id") == str(doc_uuid)

    def test_full_retrieve_bm25_only_with_document_ids_filtering(self):
        """CR-04: When document_ids is specified, only chunks matching allowed IDs survive."""
        doc1_uuid = uuid.uuid4()
        doc2_uuid = uuid.uuid4()
        c1_uuid = uuid.uuid4()
        c2_uuid = uuid.uuid4()

        cand1 = CandidateResult(chunk_id=str(c1_uuid), score=5.0, rank=1, content="", metadata={"source": "sparse"})
        cand2 = CandidateResult(chunk_id=str(c2_uuid), score=4.0, rank=2, content="", metadata={"source": "sparse"})
        mock_sparse_stage = StageTrace(stage="sparse", candidates=[cand1, cand2], duration_ms=5.0)

        mock_chunk1 = MagicMock(id=c1_uuid, content="Doc 1 chunk", document_id=doc1_uuid)
        mock_chunk2 = MagicMock(id=c2_uuid, content="Doc 2 chunk", document_id=doc2_uuid)

        mock_db_session = MagicMock()
        mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_chunk1, mock_chunk2]

        # Only allow doc1
        config = PipelineConfig(
            retrieval_strategy="bm25_only",
            document_ids=(str(doc1_uuid),),
            rerank_enabled=False,
        )

        with (
            patch("hybrid_rag.retrieval.retriever.sparse_retrieve", return_value=mock_sparse_stage),
            patch("hybrid_rag.retrieval.retriever.dense_retrieve", side_effect=RuntimeError),
            patch("hybrid_rag.retrieval.retriever.rrf_fuse", side_effect=RuntimeError),
        ):
            trace = full_retrieve("test", config, db=mock_db_session)
            final_cands = trace.stages[-1].candidates
            assert len(final_cands) == 1
            assert final_cands[0].chunk_id == str(c1_uuid)
            assert final_cands[0].metadata["document_id"] == str(doc1_uuid)

    def test_multi_query_retrieve_dense_only_respects_strategy(self):
        """CR-04: multi_query_retrieve with 'dense_only' must execute dense_retrieve for each query and bypass sparse."""
        config = PipelineConfig(retrieval_strategy="dense_only", rerank_enabled=False)
        queries = ["query variation 1", "query variation 2", "query variation 3"]

        def mock_dense_impl(q, cfg, db=None):
            return StageTrace(
                stage="dense",
                candidates=[_make_candidate(f"c_{q.replace(' ', '_')}", 0.9, 1, "dense")],
                duration_ms=4.0,
            )

        with (
            patch("hybrid_rag.retrieval.retriever.dense_retrieve", side_effect=mock_dense_impl) as mock_dense,
            patch("hybrid_rag.retrieval.retriever.sparse_retrieve", side_effect=RuntimeError("sparse_retrieve called in multi_query dense_only")),
            patch("hybrid_rag.retrieval.retriever.hydrate_candidates"),
        ):
            trace = multi_query_retrieve(queries, config)
            assert mock_dense.call_count == 3
            assert trace.stages[-1].stage == "multi_query_rrf"
            assert trace.stages[-1].metadata["expanded_queries"] == queries
            assert len(trace.stages[-1].candidates) == 3

    def test_multi_query_retrieve_bm25_only_respects_strategy(self):
        """CR-04: multi_query_retrieve with 'bm25_only' must execute sparse_retrieve for each query and bypass dense."""
        config = PipelineConfig(retrieval_strategy="bm25_only", rerank_enabled=False)
        queries = ["bm25 variation 1", "bm25 variation 2"]

        def mock_sparse_impl(q, cfg, bm25):
            return StageTrace(
                stage="sparse",
                candidates=[_make_candidate(f"c_{q[:4]}", 6.0, 1, "sparse")],
                duration_ms=3.0,
            )

        with (
            patch("hybrid_rag.retrieval.retriever.sparse_retrieve", side_effect=mock_sparse_impl) as mock_sparse,
            patch("hybrid_rag.retrieval.retriever.dense_retrieve", side_effect=RuntimeError("dense_retrieve called in multi_query bm25_only")),
            patch("hybrid_rag.retrieval.retriever.hydrate_candidates") as mock_hydrate,
        ):
            trace = multi_query_retrieve(queries, config, bm25_store=MagicMock())
            assert mock_sparse.call_count == 2
            mock_hydrate.assert_called_once()
            assert trace.stages[-1].stage == "multi_query_rrf"


def _parse_sse_events(raw: str) -> list[tuple[str, dict]]:
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


class TestCR04QueryStreamSSE:
    """Stress-test SSE stage event streams under all retrieval_strategy options."""

    @pytest.fixture
    def test_client(self):
        app = FastAPI()
        app.include_router(query_router)
        return TestClient(app)

    @pytest.mark.parametrize(
        "strategy,expected_stages,forbidden_stages",
        [
            ("dense_only", ["dense"], ["sparse", "rrf"]),
            ("dense", ["dense"], ["sparse", "rrf"]),
            ("bm25_only", ["sparse"], ["dense", "rrf"]),
            ("bm25", ["sparse"], ["dense", "rrf"]),
            ("sparse", ["sparse"], ["dense", "rrf"]),
            ("hybrid", ["dense", "sparse", "rrf"], []),
            ("hybrid_rerank", ["dense", "sparse", "rrf"], []),
        ],
    )
    def test_query_stream_sse_stages_matrix(
        self,
        test_client,
        strategy: str,
        expected_stages: list[str],
        forbidden_stages: list[str],
    ):
        """CR-04: /api/query/stream must yield truthful stage events matching user's retrieval_strategy."""
        primary_stage = "dense" if "dense" in strategy else ("sparse" if "bm25" in strategy or "sparse" in strategy else "rrf")
        mock_trace = QueryTrace(
            query="test query",
            config=PipelineConfig(retrieval_strategy=strategy, rerank_enabled=False),
            stages=[StageTrace(stage=primary_stage, candidates=[_make_candidate("c1", 0.9, 1, primary_stage)], duration_ms=5.0)],
        )

        with (
            patch("hybrid_rag.api.query.trace_store"),
            patch("hybrid_rag.retrieval.retriever.full_retrieve", return_value=mock_trace),
            patch("hybrid_rag.retrieval.bm25_store.get_bm25_store", return_value=MagicMock()),
            patch("hybrid_rag.generation.pipeline.generate_grounded_answer") as mock_gen,
        ):
            mock_gen.return_value = GenerationResponse(
                query="test query",
                answer=GroundedAnswer(answer="Answer", citations=[]),
                verification=VerificationResult(citation_coverage=1.0, all_supported=True, claims=[]),
                confidence=CompositeConfidence(overall_score=0.9, retrieval_score=0.9, citation_coverage=1.0, completeness_score=0.9),
                trace_id="test",
                model="test",
                cached=False,
            )

            resp = test_client.post(
                "/api/query/stream",
                json={"question": "test query", "retrieval_strategy": strategy, "rerank_enabled": False},
            )
            assert resp.status_code == 200
            events = _parse_sse_events(resp.text)
            stage_names = [e[1].get("name") for e in events if e[0] == "stage"]

            for exp in expected_stages:
                assert exp in stage_names, f"Expected stage '{exp}' missing for strategy '{strategy}'"
            for fbd in forbidden_stages:
                assert fbd not in stage_names, f"Forbidden stage '{fbd}' present for strategy '{strategy}'"

            # Always present lifecycle events
            assert "retrieval_complete" in stage_names
            assert "generation" in stage_names


# =====================================================================
# CR-05 Adversarial Suite: Concurrency, Eviction, Memory Leaks & Locks
# =====================================================================

class TestCR05JobRegistryAdversarial:
    """Stress-test JobRegistry under heavy concurrency, capacity bounds, and subscriber lifecycles."""

    def test_high_throughput_concurrent_job_creation_bounds(self):
        """CR-05: Concurrently create 100 jobs across 10 threads with max_jobs=5.

        Asserts that count() <= max_jobs at all times.
        """
        max_jobs = 5
        reg = JobRegistry(max_jobs=max_jobs)
        total_jobs_to_create = 100
        concurrency = 10
        jobs_per_thread = total_jobs_to_create // concurrency

        violations = []
        created_job_ids = []
        lock = threading.Lock()

        def worker(thread_idx: int):
            for i in range(jobs_per_thread):
                job = reg.create_job(f"worker_{thread_idx}")
                with lock:
                    created_job_ids.append(job.job_id)

                # Random lifecycle transition
                action = random.choice(["running", "complete", "fail", "cancel", "none"])
                if action == "running":
                    reg.update_progress(job.job_id, random.uniform(10.0, 90.0))
                elif action == "complete":
                    reg.complete_job(job.job_id, {"done": True})
                elif action == "fail":
                    reg.fail_job(job.job_id, "random error")
                elif action == "cancel":
                    reg.cancel_job(job.job_id)

                # Concurrency check
                current_count = reg.count()
                if current_count > max_jobs:
                    violations.append((thread_idx, i, current_count))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(violations) == 0, f"Capacity bound violated! Violations: {violations}"
        assert reg.count() <= max_jobs
        with reg._lock:
            assert len(reg._jobs) <= max_jobs
            assert len(reg._subscribers) <= max_jobs

    def test_subscriber_lists_never_leak_memory_during_eviction(self):
        """CR-05: Subscriber dictionary must not accumulate lingering evicted job IDs."""
        reg = JobRegistry(max_jobs=3)

        # Create 30 jobs in sequence, transition each through various states
        for i in range(30):
            j = reg.create_job(f"job_{i}")
            if i % 3 == 0:
                reg.complete_job(j.job_id)
            elif i % 3 == 1:
                reg.fail_job(j.job_id, "fail")
            # i % 3 == 2 left pending

            # Subscribers map must NEVER exceed max_jobs
            with reg._lock:
                assert len(reg._subscribers) <= 3
                assert len(reg._jobs) <= 3
                # Keys in _subscribers must be a subset of _jobs
                for sid in reg._subscribers:
                    assert sid in reg._jobs

    def test_fifo_eviction_when_all_jobs_active(self):
        """CR-05: When all jobs are non-terminal (PENDING/RUNNING), FIFO evicts the oldest active job."""
        reg = JobRegistry(max_jobs=3)
        j1 = reg.create_job("j1")
        reg.update_progress(j1.job_id, 10.0)
        j2 = reg.create_job("j2")
        reg.update_progress(j2.job_id, 20.0)
        j3 = reg.create_job("j3")
        reg.update_progress(j3.job_id, 30.0)

        assert reg.count() == 3

        # Add 4th job -> j1 must be evicted
        j4 = reg.create_job("j4")
        assert reg.count() == 3
        assert reg.get_job(j1.job_id) is None
        assert reg.get_job(j2.job_id) is not None
        assert reg.get_job(j3.job_id) is not None
        assert reg.get_job(j4.job_id) is not None

        with reg._lock:
            assert j1.job_id not in reg._subscribers

    @pytest.mark.parametrize("invalid_max", [0, -1, -100])
    def test_job_registry_clamps_non_positive_max_jobs(self, invalid_max: int):
        """CR-05: Non-positive max_jobs must be safely clamped to at least 1."""
        reg = JobRegistry(max_jobs=invalid_max)
        assert reg._max_jobs == 1
        j1 = reg.create_job("j1")
        assert reg.count() == 1
        j2 = reg.create_job("j2")
        assert reg.count() == 1
        assert reg.get_job(j1.job_id) is None
        assert reg.get_job(j2.job_id) is not None

    def test_terminal_jobs_evicted_before_active_jobs(self):
        """CR-05: Terminal jobs (COMPLETED, FAILED, CANCELLED) must be evicted before any active job."""
        reg = JobRegistry(max_jobs=3)
        j_run1 = reg.create_job("run1")
        reg.update_progress(j_run1.job_id, 50.0)

        j_comp = reg.create_job("comp")
        reg.complete_job(j_comp.job_id)

        j_run2 = reg.create_job("run2")
        reg.update_progress(j_run2.job_id, 10.0)

        # Capacity 3 reached (run1, comp, run2). Create 4th job.
        # comp is terminal, so it MUST be evicted first even though run1 is older!
        j_new = reg.create_job("new")
        assert reg.count() == 3
        assert reg.get_job(j_comp.job_id) is None
        assert reg.get_job(j_run1.job_id) is not None
        assert reg.get_job(j_run2.job_id) is not None
        assert reg.get_job(j_new.job_id) is not None

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_unsubscribe_and_eviction(self):
        """CR-05: Concurrent subscribing, unsubscribing (via task cancellation), and rapid eviction.

        Asserts no KeyErrors or unhandled exceptions thrown.
        """
        reg = JobRegistry(max_jobs=5)
        jobs = [reg.create_job(f"job_{i}") for i in range(5)]

        subscriber_events = []
        errors = []

        async def sub_task(job_id: str):
            try:
                async for event in reg.subscribe(job_id):
                    subscriber_events.append((job_id, event))
                    # Immediately unsubscribe by exiting
                    break
            except Exception as e:
                errors.append(e)

        # Launch subscriber tasks on existing jobs
        tasks = [asyncio.create_task(sub_task(j.job_id)) for j in jobs]

        # Concurrently create 20 new jobs causing eviction
        for i in range(20):
            new_j = reg.create_job(f"evict_{i}")
            reg.complete_job(new_j.job_id, {"done": True})

        # Cancel any pending subscriber tasks
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        assert len(errors) == 0, f"Errors during concurrent subscribe/unsubscribe: {errors}"
        assert reg.count() <= 5

    @pytest.mark.asyncio
    async def test_subscribe_lock_retention_deadlock_on_terminal_job(self):
        """CR-05 ADVERSARIAL CHALLENGE: Check whether subscribe() releases self._lock when yielding.

        If subscribe() yields inside 'with self._lock:', self._lock remains held while the caller
        processes the event. This causes immediate deadlock if the caller interacts with JobRegistry.
        """
        reg = JobRegistry(max_jobs=10)
        job = reg.create_job("ingest")
        reg.complete_job(job.job_id, {"status": "ok"})

        gen = reg.subscribe(job.job_id)
        event = await gen.__anext__()

        assert event["event"] == "completed"

        # CHALLENGE: Is self._lock locked right now while caller is processing the event?
        # A properly written generator must release the mutex before yielding to consumer!
        lock_held = reg._lock.locked()
        if lock_held:
            # Clean up generator before asserting so we don't leak locks
            try:
                await gen.aclose()
            except Exception:
                pass
            pytest.fail(
                "CRITICAL CONCURRENCY BUG: JobRegistry.subscribe() yielded a terminal event "
                "from INSIDE 'with self._lock:'. self._lock is still held across the yield! "
                "Any call to reg.count(), reg.create_job(), or another subscribe() on the same "
                "thread immediately deadlocks the asyncio event loop!"
            )
