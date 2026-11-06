"""Tests for trace store: save, get, missing ID, eviction."""

from hybrid_rag.core.trace_store import TraceStore
from hybrid_rag.schemas.pipeline import CandidateResult, QueryTrace, StageTrace


def _make_trace(trace_id: str = "t1") -> QueryTrace:
    return QueryTrace(
        trace_id=trace_id,
        query="test query",
        stages=[
            StageTrace(
                stage="dense",
                candidates=[
                    CandidateResult(
                        chunk_id="c1",
                        score=0.9,
                        rank=1,
                        content="text",
                        metadata={},
                    ),
                ],
                duration_ms=10.0,
            ),
        ],
    )


def test_save_and_get():
    store = TraceStore(max_capacity=10)
    trace = _make_trace("t1")
    store.save(trace)
    fetched = store.get("t1")
    assert fetched is not None
    assert fetched.trace_id == "t1"
    assert fetched.query == "test query"


def test_get_missing():
    store = TraceStore()
    assert store.get("nonexistent") is None


def test_count():
    store = TraceStore()
    assert store.count() == 0
    store.save(_make_trace("t1"))
    store.save(_make_trace("t2"))
    assert store.count() == 2


def test_clear():
    store = TraceStore()
    store.save(_make_trace("t1"))
    store.save(_make_trace("t2"))
    store.clear()
    assert store.count() == 0
    assert store.get("t1") is None


def test_eviction_at_capacity():
    store = TraceStore(max_capacity=3)
    store.save(_make_trace("t1"))
    store.save(_make_trace("t2"))
    store.save(_make_trace("t3"))
    assert store.count() == 3

    store.save(_make_trace("t4"))
    assert store.count() == 3
    assert store.get("t1") is None
    assert store.get("t4") is not None


def test_overwrite_existing():
    store = TraceStore()
    t1 = _make_trace("t1")
    t1.query = "original"
    store.save(t1)

    t1_updated = _make_trace("t1")
    t1_updated.query = "updated"
    store.save(t1_updated)

    fetched = store.get("t1")
    assert fetched.query == "updated"
    assert store.count() == 1


def test_lru_access_prevents_eviction():
    store = TraceStore(max_capacity=3)
    store.save(_make_trace("t1"))
    store.save(_make_trace("t2"))
    store.save(_make_trace("t3"))

    store.get("t1")

    store.save(_make_trace("t4"))
    assert store.get("t1") is not None
    assert store.get("t2") is None


def test_trace_with_stages():
    store = TraceStore()
    trace = _make_trace("t1")
    store.save(trace)
    fetched = store.get("t1")
    assert len(fetched.stages) == 1
    assert fetched.stages[0].stage == "dense"
    assert len(fetched.stages[0].candidates) == 1
