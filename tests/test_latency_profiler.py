"""Tests for latency profiler: percentiles, stage aggregation, warmup exclusion."""

from hybrid_rag.eval.latency import LatencyProfiler, StageLatencies

# ── StageLatencies ──


def test_stage_latencies_empty():
    sl = StageLatencies(stage="test")
    s = sl.stats()
    assert s.count == 0
    assert s.p50 == 0.0
    assert s.mean == 0.0


def test_stage_latencies_single():
    sl = StageLatencies(stage="test", durations_ms=[42.0])
    s = sl.stats()
    assert s.count == 1
    assert s.p50 == 42.0
    assert s.mean == 42.0
    assert s.min == 42.0
    assert s.max == 42.0


def test_stage_latencies_percentiles():
    durations = list(range(1, 101))
    sl = StageLatencies(stage="test", durations_ms=[float(d) for d in durations])
    s = sl.stats()
    assert s.count == 100
    assert abs(s.p50 - 50.5) < 1.0
    assert s.p90 >= 90.0
    assert s.p95 >= 95.0
    assert s.min == 1.0
    assert s.max == 100.0
    assert abs(s.mean - 50.5) < 0.01


def test_stage_latencies_known_percentiles():
    durations = [10.0, 20.0, 30.0, 40.0, 50.0]
    sl = StageLatencies(stage="test", durations_ms=durations)
    s = sl.stats()
    assert s.min == 10.0
    assert s.max == 50.0
    assert abs(s.mean - 30.0) < 1e-9
    assert s.p50 == 30.0


# ── LatencyProfiler ──


def test_profiler_default_stages():
    profiler = LatencyProfiler()
    assert "query_embedding" in profiler.stages
    assert "vector_search" in profiler.stages
    assert "cross_encoder_rerank" in profiler.stages
    assert "generation" in profiler.stages
    assert "verification" in profiler.stages


def test_profiler_record():
    profiler = LatencyProfiler()
    profiler.record("generation", 100.0)
    profiler.record("generation", 120.0)
    s = profiler.stages["generation"].stats()
    assert s.count == 2
    assert abs(s.mean - 110.0) < 1e-9


def test_profiler_record_custom_stage():
    profiler = LatencyProfiler()
    profiler.record("custom_stage", 50.0)
    assert "custom_stage" in profiler.stages
    assert profiler.stages["custom_stage"].stats().count == 1


def test_profiler_warmup_excluded():
    profiler = LatencyProfiler()
    profiler.record_from_trace(
        {"generation": 500.0, "verification": 200.0},
        is_warmup=True,
    )
    assert profiler.warmup_runs == 1
    s = profiler.summary()
    assert "generation" not in s
    assert "verification" not in s


def test_profiler_warmup_then_real():
    profiler = LatencyProfiler()
    profiler.record_from_trace({"generation": 500.0}, is_warmup=True)
    profiler.record_from_trace({"generation": 100.0}, is_warmup=False)
    profiler.record_from_trace({"generation": 110.0}, is_warmup=False)
    assert profiler.warmup_runs == 1
    s = profiler.summary()
    assert "generation" in s
    assert s["generation"].count == 2
    assert abs(s["generation"].mean - 105.0) < 1e-9


def test_profiler_record_from_trace():
    profiler = LatencyProfiler()
    profiler.record_from_trace(
        {
            "query_embedding": 5.0,
            "vector_search": 20.0,
            "generation": 100.0,
        }
    )
    summary = profiler.summary()
    assert summary["query_embedding"].count == 1
    assert summary["vector_search"].count == 1
    assert summary["generation"].count == 1


def test_profiler_summary_excludes_empty():
    profiler = LatencyProfiler()
    profiler.record("generation", 100.0)
    summary = profiler.summary()
    assert "generation" in summary
    assert "vector_search" not in summary


def test_profiler_total_latency():
    profiler = LatencyProfiler()
    for _ in range(3):
        profiler.record("generation", 100.0)
        profiler.record("verification", 50.0)
    total = profiler.total_latency_stats()
    assert total.count == 3
    assert abs(total.mean - 150.0) < 1e-9


def test_profiler_to_dict():
    profiler = LatencyProfiler(environment="hf-spaces")
    profiler.record("generation", 100.0)
    d = profiler.to_dict()
    assert d["environment"] == "hf-spaces"
    assert "generation" in d["stages"]
    assert d["stages"]["generation"]["count"] == 1


def test_profiler_to_dict_warmup():
    profiler = LatencyProfiler()
    profiler.record_from_trace({"generation": 500.0}, is_warmup=True)
    profiler.record_from_trace({"generation": 100.0})
    d = profiler.to_dict()
    assert d["warmup_runs_excluded"] == 1
