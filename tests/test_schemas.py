"""Tests for pipeline schemas."""

from hybrid_rag.schemas.pipeline import (
    CandidateResult,
    PipelineConfig,
    QueryTrace,
    StageTrace,
)


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.retrieval_strategy == "hybrid_rerank"
    assert config.alpha == 0.5
    assert config.rrf_k == 60
    assert config.rerank_enabled is True
    assert config.temperature == 0.0


def test_pipeline_config_frozen():
    config = PipelineConfig()
    try:
        config.alpha = 0.8
        assert False, "Should raise FrozenInstanceError"
    except AttributeError:
        pass


def test_query_trace_add_stage():
    trace = QueryTrace(query="test")
    stage1 = StageTrace(stage="dense", duration_ms=10.0)
    stage2 = StageTrace(stage="bm25", duration_ms=15.0)
    trace.add_stage(stage1)
    trace.add_stage(stage2)
    assert len(trace.stages) == 2
    assert trace.total_duration_ms == 25.0


def test_candidate_result():
    c = CandidateResult(chunk_id="abc", score=0.95, rank=1)
    assert c.chunk_id == "abc"
    assert c.score == 0.95
    assert c.metadata == {}
