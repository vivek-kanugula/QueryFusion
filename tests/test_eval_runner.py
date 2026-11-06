"""Tests for experiment runner with mock retrieval."""

from hybrid_rag.eval.dataset import ChunkInfo, GoldenItem, SourceSpan, dataset_hash
from hybrid_rag.eval.runner import (
    compare_experiments,
    config_hash,
    run_experiment,
    run_single_question,
)
from hybrid_rag.schemas.pipeline import PipelineConfig

# ── Test fixtures ──


def _make_chunks() -> list[ChunkInfo]:
    return [
        ChunkInfo(chunk_id="c1", document_id="doc-a", char_offset_start=0, char_offset_end=200),
        ChunkInfo(chunk_id="c2", document_id="doc-a", char_offset_start=200, char_offset_end=400),
        ChunkInfo(chunk_id="c3", document_id="doc-a", char_offset_start=400, char_offset_end=600),
    ]


def _make_dataset() -> list[GoldenItem]:
    return [
        GoldenItem(
            id="q1",
            question="What is X?",
            expected_answer="X is ...",
            category="factual",
            source_spans=[SourceSpan("doc-a", 50, 150)],
        ),
        GoldenItem(
            id="q2",
            question="What is Y?",
            expected_answer="Y is ...",
            category="factual",
            source_spans=[SourceSpan("doc-a", 250, 350)],
        ),
        GoldenItem(
            id="q3",
            question="Unrelated?",
            expected_answer="Not found.",
            category="unanswerable",
            source_spans=[],
        ),
    ]


# ── Config hash ──


def test_config_hash_deterministic():
    config = PipelineConfig()
    assert config_hash(config) == config_hash(config)


def test_config_hash_different_alpha():
    c1 = PipelineConfig(alpha=0.5)
    c2 = PipelineConfig(alpha=0.7)
    assert config_hash(c1) != config_hash(c2)


# ── Single question evaluation ──


def test_single_question_perfect_retrieval():
    chunks = _make_chunks()

    def retrieve_fn(query, config):
        return ["c1"]

    item = GoldenItem(
        id="q1",
        question="What?",
        expected_answer="A",
        category="factual",
        source_spans=[SourceSpan("doc-a", 50, 150)],
    )
    result = run_single_question(item, PipelineConfig(), retrieve_fn, chunks, k=5)
    assert result.question_id == "q1"
    assert result.metrics["precision_at_k"] > 0
    assert "c1" in result.relevant_ids


def test_single_question_unanswerable():
    def retrieve_fn(query, config):
        return ["c1", "c2"]

    item = GoldenItem(
        id="q3",
        question="Unrelated?",
        expected_answer="N/A",
        category="unanswerable",
        source_spans=[],
    )
    result = run_single_question(item, PipelineConfig(), retrieve_fn, _make_chunks(), k=5)
    assert result.relevant_ids == set()
    assert result.metrics["recall_at_k"] == 0.0


def test_single_question_latency_recorded():
    def retrieve_fn(query, config):
        return ["c1"]

    item = _make_dataset()[0]
    result = run_single_question(item, PipelineConfig(), retrieve_fn, _make_chunks(), k=5)
    assert result.latency_ms >= 0


# ── Full experiment ──


def test_run_experiment():
    def retrieve_fn(query, config):
        return ["c1", "c2", "c3"]

    dataset = _make_dataset()
    ds_hash = dataset_hash(dataset)
    result = run_experiment(
        name="test_exp",
        config=PipelineConfig(),
        dataset=dataset,
        retrieve_fn=retrieve_fn,
        chunks=_make_chunks(),
        dataset_hash_value=ds_hash,
        corpus_snapshot_id="snap-001",
    )

    assert result.name == "test_exp"
    assert len(result.per_question) == 3
    assert "precision_at_k" in result.overall_metrics
    assert "ndcg_at_k" in result.overall_metrics
    assert "latency_ms" in result.overall_metrics
    assert "factual" in result.per_category
    assert "unanswerable" in result.per_category


def test_experiment_manifest():
    def retrieve_fn(query, config):
        return []

    dataset = _make_dataset()
    result = run_experiment(
        name="manifest_test",
        config=PipelineConfig(alpha=0.6),
        dataset=dataset,
        retrieve_fn=retrieve_fn,
        chunks=_make_chunks(),
        dataset_hash_value="abc123",
        corpus_snapshot_id="snap-002",
        model_names={"embedder": "bge-small", "reranker": "MiniLM-L-6-v2"},
    )
    m = result.manifest
    assert m.dataset_hash == "abc123"
    assert m.corpus_snapshot_id == "snap-002"
    assert m.model_names["embedder"] == "bge-small"
    assert m.config_hash == config_hash(PipelineConfig(alpha=0.6))


def test_experiment_to_dict():
    def retrieve_fn(query, config):
        return ["c1"]

    result = run_experiment(
        name="dict_test",
        config=PipelineConfig(),
        dataset=_make_dataset()[:1],
        retrieve_fn=retrieve_fn,
        chunks=_make_chunks(),
        dataset_hash_value="x",
    )
    d = result.to_dict()
    assert d["name"] == "dict_test"
    assert "manifest" in d
    assert d["n_questions"] == 1


# ── Comparative experiments ──


def test_compare_experiments():
    def good_retrieve(query, config):
        return ["c1", "c2"]

    def bad_retrieve(query, config):
        return ["c3"]

    dataset = _make_dataset()[:2]
    chunks = _make_chunks()
    ds_hash = dataset_hash(dataset)

    result_a = run_experiment("good", PipelineConfig(), dataset, good_retrieve, chunks, ds_hash)
    result_b = run_experiment("bad", PipelineConfig(), dataset, bad_retrieve, chunks, ds_hash)

    comparison = compare_experiments(result_a, result_b, metric="precision_at_k")
    assert comparison["experiment_a"] == "good"
    assert comparison["experiment_b"] == "bad"
    assert "delta_mean" in comparison
    assert "wilcoxon_p_value" in comparison
    assert "wins_losses_ties" in comparison
