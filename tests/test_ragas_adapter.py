"""Tests for RAGAS offline adapter: NaN handling, aggregation, embeddings wrapper."""

import math

from hybrid_rag.eval.ragas_eval import (
    LocalEmbeddingsWrapper,
    RagasScores,
    _is_nan,
    aggregate_scores,
)

# ── NaN detection ──


def test_is_nan_none():
    assert _is_nan(None) is True


def test_is_nan_float_nan():
    assert _is_nan(float("nan")) is True


def test_is_nan_math_nan():
    assert _is_nan(math.nan) is True


def test_is_nan_valid_float():
    assert _is_nan(0.85) is False


def test_is_nan_zero():
    assert _is_nan(0.0) is False


def test_is_nan_string():
    assert _is_nan("not a number") is True


def test_is_nan_int():
    assert _is_nan(1) is False


# ── Aggregation ──


def test_aggregate_all_valid():
    per_q = [
        {"faithfulness": 0.9, "answer_relevancy": 0.8},
        {"faithfulness": 0.7, "answer_relevancy": 0.6},
    ]
    scores = aggregate_scores(per_q)
    assert abs(scores.faithfulness - 0.8) < 1e-9
    assert abs(scores.answer_relevancy - 0.7) < 1e-9
    assert scores.nan_count == 0
    assert scores.total_count == 2
    assert scores.nan_rate == 0.0


def test_aggregate_with_nans():
    per_q = [
        {"faithfulness": 0.9, "answer_relevancy": None},
        {"faithfulness": None, "answer_relevancy": 0.6},
        {"faithfulness": 0.7, "answer_relevancy": 0.8},
    ]
    scores = aggregate_scores(per_q)
    assert abs(scores.faithfulness - 0.8) < 1e-9
    assert abs(scores.answer_relevancy - 0.7) < 1e-9
    assert scores.nan_count == 2
    assert scores.total_count == 3
    assert abs(scores.nan_rate - 2 / 3) < 1e-9


def test_aggregate_all_nans():
    per_q = [
        {"faithfulness": None, "answer_relevancy": None},
        {"faithfulness": float("nan"), "answer_relevancy": None},
    ]
    scores = aggregate_scores(per_q)
    assert scores.faithfulness is None
    assert scores.answer_relevancy is None
    assert scores.nan_count == 2
    assert scores.nan_rate == 1.0


def test_aggregate_empty():
    scores = aggregate_scores([])
    assert scores.faithfulness is None
    assert scores.answer_relevancy is None
    assert scores.nan_count == 0
    assert scores.total_count == 0
    assert scores.nan_rate == 0.0


def test_aggregate_single_item():
    per_q = [{"faithfulness": 0.5, "answer_relevancy": 0.9}]
    scores = aggregate_scores(per_q)
    assert abs(scores.faithfulness - 0.5) < 1e-9
    assert abs(scores.answer_relevancy - 0.9) < 1e-9
    assert scores.nan_count == 0


def test_aggregate_partial_nan_one_metric():
    per_q = [
        {"faithfulness": 0.8, "answer_relevancy": 0.7},
        {"faithfulness": 0.6, "answer_relevancy": None},
    ]
    scores = aggregate_scores(per_q)
    assert abs(scores.faithfulness - 0.7) < 1e-9
    assert abs(scores.answer_relevancy - 0.7) < 1e-9
    assert scores.nan_count == 1


# ── RagasScores ──


def test_ragas_scores_nan_rate():
    s = RagasScores(nan_count=3, total_count=10)
    assert abs(s.nan_rate - 0.3) < 1e-9


def test_ragas_scores_nan_rate_zero_total():
    s = RagasScores(nan_count=0, total_count=0)
    assert s.nan_rate == 0.0


# ── LocalEmbeddingsWrapper ──


def test_local_embeddings_wrapper_init():
    wrapper = LocalEmbeddingsWrapper(model_name="BAAI/bge-small-en-v1.5")
    assert wrapper._model_name == "BAAI/bge-small-en-v1.5"
    assert wrapper._model is None
