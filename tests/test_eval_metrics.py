"""Tests for IR metrics with hand-computed reference values."""

import math

import pytest

from hybrid_rag.eval.metrics import (
    compute_all,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ── Precision@K ──


def test_precision_perfect():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)


def test_precision_none_relevant():
    assert precision_at_k(["x", "y", "z"], {"a", "b"}, 3) == pytest.approx(0.0)


def test_precision_partial():
    assert precision_at_k(["a", "x", "b", "y", "c"], {"a", "b", "c"}, 3) == pytest.approx(2 / 3)


def test_precision_k_larger_than_retrieved():
    assert precision_at_k(["a", "b"], {"a", "b", "c"}, 5) == pytest.approx(2 / 5)


def test_precision_k_zero():
    assert precision_at_k(["a"], {"a"}, 0) == 0.0


def test_precision_empty_retrieved():
    assert precision_at_k([], {"a"}, 3) == 0.0


def test_precision_empty_relevant():
    assert precision_at_k(["a", "b"], set(), 2) == 0.0


# ── Recall@K ──


def test_recall_perfect():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)


def test_recall_none_found():
    assert recall_at_k(["x", "y"], {"a", "b"}, 2) == pytest.approx(0.0)


def test_recall_partial():
    assert recall_at_k(["a", "x", "c"], {"a", "b", "c"}, 3) == pytest.approx(2 / 3)


def test_recall_empty_relevant():
    assert recall_at_k(["a"], set(), 1) == 0.0


def test_recall_k_zero():
    assert recall_at_k(["a"], {"a"}, 0) == 0.0


def test_recall_all_relevant_in_top_k():
    assert recall_at_k(["a", "b", "x", "y"], {"a", "b"}, 4) == pytest.approx(1.0)


# ── MRR ──


def test_mrr_first_position():
    assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_mrr_second_position():
    assert mrr(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)


def test_mrr_third_position():
    assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_mrr_not_found():
    assert mrr(["x", "y", "z"], {"a"}) == 0.0


def test_mrr_empty_retrieved():
    assert mrr([], {"a"}) == 0.0


def test_mrr_multiple_relevant():
    assert mrr(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)


# ── NDCG@K ──


def test_ndcg_perfect_ranking():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)


def test_ndcg_worst_ranking():
    """3 relevant items all at bottom of top 6."""
    retrieved = ["x", "y", "z", "a", "b", "c"]
    relevant = {"a", "b", "c"}
    dcg = 1 / math.log2(5) + 1 / math.log2(6) + 1 / math.log2(7)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(retrieved, relevant, 6) == pytest.approx(dcg / idcg)


def test_ndcg_hand_computed_example_1():
    """Retrieved: [A,B,C,D,E], Relevant: {A,C,E}, k=3
    DCG@3 = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0 + 0.5 = 1.5
    IDCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.6309 + 0.5 = 2.1309
    """
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(["A", "B", "C", "D", "E"], {"A", "C", "E"}, 3) == pytest.approx(dcg / idcg)


def test_ndcg_hand_computed_example_2():
    """Retrieved: [B,D,A,C,E], Relevant: {A,C}, k=5
    DCG@5 = 0 + 0 + 1/log2(4) + 1/log2(5) + 0 = 0.5 + 0.4307 = 0.9307
    IDCG@2 = 1/log2(2) + 1/log2(3) = 1.6309
    """
    dcg = 1 / math.log2(4) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["B", "D", "A", "C", "E"], {"A", "C"}, 5) == pytest.approx(dcg / idcg)


def test_ndcg_no_relevant():
    assert ndcg_at_k(["a", "b"], set(), 2) == 0.0


def test_ndcg_k_zero():
    assert ndcg_at_k(["a"], {"a"}, 0) == 0.0


def test_ndcg_single_relevant_at_rank_1():
    assert ndcg_at_k(["a", "b", "c"], {"a"}, 3) == pytest.approx(1.0)


def test_ndcg_single_relevant_at_rank_3():
    dcg = 1 / math.log2(4)
    idcg = 1 / math.log2(2)
    assert ndcg_at_k(["x", "y", "a"], {"a"}, 3) == pytest.approx(dcg / idcg)


# ── compute_all ──


def test_compute_all_returns_all_four():
    result = compute_all(["a", "b"], {"a"}, 2)
    assert set(result.keys()) == {"precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"}
    assert result["precision_at_k"] == pytest.approx(0.5)
    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)


def test_compute_all_empty():
    result = compute_all([], set(), 5)
    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0
    assert result["mrr"] == 0.0
    assert result["ndcg_at_k"] == 0.0


# ── Duplicate handling ──


def test_precision_with_duplicates_in_retrieved():
    assert precision_at_k(["a", "a", "b"], {"a", "b"}, 3) == pytest.approx(2 / 3)


def test_ndcg_monotonic_with_rank():
    """Better ranking = higher NDCG."""
    good = ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3)
    bad = ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3)
    assert good > bad
