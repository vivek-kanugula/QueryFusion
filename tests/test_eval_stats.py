"""Tests for statistical utilities: bootstrap CI, Wilcoxon, per-category, win/loss/tie."""

import numpy as np
import pytest

from hybrid_rag.eval.stats import (
    bootstrap_ci,
    paired_significance,
    per_category_summary,
    win_loss_tie,
)

# ── Bootstrap CI ──


def test_bootstrap_ci_constant_array():
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    ci = bootstrap_ci(scores)
    assert ci.mean == pytest.approx(0.5)
    assert ci.lower == pytest.approx(0.5)
    assert ci.upper == pytest.approx(0.5)
    assert ci.confidence == 0.95


def test_bootstrap_ci_known_range():
    rng = np.random.default_rng(123)
    scores = rng.normal(0.7, 0.1, size=100)
    ci = bootstrap_ci(scores, confidence=0.95)
    assert 0.65 < ci.lower < ci.mean < ci.upper < 0.75
    assert ci.confidence == 0.95


def test_bootstrap_ci_single_element():
    ci = bootstrap_ci(np.array([0.42]))
    assert ci.mean == pytest.approx(0.42)
    assert ci.lower == pytest.approx(0.42)
    assert ci.upper == pytest.approx(0.42)


def test_bootstrap_ci_empty():
    ci = bootstrap_ci(np.array([]))
    assert ci.mean == 0.0


def test_bootstrap_ci_deterministic():
    scores = np.array([0.1, 0.5, 0.8, 0.3, 0.9])
    ci1 = bootstrap_ci(scores, seed=42)
    ci2 = bootstrap_ci(scores, seed=42)
    assert ci1.lower == ci2.lower
    assert ci1.upper == ci2.upper


# ── Paired significance (Wilcoxon) ──


def test_wilcoxon_identical_arrays():
    a = np.array([0.5, 0.6, 0.7])
    stat, p = paired_significance(a, a)
    assert p == pytest.approx(1.0)


def test_wilcoxon_clearly_different():
    a = np.array([0.9, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86, 0.89])
    b = np.array([0.3, 0.25, 0.28, 0.32, 0.27, 0.31, 0.26, 0.29])
    stat, p = paired_significance(a, b)
    assert p < 0.05


def test_wilcoxon_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        paired_significance(np.array([1, 2]), np.array([1, 2, 3]))


def test_wilcoxon_single_pair():
    stat, p = paired_significance(np.array([0.5]), np.array([0.3]))
    assert p == 1.0


# ── Per-category summary ──


def test_per_category_all_categories():
    cat_scores = {
        "exact": [0.8, 0.9, 0.85],
        "factual": [0.7, 0.6, 0.65],
    }
    summary = per_category_summary(cat_scores)
    assert "exact" in summary
    assert "factual" in summary
    assert summary["exact"].mean > summary["factual"].mean


def test_per_category_single_value():
    summary = per_category_summary({"cat": [0.5]})
    assert summary["cat"].mean == pytest.approx(0.5)


# ── Win / loss / tie ──


def test_win_loss_tie_clear():
    a = np.array([0.9, 0.8, 0.7])
    b = np.array([0.3, 0.4, 0.5])
    wlt = win_loss_tie(a, b)
    assert wlt == {"wins": 3, "losses": 0, "ties": 0}


def test_win_loss_tie_mixed():
    a = np.array([0.9, 0.3, 0.5])
    b = np.array([0.3, 0.8, 0.5])
    wlt = win_loss_tie(a, b)
    assert wlt == {"wins": 1, "losses": 1, "ties": 1}


def test_win_loss_tie_all_tied():
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    wlt = win_loss_tie(a, b)
    assert wlt == {"wins": 0, "losses": 0, "ties": 2}


def test_win_loss_tie_tolerance():
    a = np.array([0.500001])
    b = np.array([0.500000])
    wlt = win_loss_tie(a, b, tolerance=1e-4)
    assert wlt["ties"] == 1
