"""Bootstrap confidence intervals, significance tests, per-category aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    confidence: float


def bootstrap_ci(
    scores: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 9999,
    seed: int = 42,
) -> ConfidenceInterval:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return ConfidenceInterval(mean=0.0, lower=0.0, upper=0.0, confidence=confidence)
    if len(scores) == 1:
        val = float(scores[0])
        return ConfidenceInterval(mean=val, lower=val, upper=val, confidence=confidence)

    result = sp_stats.bootstrap(
        (scores,),
        statistic=np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence,
        random_state=np.random.default_rng(seed),
        method="percentile",
    )
    return ConfidenceInterval(
        mean=float(np.mean(scores)),
        lower=float(result.confidence_interval.low),
        upper=float(result.confidence_interval.high),
        confidence=confidence,
    )


def paired_significance(scores_a: np.ndarray, scores_b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Arrays must have equal length for paired test")
    if len(a) < 2:
        return (0.0, 1.0)
    diff = a - b
    if np.all(diff == 0):
        return (0.0, 1.0)
    result = sp_stats.wilcoxon(diff, alternative="two-sided")
    return (float(result.statistic), float(result.pvalue))


def per_category_summary(
    category_scores: dict[str, list[float]],
    confidence: float = 0.95,
) -> dict[str, ConfidenceInterval]:
    out: dict[str, ConfidenceInterval] = {}
    for category, scores in category_scores.items():
        out[category] = bootstrap_ci(np.array(scores), confidence=confidence)
    return out


def win_loss_tie(
    scores_a: np.ndarray, scores_b: np.ndarray, tolerance: float = 1e-6
) -> dict[str, int]:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b
    wins = int(np.sum(diff > tolerance))
    losses = int(np.sum(diff < -tolerance))
    ties = int(np.sum(np.abs(diff) <= tolerance))
    return {"wins": wins, "losses": losses, "ties": ties}
