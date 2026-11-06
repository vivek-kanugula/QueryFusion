"""Tests for citation accuracy and precision metrics."""

from hybrid_rag.eval.citation_metrics import compute_citation_metrics
from hybrid_rag.schemas.generation import (
    ClaimVerification,
    VerificationResult,
    VerificationVerdict,
)


def _make_verification(
    supported: int = 0,
    unsupported: int = 0,
    misattributed: int = 0,
) -> VerificationResult:
    claims = []
    for i in range(supported):
        claims.append(
            ClaimVerification(
                claim_text=f"s{i}",
                verdict=VerificationVerdict.SUPPORTED,
                explanation="ok",
            )
        )
    for i in range(unsupported):
        claims.append(
            ClaimVerification(
                claim_text=f"u{i}",
                verdict=VerificationVerdict.UNSUPPORTED,
                explanation="no",
            )
        )
    for i in range(misattributed):
        claims.append(
            ClaimVerification(
                claim_text=f"m{i}",
                verdict=VerificationVerdict.MISATTRIBUTED,
                explanation="wrong",
            )
        )
    total = len(claims)
    cov = supported / total if total > 0 else 0.0
    return VerificationResult(
        claims=claims,
        citation_coverage=cov,
        all_supported=(total > 0 and supported == total),
    )


# ── None / empty ──


def test_none_verification():
    scores = compute_citation_metrics(None)
    assert scores.citation_accuracy == 0.0
    assert scores.citation_precision == 0.0
    assert scores.total_claims == 0


def test_empty_claims():
    v = VerificationResult(claims=[], citation_coverage=0.0)
    scores = compute_citation_metrics(v)
    assert scores.citation_accuracy == 0.0
    assert scores.total_claims == 0


# ── All supported ──


def test_all_supported():
    v = _make_verification(supported=4)
    scores = compute_citation_metrics(v)
    assert scores.citation_accuracy == 1.0
    assert scores.citation_precision == 1.0
    assert scores.supported == 4
    assert scores.unsupported == 0
    assert scores.misattributed == 0


# ── Mixed verdicts ──


def test_mixed_verdicts():
    v = _make_verification(supported=2, unsupported=1, misattributed=1)
    scores = compute_citation_metrics(v)
    assert abs(scores.citation_accuracy - 0.5) < 1e-9
    assert abs(scores.citation_precision - 0.5) < 1e-9
    assert scores.total_claims == 4


def test_all_unsupported():
    v = _make_verification(unsupported=3)
    scores = compute_citation_metrics(v)
    assert scores.citation_accuracy == 0.0
    assert scores.citation_precision == 0.0
    assert scores.unsupported == 3


def test_all_misattributed():
    v = _make_verification(misattributed=2)
    scores = compute_citation_metrics(v)
    assert scores.citation_accuracy == 0.0
    assert scores.citation_precision == 0.0
    assert scores.misattributed == 2


def test_one_supported_one_misattributed():
    v = _make_verification(supported=1, misattributed=1)
    scores = compute_citation_metrics(v)
    assert abs(scores.citation_accuracy - 0.5) < 1e-9
    assert abs(scores.citation_precision - 0.5) < 1e-9


def test_single_supported():
    v = _make_verification(supported=1)
    scores = compute_citation_metrics(v)
    assert scores.citation_accuracy == 1.0
    assert scores.citation_precision == 1.0
    assert scores.total_claims == 1


def test_precision_equals_accuracy():
    """accuracy = S/total, precision = S/(S+U+M) = S/total. Same formula."""
    v = _make_verification(supported=3, unsupported=2)
    scores = compute_citation_metrics(v)
    assert abs(scores.citation_accuracy - scores.citation_precision) < 1e-9
