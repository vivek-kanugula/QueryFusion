"""Citation accuracy metrics derived from VerificationResult."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_rag.schemas.generation import VerificationResult, VerificationVerdict


@dataclass(frozen=True)
class CitationScores:
    citation_accuracy: float
    citation_precision: float
    total_claims: int
    supported: int
    unsupported: int
    misattributed: int


def compute_citation_metrics(
    verification: VerificationResult | None,
) -> CitationScores:
    if verification is None or not verification.claims:
        return CitationScores(
            citation_accuracy=0.0,
            citation_precision=0.0,
            total_claims=0,
            supported=0,
            unsupported=0,
            misattributed=0,
        )

    supported = sum(1 for c in verification.claims if c.verdict == VerificationVerdict.SUPPORTED)
    unsupported = sum(
        1 for c in verification.claims if c.verdict == VerificationVerdict.UNSUPPORTED
    )
    misattributed = sum(
        1 for c in verification.claims if c.verdict == VerificationVerdict.MISATTRIBUTED
    )
    total = len(verification.claims)

    accuracy = supported / total if total > 0 else 0.0

    precision_denom = supported + misattributed
    precision = supported / precision_denom if precision_denom > 0 else 0.0

    return CitationScores(
        citation_accuracy=accuracy,
        citation_precision=precision,
        total_claims=total,
        supported=supported,
        unsupported=unsupported,
        misattributed=misattributed,
    )
