"""Composite confidence scoring: 0.4*Retrieval + 0.4*Citation + 0.2*Completeness."""

from __future__ import annotations

from hybrid_rag.schemas.generation import CompositeConfidence, GroundedAnswer, VerificationResult
from hybrid_rag.schemas.pipeline import QueryTrace

WEIGHT_RETRIEVAL = 0.4
WEIGHT_CITATION = 0.4
WEIGHT_COMPLETENESS = 0.2


def retrieval_score(trace: QueryTrace) -> float:
    if not trace.stages:
        return 0.0
    last_stage = trace.stages[-1]
    if not last_stage.candidates:
        return 0.0
    top_score = max(c.score for c in last_stage.candidates)
    if last_stage.stage == "rrf":
        rrf_k = (
            trace.config.rrf_k
            if (trace.config and hasattr(trace.config, "rrf_k") and trace.config.rrf_k > 0)
            else 60
        )
        if top_score <= 1.0 / rrf_k:
            top_score = top_score * (rrf_k + 1)
    return max(0.0, min(1.0, top_score))


def citation_coverage_score(verification: VerificationResult | None) -> float:
    if verification is None:
        return 0.0
    return max(0.0, min(1.0, verification.citation_coverage))


def completeness_score(answer: GroundedAnswer) -> float:
    if not answer.has_sufficient_context:
        return 0.0
    if not answer.citations:
        return 0.0
    return 1.0


def compute_confidence(
    answer: GroundedAnswer,
    verification: VerificationResult | None,
    trace: QueryTrace,
) -> CompositeConfidence:
    r_score = retrieval_score(trace)
    c_score = citation_coverage_score(verification)
    comp_score = completeness_score(answer)

    overall = (
        WEIGHT_RETRIEVAL * r_score + WEIGHT_CITATION * c_score + WEIGHT_COMPLETENESS * comp_score
    )
    overall = max(0.0, min(1.0, overall))

    return CompositeConfidence(
        overall_score=overall,
        retrieval_score=r_score,
        citation_coverage=c_score,
        completeness_score=comp_score,
        components_breakdown={
            "retrieval_weighted": WEIGHT_RETRIEVAL * r_score,
            "citation_weighted": WEIGHT_CITATION * c_score,
            "completeness_weighted": WEIGHT_COMPLETENESS * comp_score,
        },
    )
