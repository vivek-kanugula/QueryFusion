import pytest

from hybrid_rag.generation.confidence import (
    WEIGHT_CITATION,
    WEIGHT_COMPLETENESS,
    WEIGHT_RETRIEVAL,
    citation_coverage_score,
    completeness_score,
    compute_confidence,
    retrieval_score,
)
from hybrid_rag.schemas.generation import (
    Citation,
    ClaimVerification,
    GroundedAnswer,
    VerificationResult,
    VerificationVerdict,
)
from hybrid_rag.schemas.pipeline import CandidateResult, QueryTrace, StageTrace


def _trace_with_score(top_score: float) -> QueryTrace:
    return QueryTrace(
        trace_id="t1",
        stages=[
            StageTrace(
                stage="rerank",
                candidates=[
                    CandidateResult(
                        chunk_id="c1",
                        score=top_score,
                        rank=1,
                        content="text",
                        metadata={},
                    ),
                ],
            ),
        ],
    )


def _grounded(sufficient: bool = True, has_citations: bool = True) -> GroundedAnswer:
    citations = [Citation(citation_id="[C1]", chunk_id="c1")] if has_citations else []
    return GroundedAnswer(
        answer="Answer [C1]." if has_citations else "Cannot answer.",
        citations=citations,
        has_sufficient_context=sufficient,
        fallback_reason=None if sufficient else "No context.",
    )


def _verification(supported: int, unsupported: int, misattributed: int = 0) -> VerificationResult:
    claims = []
    for i in range(supported):
        claims.append(
            ClaimVerification(
                claim_text=f"claim-s{i}",
                verdict=VerificationVerdict.SUPPORTED,
                explanation="ok",
            )
        )
    for i in range(unsupported):
        claims.append(
            ClaimVerification(
                claim_text=f"claim-u{i}",
                verdict=VerificationVerdict.UNSUPPORTED,
                explanation="no",
            )
        )
    for i in range(misattributed):
        claims.append(
            ClaimVerification(
                claim_text=f"claim-m{i}",
                verdict=VerificationVerdict.MISATTRIBUTED,
                explanation="wrong",
            )
        )
    total = len(claims)
    coverage = supported / total if total > 0 else 0.0
    all_sup = total > 0 and supported == total
    return VerificationResult(claims=claims, citation_coverage=coverage, all_supported=all_sup)


# ── Component functions ──


def test_retrieval_score_empty_trace():
    trace = QueryTrace(trace_id="t", stages=[])
    assert retrieval_score(trace) == 0.0


def test_retrieval_score_empty_candidates():
    trace = QueryTrace(trace_id="t", stages=[StageTrace(stage="s", candidates=[])])
    assert retrieval_score(trace) == 0.0


def test_retrieval_score_clamped():
    assert retrieval_score(_trace_with_score(1.5)) == 1.0
    assert retrieval_score(_trace_with_score(-0.3)) == 0.0


def test_retrieval_score_normal():
    assert retrieval_score(_trace_with_score(0.85)) == 0.85


def test_citation_coverage_none():
    assert citation_coverage_score(None) == 0.0


def test_citation_coverage_normal():
    v = _verification(3, 1)
    assert citation_coverage_score(v) == 0.75


def test_completeness_insufficient():
    assert completeness_score(_grounded(sufficient=False)) == 0.0


def test_completeness_no_citations():
    assert completeness_score(_grounded(sufficient=True, has_citations=False)) == 0.0


def test_completeness_full():
    assert completeness_score(_grounded(sufficient=True, has_citations=True)) == 1.0


# ── Hand-calculated scenarios ──


def test_perfect_grounding():
    """R=0.95, 3 SUPPORTED (cov=1.0), sufficient+citations (comp=1.0).
    Expected: 0.4*0.95 + 0.4*1.0 + 0.2*1.0 = 0.38 + 0.40 + 0.20 = 0.98
    """
    trace = _trace_with_score(0.95)
    answer = _grounded(sufficient=True, has_citations=True)
    verification = _verification(supported=3, unsupported=0)
    conf = compute_confidence(answer, verification, trace)

    assert abs(conf.retrieval_score - 0.95) < 1e-9
    assert abs(conf.citation_coverage - 1.0) < 1e-9
    assert abs(conf.completeness_score - 1.0) < 1e-9
    assert abs(conf.overall_score - 0.98) < 1e-9


def test_partial_misattribution():
    """Retrieval=0.80, 2 SUPPORTED + 1 MISATTRIBUTED (coverage=2/3), sufficient+citations.
    Expected: 0.4*0.80 + 0.4*(2/3) + 0.2*1.0 = 0.32 + 0.2667 + 0.20 = 0.7867
    """
    trace = _trace_with_score(0.80)
    answer = _grounded(sufficient=True, has_citations=True)
    verification = _verification(supported=2, unsupported=0, misattributed=1)
    conf = compute_confidence(answer, verification, trace)

    assert abs(conf.retrieval_score - 0.80) < 1e-9
    assert abs(conf.citation_coverage - 2.0 / 3.0) < 1e-9
    assert abs(conf.completeness_score - 1.0) < 1e-9
    expected = 0.4 * 0.80 + 0.4 * (2.0 / 3.0) + 0.2 * 1.0
    assert abs(conf.overall_score - expected) < 1e-9


def test_unanswerable_zero_completeness():
    """Retrieval=0.90, insufficient context, no verification.
    Expected: 0.4*0.90 + 0.4*0.0 + 0.2*0.0 = 0.36
    """
    trace = _trace_with_score(0.90)
    answer = _grounded(sufficient=False, has_citations=False)
    conf = compute_confidence(answer, None, trace)

    assert abs(conf.retrieval_score - 0.90) < 1e-9
    assert abs(conf.citation_coverage - 0.0) < 1e-9
    assert abs(conf.completeness_score - 0.0) < 1e-9
    assert abs(conf.overall_score - 0.36) < 1e-9


def test_edge_all_zero():
    """Empty trace, insufficient context, no verification. All zero."""
    trace = QueryTrace(trace_id="t", stages=[])
    answer = _grounded(sufficient=False, has_citations=False)
    conf = compute_confidence(answer, None, trace)

    assert conf.overall_score == 0.0
    assert conf.retrieval_score == 0.0
    assert conf.citation_coverage == 0.0
    assert conf.completeness_score == 0.0


def test_edge_all_one():
    """Retrieval=1.0, all SUPPORTED (coverage=1.0), sufficient+citations (completeness=1.0).
    Expected: 0.4 + 0.4 + 0.2 = 1.0
    """
    trace = _trace_with_score(1.0)
    answer = _grounded(sufficient=True, has_citations=True)
    verification = _verification(supported=5, unsupported=0)
    conf = compute_confidence(answer, verification, trace)

    assert abs(conf.overall_score - 1.0) < 1e-9


def test_weights_sum_to_one():
    assert abs(WEIGHT_RETRIEVAL + WEIGHT_CITATION + WEIGHT_COMPLETENESS - 1.0) < 1e-9


def test_components_breakdown_present():
    trace = _trace_with_score(0.7)
    answer = _grounded(sufficient=True, has_citations=True)
    verification = _verification(supported=1, unsupported=1)
    conf = compute_confidence(answer, verification, trace)

    assert "retrieval_weighted" in conf.components_breakdown
    assert "citation_weighted" in conf.components_breakdown
    assert "completeness_weighted" in conf.components_breakdown
    total = sum(conf.components_breakdown.values())
    assert abs(total - conf.overall_score) < 1e-9


def test_retrieval_score_rrf_stage_consensus_normalized():
    from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, StageTrace

    k = 60
    rrf_consensus_score = (0.5 / (k + 1)) + (0.5 / (k + 1))  # 1 / 61
    trace = QueryTrace(
        trace_id="t_rrf",
        config=PipelineConfig(rrf_k=k),
        stages=[
            StageTrace(
                stage="rrf",
                candidates=[CandidateResult(chunk_id="c1", score=rrf_consensus_score, rank=1)],
            )
        ],
    )
    score = retrieval_score(trace)
    assert score == pytest.approx(1.0, rel=1e-6)


def test_retrieval_score_rrf_stage_single_arm_normalized():
    from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, StageTrace

    k = 60
    rrf_single_arm_score = 0.5 / (k + 1)  # 0.5 / 61
    trace = QueryTrace(
        trace_id="t_rrf_single",
        config=PipelineConfig(rrf_k=k),
        stages=[
            StageTrace(
                stage="rrf",
                candidates=[CandidateResult(chunk_id="c1", score=rrf_single_arm_score, rank=1)],
            )
        ],
    )
    score = retrieval_score(trace)
    assert score == pytest.approx(0.7, rel=1e-6)


def test_retrieval_score_rrf_stage_custom_k():
    from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, StageTrace

    k = 20
    rrf_score = 1.0 / (k + 1)
    trace = QueryTrace(
        trace_id="t_rrf_k20",
        config=PipelineConfig(rrf_k=k),
        stages=[
            StageTrace(
                stage="rrf",
                candidates=[CandidateResult(chunk_id="c1", score=rrf_score, rank=1)],
            )
        ],
    )
    score = retrieval_score(trace)
    assert score == pytest.approx(1.0, rel=1e-6)


def test_retrieval_score_multi_query_rrf_stage():
    from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, StageTrace

    k = 60
    rrf_score = 1.0 / (k + 1)
    trace = QueryTrace(
        trace_id="t_mq_rrf",
        config=PipelineConfig(rrf_k=k),
        stages=[
            StageTrace(
                stage="multi_query_rrf",
                candidates=[CandidateResult(chunk_id="c1", score=rrf_score, rank=1)],
            )
        ],
    )
    score = retrieval_score(trace)
    assert score == pytest.approx(1.0, rel=1e-6)


def test_retrieval_score_rrf_stage_passthrough_not_overscaled():
    from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, StageTrace

    # When alpha=1.0 passthrough occurs, score is raw cosine e.g. 0.85
    trace = QueryTrace(
        trace_id="t_rrf_passthrough",
        config=PipelineConfig(rrf_k=60),
        stages=[
            StageTrace(
                stage="rrf",
                candidates=[CandidateResult(chunk_id="c1", score=0.85, rank=1)],
            )
        ],
    )
    score = retrieval_score(trace)
    assert score == pytest.approx(0.85, rel=1e-6)


def test_composite_confidence_rrf_stage_full_pipeline():
    from hybrid_rag.schemas.pipeline import CandidateResult, PipelineConfig, StageTrace

    k = 60
    trace = QueryTrace(
        trace_id="t_full_rrf",
        config=PipelineConfig(rrf_k=k),
        stages=[
            StageTrace(
                stage="rrf",
                candidates=[CandidateResult(chunk_id="c1", score=1.0 / 61, rank=1)],
            )
        ],
    )
    answer = _grounded(sufficient=True, has_citations=True)
    verification = _verification(supported=3, unsupported=0)

    conf = compute_confidence(answer, verification, trace)
    assert conf.retrieval_score == pytest.approx(1.0, rel=1e-6)
    assert conf.citation_coverage == pytest.approx(1.0, rel=1e-6)
    assert conf.completeness_score == pytest.approx(1.0, rel=1e-6)
    assert conf.overall_score == pytest.approx(1.0, rel=1e-6)
