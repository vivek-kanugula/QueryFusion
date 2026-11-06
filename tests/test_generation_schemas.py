"""Tests for generation schemas: validation, serialization, enums."""

from hybrid_rag.schemas.generation import (
    Citation,
    ClaimVerification,
    CompositeConfidence,
    GenerationResponse,
    GroundedAnswer,
    VerificationResult,
    VerificationVerdict,
)


def test_citation_basic():
    c = Citation(citation_id="[C1]", chunk_id="abc-123", document_id="doc-1", evidence_quote="foo")
    assert c.citation_id == "[C1]"
    assert c.chunk_id == "abc-123"


def test_grounded_answer_with_citations():
    a = GroundedAnswer(
        answer="Result [C1].",
        citations=[Citation(citation_id="[C1]", chunk_id="c1")],
        has_sufficient_context=True,
    )
    assert len(a.citations) == 1
    assert a.fallback_reason is None


def test_grounded_answer_fallback():
    a = GroundedAnswer(
        answer="Cannot answer.",
        citations=[],
        has_sufficient_context=False,
        fallback_reason="No context.",
    )
    assert not a.has_sufficient_context
    assert a.fallback_reason == "No context."


def test_verification_verdict_enum():
    assert VerificationVerdict.SUPPORTED.value == "SUPPORTED"
    assert VerificationVerdict.UNSUPPORTED.value == "UNSUPPORTED"
    assert VerificationVerdict.MISATTRIBUTED.value == "MISATTRIBUTED"


def test_claim_verification():
    cv = ClaimVerification(
        claim_text="Python is fast",
        citation_id="[C1]",
        verdict=VerificationVerdict.SUPPORTED,
        evidence_quote="Python runs quickly",
        explanation="Direct support",
    )
    assert cv.verdict == VerificationVerdict.SUPPORTED


def test_verification_result():
    vr = VerificationResult(
        claims=[
            ClaimVerification(
                claim_text="X", verdict=VerificationVerdict.SUPPORTED, explanation="ok"
            ),
            ClaimVerification(
                claim_text="Y", verdict=VerificationVerdict.UNSUPPORTED, explanation="no"
            ),
        ],
        citation_coverage=0.5,
        all_supported=False,
    )
    assert len(vr.claims) == 2
    assert vr.citation_coverage == 0.5


def test_composite_confidence_bounds():
    cc = CompositeConfidence(
        overall_score=0.8,
        retrieval_score=1.0,
        citation_coverage=0.75,
        completeness_score=0.5,
    )
    assert 0.0 <= cc.overall_score <= 1.0


def test_generation_response_full():
    resp = GenerationResponse(
        query="What is X?",
        answer=GroundedAnswer(answer="X is Y [C1].", citations=[], has_sufficient_context=True),
        trace_id="trace-1",
        model="gemini-3.5-flash-lite",
    )
    assert resp.query == "What is X?"
    assert resp.model == "gemini-3.5-flash-lite"


def test_grounded_answer_serialization():
    a = GroundedAnswer(
        answer="Test [C1].",
        citations=[Citation(citation_id="[C1]", chunk_id="c1")],
        has_sufficient_context=True,
    )
    json_str = a.model_dump_json()
    loaded = GroundedAnswer.model_validate_json(json_str)
    assert loaded.answer == a.answer
    assert len(loaded.citations) == 1


def test_verification_result_serialization():
    vr = VerificationResult(
        claims=[
            ClaimVerification(
                claim_text="test", verdict=VerificationVerdict.SUPPORTED, explanation=""
            )
        ],
        citation_coverage=1.0,
        all_supported=True,
    )
    json_str = vr.model_dump_json()
    loaded = VerificationResult.model_validate_json(json_str)
    assert loaded.all_supported is True
    assert loaded.claims[0].verdict == VerificationVerdict.SUPPORTED
