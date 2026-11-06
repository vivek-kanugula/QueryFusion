"""Schemas for grounded generation, citation verification, and confidence scoring."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Citation(BaseModel):
    citation_id: str = Field(description="Inline marker, e.g. [C1]")
    chunk_id: str
    document_id: str = ""
    evidence_quote: str = ""


class GroundedAnswer(BaseModel):
    answer: str = Field(description="Answer text with inline [C1] markers")
    citations: list[Citation] = Field(default_factory=list)
    has_sufficient_context: bool = True
    fallback_reason: str | None = None


class VerificationVerdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    MISATTRIBUTED = "MISATTRIBUTED"


class ClaimVerification(BaseModel):
    claim_text: str
    citation_id: str | None = None
    verdict: VerificationVerdict
    evidence_quote: str | None = None
    explanation: str = ""


class VerificationResult(BaseModel):
    claims: list[ClaimVerification] = Field(default_factory=list)
    citation_coverage: float = 0.0
    all_supported: bool = False


class CompositeConfidence(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    retrieval_score: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    components_breakdown: dict[str, float] = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    query: str
    answer: GroundedAnswer
    verification: VerificationResult | None = None
    confidence: CompositeConfidence | None = None
    trace_id: str = ""
    model: str = ""
    cached: bool = False
    was_corrected: bool = False
    correction_reason: str | None = None
