"""End-to-end generation pipeline: trace to grounded answer with verification."""

from __future__ import annotations

from hybrid_rag.generation.cache import LLMCache, TokenBucketRateLimiter
from hybrid_rag.generation.confidence import compute_confidence
from hybrid_rag.generation.generator import generate_answer
from hybrid_rag.generation.verifier import verify_citations
from hybrid_rag.schemas.generation import GenerationResponse
from hybrid_rag.schemas.pipeline import CandidateResult, QueryTrace
from hybrid_rag.settings import settings


def _get_final_candidates(trace: QueryTrace) -> list[CandidateResult]:
    if not trace.stages:
        return []
    return trace.stages[-1].candidates


def generate_grounded_answer(
    query: str,
    trace: QueryTrace,
    llm_cache: LLMCache | None = None,
    rate_limiter: TokenBucketRateLimiter | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    verify: bool = True,
) -> GenerationResponse:
    model_name = model or settings.gemini_model
    candidates = _get_final_candidates(trace)

    answer, cached = generate_answer(
        question=query,
        candidates=candidates,
        llm_cache=llm_cache,
        rate_limiter=rate_limiter,
        model=model_name,
        temperature=temperature,
    )

    verification = None
    if verify and answer.has_sufficient_context and answer.citations:
        verification = verify_citations(
            answer=answer,
            candidates=candidates,
            llm_cache=llm_cache,
            rate_limiter=rate_limiter,
            model=model_name,
            temperature=temperature,
        )

    confidence = compute_confidence(answer, verification, trace)

    return GenerationResponse(
        query=query,
        answer=answer,
        verification=verification,
        confidence=confidence,
        trace_id=trace.trace_id,
        model=model_name,
        cached=cached,
    )
