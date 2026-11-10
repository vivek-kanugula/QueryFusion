"""Corrective RAG (CRAG) self-evaluator and fallback refinement loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hybrid_rag.schemas.generation import (
    GenerationResponse,
    VerificationVerdict,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from hybrid_rag.retrieval.bm25_store import Bm25Store
    from hybrid_rag.schemas.pipeline import PipelineConfig, QueryTrace


def should_trigger_crag(
    gen_response: GenerationResponse,
    min_confidence: float = 0.5,
) -> tuple[bool, str]:
    """Evaluate whether an initial generation response requires corrective refinement.

    Checks:
    1. Insufficient context reported by generator.
    2. Verification engine identified ungrounded or fabricated claims.
    3. Composite confidence score fell below acceptable threshold.
    """
    if not gen_response.answer.has_sufficient_context:
        return True, "Insufficient context reported in initial retrieval"

    if gen_response.verification is not None:
        unsupported = [
            c
            for c in gen_response.verification.claims
            if c.verdict in (VerificationVerdict.UNSUPPORTED, VerificationVerdict.MISATTRIBUTED)
        ]
        if unsupported:
            return (
                True,
                f"Citation verifier detected {len(unsupported)} ungrounded claim(s)",
            )
        if gen_response.verification.citation_coverage < 0.6:
            return (
                True,
                f"Low citation coverage ({gen_response.verification.citation_coverage * 100:.0f}%)",
            )

    if (
        gen_response.confidence is not None
        and gen_response.confidence.overall_score < min_confidence
    ):
        score_val = gen_response.confidence.overall_score
        return (
            True,
            f"Confidence score ({score_val:.2f}) below threshold ({min_confidence:.2f})",
        )

    return False, ""


def execute_corrective_fallback(
    query: str,
    original_trace: QueryTrace,
    initial_response: GenerationResponse,
    reason: str,
    config: PipelineConfig,
    bm25_store: Bm25Store | None = None,
    db: Session | None = None,
    chat_history: list[dict[str, str]] | None = None,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
) -> GenerationResponse:
    """Execute Corrective RAG (CRAG) fallback retrieval and re-generation."""
    from hybrid_rag.generation.pipeline import generate_grounded_answer
    from hybrid_rag.retrieval.retriever import full_retrieve
    from hybrid_rag.schemas.pipeline import PipelineConfig as ConfigCls

    corrected_config = ConfigCls(
        retrieval_strategy="hybrid_rerank",
        chunking_strategy=config.chunking_strategy,
        alpha=0.5,
        dense_top_k=min(config.dense_top_k * 2, 60),
        sparse_top_k=min(config.sparse_top_k * 2, 60),
        fusion_top_k=min(30, config.fusion_top_k + 10),
        rerank_enabled=True,
        rerank_top_k=min(config.rerank_top_k + 2, 10),
        document_ids=config.document_ids,
        temperature=temperature,
    )

    # Perform fallback retrieval
    fallback_trace = full_retrieve(query, corrected_config, bm25_store=bm25_store, db=db)

    # Re-generate with grounded pipeline
    corrected_response = generate_grounded_answer(
        query=query,
        trace=fallback_trace,
        chat_history=chat_history,
        model=model,
        provider=provider,
        temperature=temperature,
        verify=True,
    )

    corrected_response.was_corrected = True
    corrected_response.correction_reason = reason
    return corrected_response
