"""Streaming RAG query endpoint with SSE stage milestones."""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from hybrid_rag.core.trace_store import trace_store
from hybrid_rag.db.engine import get_db
from hybrid_rag.schemas.pipeline import PipelineConfig, QueryTrace

router = APIRouter(
    prefix="/api/query",
    tags=["query"],
)


class QueryRequest(BaseModel):
    question: str
    retrieval_strategy: str = "hybrid_rerank"
    chunking_strategy: str = "fixed"
    alpha: float = 0.5
    rerank_enabled: bool = True
    verify: bool = True
    temperature: float = 0.0
    document_ids: list[str] | None = None
    chat_history: list[dict[str, str]] | None = None
    model: str | None = None
    provider: str | None = None
    query_expansion: bool = False
    enable_crag: bool = True


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/stream")
async def stream_query(request: QueryRequest, db: Session = Depends(get_db)):
    async def event_stream():
        import asyncio

        doc_ids = tuple(request.document_ids) if request.document_ids else None
        config = PipelineConfig(
            retrieval_strategy=request.retrieval_strategy,
            chunking_strategy=request.chunking_strategy,
            alpha=request.alpha,
            rerank_enabled=request.rerank_enabled,
            temperature=request.temperature,
            document_ids=doc_ids,
        )

        trace = QueryTrace(config=config, query=request.question)

        yield _sse_event("stage", {"name": "started", "trace_id": trace.trace_id})

        search_query = request.question
        if request.chat_history:
            from hybrid_rag.generation.contextualizer import contextualize_query

            yield _sse_event("stage", {"name": "contextualize"})
            search_query = await asyncio.to_thread(
                contextualize_query,
                request.question,
                request.chat_history,
                model=request.model,
                provider=request.provider,
            )

        expanded_queries = [search_query]
        if request.query_expansion:
            from hybrid_rag.retrieval.expansion import expand_query

            yield _sse_event("stage", {"name": "expansion"})
            expanded_queries = await asyncio.to_thread(
                expand_query,
                search_query,
                model=request.model,
                provider=request.provider,
            )

        try:
            from hybrid_rag.retrieval.bm25_store import get_bm25_store
            from hybrid_rag.retrieval.retriever import full_retrieve, multi_query_retrieve

            yield _sse_event("stage", {"name": "dense"})
            yield _sse_event("stage", {"name": "sparse"})
            yield _sse_event("stage", {"name": "rrf"})

            start = time.perf_counter()
            bm25 = get_bm25_store()
            if len(expanded_queries) > 1:
                retrieval_trace = await asyncio.to_thread(
                    multi_query_retrieve,
                    expanded_queries,
                    config,
                    bm25,
                    db=db,
                )
            else:
                retrieval_trace = await asyncio.to_thread(
                    full_retrieve,
                    search_query,
                    config,
                    bm25,
                    db=db,
                )
            retrieval_ms = (time.perf_counter() - start) * 1000

            if config.rerank_enabled:
                yield _sse_event("stage", {"name": "rerank"})

            yield _sse_event(
                "stage",
                {
                    "name": "retrieval_complete",
                    "duration_ms": round(retrieval_ms, 1),
                    "candidates": len(retrieval_trace.stages[-1].candidates)
                    if retrieval_trace.stages
                    else 0,
                },
            )

            from hybrid_rag.generation.pipeline import (
                generate_grounded_answer,
            )

            yield _sse_event("stage", {"name": "generation"})

            start = time.perf_counter()
            gen_response = await asyncio.to_thread(
                generate_grounded_answer,
                request.question,
                retrieval_trace,
                chat_history=request.chat_history,
                model=request.model,
                provider=request.provider,
                verify=request.verify,
                temperature=request.temperature,
            )

            if request.verify:
                yield _sse_event("stage", {"name": "verification"})

            if request.enable_crag:
                from hybrid_rag.generation.crag import (
                    execute_corrective_fallback,
                    should_trigger_crag,
                )

                needs_crag, crag_reason = should_trigger_crag(gen_response, min_confidence=0.5)
                if needs_crag:
                    yield _sse_event(
                        "stage",
                        {"name": "corrective_fallback", "reason": crag_reason},
                    )
                    gen_response = await asyncio.to_thread(
                        execute_corrective_fallback,
                        search_query,
                        retrieval_trace,
                        gen_response,
                        crag_reason,
                        config,
                        bm25_store=bm25,
                        db=db,
                        chat_history=request.chat_history,
                        model=request.model,
                        provider=request.provider,
                        temperature=request.temperature,
                    )

            gen_ms = (time.perf_counter() - start) * 1000

            trace_store.save(retrieval_trace)

            yield _sse_event(
                "answer",
                {
                    "answer": gen_response.answer.answer,
                    "citations": [c.model_dump() for c in gen_response.answer.citations],
                    "has_sufficient_context": (gen_response.answer.has_sufficient_context),
                    "fallback_reason": gen_response.answer.fallback_reason,
                },
            )

            if gen_response.verification:
                yield _sse_event(
                    "verification",
                    {
                        "citation_coverage": (gen_response.verification.citation_coverage),
                        "all_supported": (gen_response.verification.all_supported),
                        "claims": [c.model_dump() for c in gen_response.verification.claims],
                    },
                )

            if gen_response.confidence:
                yield _sse_event("confidence", (gen_response.confidence.model_dump()))

            stage_durations = {
                s.stage: round(s.duration_ms, 1) for s in retrieval_trace.stages
            }
            total_elapsed = round(retrieval_ms + gen_ms, 1)

            yield _sse_event(
                "completed",
                {
                    "trace_id": retrieval_trace.trace_id,
                    "model": gen_response.model,
                    "cached": gen_response.cached,
                    "retrieval_ms": round(retrieval_ms, 1),
                    "generation_ms": round(gen_ms, 1),
                    "total_ms": total_elapsed,
                    "stages": stage_durations,
                    "contextualized_query": (
                        search_query if search_query != request.question else None
                    ),
                    "expanded_queries": (
                        expanded_queries if len(expanded_queries) > 1 else None
                    ),
                    "was_corrected": gen_response.was_corrected,
                    "correction_reason": gen_response.correction_reason,
                },
            )

        except Exception as exc:
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
