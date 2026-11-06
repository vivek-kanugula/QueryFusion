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

        try:
            from hybrid_rag.retrieval.bm25_store import get_bm25_store
            from hybrid_rag.retrieval.retriever import full_retrieve

            yield _sse_event("stage", {"name": "dense"})
            yield _sse_event("stage", {"name": "sparse"})
            yield _sse_event("stage", {"name": "rrf"})

            start = time.perf_counter()
            bm25 = get_bm25_store()
            retrieval_trace = await asyncio.to_thread(
                full_retrieve,
                request.question,
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
                verify=request.verify,
                temperature=request.temperature,
            )
            gen_ms = (time.perf_counter() - start) * 1000

            if request.verify:
                yield _sse_event("stage", {"name": "verification"})

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
