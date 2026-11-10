"""Trace inspection endpoint (development only)."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from hybrid_rag.core.trace_store import trace_store
from hybrid_rag.settings import settings

router = APIRouter(
    prefix="/api/traces",
    tags=["traces"],
)


@router.get("/{trace_id}")
def get_trace(trace_id: str):
    if settings.is_production:
        raise HTTPException(404, "Not Found")
    trace = trace_store.get(trace_id)
    if trace is None:
        raise HTTPException(404, "Trace not found")
    data = asdict(trace)
    data["stages"] = [
        {
            "stage": s.stage,
            "duration_ms": s.duration_ms,
            "candidate_count": len(s.candidates),
            "top_scores": [
                {"chunk_id": c.chunk_id, "score": c.score, "rank": c.rank} for c in s.candidates[:5]
            ],
            "metadata": s.metadata,
        }
        for s in trace.stages
    ]
    return data
