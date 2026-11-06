"""Job progress API and SSE streaming endpoint."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from hybrid_rag.core.jobs import job_registry

router = APIRouter(
    prefix="/api/jobs",
    tags=["jobs"],
)

HEARTBEAT_INTERVAL = 15


@router.get("/{job_id}")
def get_job(job_id: str):
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str):
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    cancelled = job_registry.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            409,
            f"Cannot cancel job in {job.status} state",
        )
    return {"job_id": job_id, "status": "CANCELLED"}


@router.get("/{job_id}/events")
async def job_events(job_id: str):
    job = job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    async def event_stream():
        async for event in job_registry.subscribe(job_id):
            event_type = event.get("event", "message")
            data = json.dumps(event.get("data", {}))
            yield f"event: {event_type}\ndata: {data}\n\n"

            if event_type in ("completed", "error", "cancelled"):
                return

    async def stream_with_heartbeat():
        gen = event_stream()
        while True:
            try:
                chunk = await asyncio.wait_for(
                    gen.__anext__(),
                    timeout=HEARTBEAT_INTERVAL,
                )
                yield chunk
            except StopAsyncIteration:
                return
            except TimeoutError:
                yield "event: ping\ndata: {}\n\n"

    return StreamingResponse(
        stream_with_heartbeat(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
