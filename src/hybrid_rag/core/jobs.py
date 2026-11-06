"""Thread-safe background job registry with SSE subscriber fan-out."""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Job:
    job_id: str
    job_type: str
    status: JobStatus = JobStatus.PENDING
    progress_percent: float = 0.0
    status_message: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
            "status_message": self.status_message,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}

    def create_job(self, job_type: str) -> Job:
        job = Job(job_id=str(uuid.uuid4()), job_type=job_type)
        with self._lock:
            self._jobs[job.job_id] = job
            self._subscribers[job.job_id] = []
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_progress(
        self,
        job_id: str,
        progress: float,
        message: str = "",
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.RUNNING
            job.progress_percent = min(100.0, max(0.0, progress))
            job.status_message = message
        self._notify(
            job_id,
            "progress",
            {
                "progress_percent": job.progress_percent,
                "status_message": message,
            },
        )

    def complete_job(
        self,
        job_id: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.COMPLETED
            job.progress_percent = 100.0
            job.completed_at = datetime.now(UTC).isoformat()
            job.result = result
        self._notify(job_id, "completed", result or {})

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(UTC).isoformat()
            job.error = error
        self._notify(job_id, "error", {"error": error})

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                return False
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(UTC).isoformat()
        self._notify(job_id, "cancelled", {})
        return True

    async def subscribe(self, job_id: str) -> AsyncGenerator[dict, None]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            if job_id not in self._jobs:
                return
            job = self._jobs[job_id]
            if job.status == JobStatus.COMPLETED:
                yield {"event": "completed", "data": job.result or {}}
                return
            elif job.status == JobStatus.FAILED:
                yield {"event": "error", "data": {"error": job.error or ""}}
                return
            elif job.status == JobStatus.CANCELLED:
                yield {"event": "cancelled", "data": {}}
                return

            if job_id not in self._subscribers:
                self._subscribers[job_id] = []
            self._subscribers[job_id].append((loop, queue))
        try:
            while True:
                event = await queue.get()
                is_terminal = event.get("event") in ("completed", "error", "cancelled")
                if is_terminal:
                    with self._lock:
                        subs = self._subscribers.get(job_id, [])
                        self._subscribers[job_id] = [pair for pair in subs if pair[1] is not queue]
                yield event
                if is_terminal:
                    break
        finally:
            with self._lock:
                subs = self._subscribers.get(job_id, [])
                self._subscribers[job_id] = [pair for pair in subs if pair[1] is not queue]

    def _notify(
        self,
        job_id: str,
        event_type: str,
        data: dict,
    ) -> None:
        with self._lock:
            subs = list(self._subscribers.get(job_id, []))
        message = {"event": event_type, "data": data}
        for loop, q in subs:
            try:
                if loop.is_running() and not loop.is_closed():
                    loop.call_soon_threadsafe(q.put_nowait, message)
                else:
                    q.put_nowait(message)
            except (asyncio.QueueFull, RuntimeError):
                pass


job_registry = JobRegistry()
