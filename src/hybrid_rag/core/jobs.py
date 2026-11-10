"""Thread-safe background job registry with SSE subscriber fan-out."""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections import OrderedDict
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
    def __init__(self, max_jobs: int = 1000) -> None:
        self._lock = threading.Lock()
        self._max_jobs = max(1, max_jobs)
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}

    def _evict_one(self) -> None:
        """Evict one job to maintain capacity. Caller must hold self._lock.

        Eviction priority:
        1. Oldest terminal job (COMPLETED, FAILED, CANCELLED)
        2. If no terminal jobs exist, oldest job (FIFO)
        """
        terminal_statuses = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        target_id: str | None = None
        for j_id, j in self._jobs.items():
            if j.status in terminal_statuses:
                target_id = j_id
                break

        if target_id is not None:
            del self._jobs[target_id]
            self._subscribers.pop(target_id, None)
        elif self._jobs:
            oldest_id, _ = self._jobs.popitem(last=False)
            self._subscribers.pop(oldest_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()
            self._subscribers.clear()

    def create_job(self, job_type: str) -> Job:
        job = Job(job_id=str(uuid.uuid4()), job_type=job_type)
        with self._lock:
            while len(self._jobs) >= self._max_jobs:
                self._evict_one()
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
        terminal_event: dict | None = None
        with self._lock:
            if job_id not in self._jobs:
                return
            job = self._jobs[job_id]
            if job.status == JobStatus.COMPLETED:
                terminal_event = {"event": "completed", "data": job.result or {}}
            elif job.status == JobStatus.FAILED:
                terminal_event = {"event": "error", "data": {"error": job.error or ""}}
            elif job.status == JobStatus.CANCELLED:
                terminal_event = {"event": "cancelled", "data": {}}
            else:
                if job_id not in self._subscribers:
                    self._subscribers[job_id] = []
                self._subscribers[job_id].append((loop, queue))

        if terminal_event is not None:
            yield terminal_event
            return
        try:
            while True:
                event = await queue.get()
                is_terminal = event.get("event") in ("completed", "error", "cancelled")
                if is_terminal:
                    with self._lock:
                        if job_id in self._subscribers:
                            subs = self._subscribers[job_id]
                            self._subscribers[job_id] = [
                                pair for pair in subs if pair[1] is not queue
                            ]
                yield event
                if is_terminal:
                    break
        finally:
            with self._lock:
                if job_id in self._subscribers:
                    subs = self._subscribers[job_id]
                    self._subscribers[job_id] = [
                        pair for pair in subs if pair[1] is not queue
                    ]

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
