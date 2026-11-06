"""Tests for core JobRegistry async queue thread safety and subscriber lifecycle."""

import asyncio
import threading
import time

import pytest

from hybrid_rag.core.jobs import JobRegistry


@pytest.mark.asyncio
async def test_late_subscriber_completed_job():
    """Verify subscribing to an already COMPLETED job immediately yields completion event."""
    registry = JobRegistry()
    job = registry.create_job("ingest")
    registry.complete_job(job.job_id, {"status": "success", "count": 10})

    events = []
    async for event in registry.subscribe(job.job_id):
        events.append(event)

    assert len(events) == 1
    assert events[0]["event"] == "completed"
    assert events[0]["data"] == {"status": "success", "count": 10}


@pytest.mark.asyncio
async def test_late_subscriber_failed_job():
    """Verify subscribing to an already FAILED job immediately yields error event."""
    registry = JobRegistry()
    job = registry.create_job("eval")
    registry.fail_job(job.job_id, "Corrupted evaluation dataset")

    events = []
    async for event in registry.subscribe(job.job_id):
        events.append(event)

    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert events[0]["data"]["error"] == "Corrupted evaluation dataset"


@pytest.mark.asyncio
async def test_late_subscriber_cancelled_job():
    """Verify subscribing to an already CANCELLED job immediately yields cancelled event."""
    registry = JobRegistry()
    job = registry.create_job("query")
    registry.cancel_job(job.job_id)

    events = []
    async for event in registry.subscribe(job.job_id):
        events.append(event)

    assert len(events) == 1
    assert events[0]["event"] == "cancelled"


@pytest.mark.asyncio
async def test_threadsafe_worker_notifications():
    """Verify worker threads can dispatch notifications safely to async subscribers."""
    registry = JobRegistry()
    job = registry.create_job("background_task")
    received_events = []

    async def subscriber():
        async for event in registry.subscribe(job.job_id):
            received_events.append(event)
            if event["event"] == "completed":
                break

    sub_task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.01)

    # Simulate worker background thread
    def worker():
        time.sleep(0.05)
        registry.update_progress(job.job_id, 25.0, "step 1")
        time.sleep(0.02)
        registry.update_progress(job.job_id, 50.0, "step 2")
        time.sleep(0.02)
        registry.update_progress(job.job_id, 100.0, "step 3")
        time.sleep(0.02)
        registry.complete_job(job.job_id, {"done": True})

    thread = threading.Thread(target=worker)
    thread.start()

    await asyncio.wait_for(sub_task, timeout=3.0)
    thread.join()

    assert len(received_events) >= 4
    event_types = [e["event"] for e in received_events]
    assert "progress" in event_types
    assert event_types[-1] == "completed"
    assert received_events[-1]["data"] == {"done": True}


@pytest.mark.asyncio
async def test_subscriber_cleanup_on_completion():
    """Verify subscriber queues are removed from registry when subscription ends."""
    registry = JobRegistry()
    job = registry.create_job("cleanup_test")

    async def sub():
        async for event in registry.subscribe(job.job_id):
            if event["event"] == "completed":
                break

    task = asyncio.create_task(sub())
    await asyncio.sleep(0.01)

    # While subscribed, registry has 1 subscriber
    with registry._lock:
        assert len(registry._subscribers[job.job_id]) == 1

    registry.complete_job(job.job_id)
    await asyncio.wait_for(task, timeout=1.0)

    # After completion, subscriber is cleaned up
    with registry._lock:
        assert len(registry._subscribers[job.job_id]) == 0


@pytest.mark.asyncio
async def test_subscribe_nonexistent_job():
    """Verify subscribing to a non-existent job yields nothing."""
    registry = JobRegistry()
    events = []
    async for event in registry.subscribe("unknown-id"):
        events.append(event)
    assert len(events) == 0
