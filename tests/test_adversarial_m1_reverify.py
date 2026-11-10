"""Adversarial re-verification stress test harness for Milestone 1.

Specifically focuses on:
- Deep concurrency stress testing of JobRegistry.subscribe()
- Verifying lock release across all terminal statuses (COMPLETED, FAILED, CANCELLED)
- Reentrancy / operations inside consumer generator loops
- Subscriber cleanup under break, exceptions, and task cancellation
- High-concurrency multi-threaded and multi-async worker churn
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time

import pytest

from hybrid_rag.core.jobs import JobRegistry, JobStatus


@pytest.mark.asyncio
@pytest.mark.parametrize("status,extra_call", [
    (JobStatus.COMPLETED, lambda reg, jid: reg.complete_job(jid, {"result": "success"})),
    (JobStatus.FAILED, lambda reg, jid: reg.fail_job(jid, "critical error occurred")),
    (JobStatus.CANCELLED, lambda reg, jid: reg.cancel_job(jid)),
])
async def test_all_terminal_statuses_release_lock_during_consumption(status, extra_call):
    """Verify that reg._lock is unlocked during event consumption for ALL terminal statuses."""
    reg = JobRegistry(max_jobs=10)
    job = reg.create_job("test_job")
    extra_call(reg, job.job_id)

    events_received = []
    async for event in reg.subscribe(job.job_id):
        # 1. Assert lock is NOT held by generator
        assert not reg._lock.locked(), (
            f"reg._lock is held during consumption of {event['event']} event!"
        )

        # 2. Assert reentrant operations on reg succeed without deadlocking
        assert reg.count() == 1
        fetched = reg.get_job(job.job_id)
        assert fetched is not None
        assert fetched.status == status

        # 3. Assert creating a new job inside the loop does not deadlock
        new_j = reg.create_job("child_job")
        assert new_j is not None
        assert reg.count() == 2

        events_received.append(event)

    assert len(events_received) == 1


@pytest.mark.asyncio
async def test_active_job_progress_and_completion_never_hold_lock_during_yield():
    """Verify that lock is never held when consuming events on active jobs."""
    reg = JobRegistry(max_jobs=10)
    job = reg.create_job("active_job")

    consumed_events = []
    lock_states = []

    async def consumer():
        async for event in reg.subscribe(job.job_id):
            lock_states.append(reg._lock.locked())
            # Perform reentrant registry calls
            c = reg.count()
            consumed_events.append((event["event"], c))
            if event["event"] == "completed":
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.02)

    # Dispatch progress updates and completion
    reg.update_progress(job.job_id, 25.0, "step 1")
    await asyncio.sleep(0.01)
    reg.update_progress(job.job_id, 50.0, "step 2")
    await asyncio.sleep(0.01)
    reg.complete_job(job.job_id, {"done": True})

    await asyncio.wait_for(task, timeout=2.0)

    assert len(consumed_events) == 3
    # Lock MUST be false during all yields
    assert all(locked is False for locked in lock_states), (
        f"Lock was held during yield: {lock_states}"
    )


@pytest.mark.asyncio
async def test_subscriber_early_break_cleans_up_subscribers_list():
    """Verify that breaking early from subscribe() removes the queue from _subscribers."""
    reg = JobRegistry(max_jobs=10)
    job = reg.create_job("early_exit_job")

    async def subscriber():
        async for event in reg.subscribe(job.job_id):
            if event["event"] == "progress":
                break

    task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.02)

    with reg._lock:
        assert len(reg._subscribers[job.job_id]) == 1

    reg.update_progress(job.job_id, 10.0, "starting")
    await asyncio.wait_for(task, timeout=1.0)
    gc.collect()
    await asyncio.sleep(0.05)

    # After break and event loop tick, queue should be cleaned up via finally block
    with reg._lock:
        assert len(reg._subscribers[job.job_id]) == 0


@pytest.mark.asyncio
async def test_subscriber_task_cancellation_cleans_up_subscribers_list():
    """Verify that cancelling the subscriber task removes the queue from _subscribers."""
    reg = JobRegistry(max_jobs=10)
    job = reg.create_job("cancelled_task_job")

    async def subscriber():
        async for event in reg.subscribe(job.job_id):
            pass

    task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.02)

    with reg._lock:
        assert len(reg._subscribers[job.job_id]) == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # After cancellation, queue should be cleaned up via finally block
    with reg._lock:
        assert len(reg._subscribers[job.job_id]) == 0


@pytest.mark.asyncio
async def test_high_intensity_concurrent_subscribers_and_workers():
    """Stress test with 20 concurrent subscriber coroutines and 5 threaded workers churning jobs."""
    reg = JobRegistry(max_jobs=15)
    jobs = [reg.create_job(f"churn_{i}") for i in range(10)]
    stop_event = threading.Event()
    errors = []

    def background_worker(worker_idx: int):
        step = 0
        while not stop_event.is_set():
            step += 1
            jid = jobs[step % len(jobs)].job_id
            reg.update_progress(jid, float((step * 10) % 100), f"worker {worker_idx}")
            if step % 7 == 0:
                new_j = reg.create_job(f"w_{worker_idx}_{step}")
                reg.complete_job(new_j.job_id, {"worker": worker_idx})
            time.sleep(0.005)

    worker_threads = [
        threading.Thread(target=background_worker, args=(w,), daemon=True)
        for w in range(5)
    ]
    for wt in worker_threads:
        wt.start()

    async def subscriber_coro(sub_idx: int):
        try:
            for _ in range(5):
                jid = jobs[sub_idx % len(jobs)].job_id
                async for event in reg.subscribe(jid):
                    # Do non-trivial work without deadlocking
                    c = reg.count()
                    assert c <= 15
                    assert not reg._lock.locked()
                    break
                await asyncio.sleep(0.005)
        except Exception as ex:
            errors.append(ex)

    subscriber_tasks = [
        asyncio.create_task(subscriber_coro(i))
        for i in range(20)
    ]

    await asyncio.gather(*subscriber_tasks)
    stop_event.set()
    for wt in worker_threads:
        wt.join(timeout=1.0)

    assert len(errors) == 0, f"Errors occurred during high intensity stress test: {errors}"
    assert reg.count() <= 15
    with reg._lock:
        assert len(reg._subscribers) <= 15


@pytest.mark.asyncio
async def test_100_concurrent_subscribers_fanout_on_terminal_events():
    """Verify 100 concurrent subscribers fanout and cleanly complete without deadlocks."""
    reg = JobRegistry(max_jobs=10)
    job = reg.create_job("fanout_job")

    async def single_sub():
        events = []
        async for ev in reg.subscribe(job.job_id):
            assert not reg._lock.locked()
            events.append(ev)
            if ev["event"] == "completed":
                break
        return events

    tasks = [asyncio.create_task(single_sub()) for _ in range(100)]
    await asyncio.sleep(0.05)

    with reg._lock:
        assert len(reg._subscribers[job.job_id]) == 100

    reg.complete_job(job.job_id, {"status": "all_done"})

    all_events = await asyncio.gather(*tasks)
    assert len(all_events) == 100
    for evs in all_events:
        assert len(evs) == 1
        assert evs[0]["event"] == "completed"
        assert evs[0]["data"] == {"status": "all_done"}

    with reg._lock:
        assert len(reg._subscribers[job.job_id]) == 0


@pytest.mark.asyncio
async def test_rapid_fire_subscribe_and_immediate_completion_race():
    """Race test: 50 cycles of subscribe + immediate complete in separate thread."""
    reg = JobRegistry(max_jobs=10)

    for i in range(50):
        job = reg.create_job(f"race_{i}")

        async def sub():
            evs = []
            async for ev in reg.subscribe(job.job_id):
                evs.append(ev)
            return evs

        sub_task = asyncio.create_task(sub())

        # Concurrently complete from thread
        def complete():
            reg.complete_job(job.job_id, {"cycle": i})

        t = threading.Thread(target=complete)
        t.start()
        t.join()

        events = await asyncio.wait_for(sub_task, timeout=1.0)
        assert len(events) == 1
        assert events[0]["event"] == "completed"
        assert events[0]["data"]["cycle"] == i
