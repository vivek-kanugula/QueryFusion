"""Tests for job registry: lifecycle, progress, cancellation, subscriber fan-out."""

import asyncio

import pytest

from hybrid_rag.core.jobs import JobRegistry, JobStatus


def test_create_job():
    reg = JobRegistry()
    job = reg.create_job("ingest")
    assert job.status == JobStatus.PENDING
    assert job.job_type == "ingest"
    assert job.progress_percent == 0.0


def test_get_job():
    reg = JobRegistry()
    job = reg.create_job("eval")
    fetched = reg.get_job(job.job_id)
    assert fetched is not None
    assert fetched.job_id == job.job_id


def test_get_job_missing():
    reg = JobRegistry()
    assert reg.get_job("nonexistent") is None


def test_update_progress():
    reg = JobRegistry()
    job = reg.create_job("ingest")
    reg.update_progress(job.job_id, 50.0, "Half done")
    fetched = reg.get_job(job.job_id)
    assert fetched.status == JobStatus.RUNNING
    assert fetched.progress_percent == 50.0
    assert fetched.status_message == "Half done"


def test_progress_clamped():
    reg = JobRegistry()
    job = reg.create_job("ingest")
    reg.update_progress(job.job_id, 150.0)
    assert reg.get_job(job.job_id).progress_percent == 100.0
    reg.update_progress(job.job_id, -10.0)
    assert reg.get_job(job.job_id).progress_percent == 0.0


def test_complete_job():
    reg = JobRegistry()
    job = reg.create_job("ingest")
    reg.update_progress(job.job_id, 50.0)
    reg.complete_job(job.job_id, {"chunks": 42})
    fetched = reg.get_job(job.job_id)
    assert fetched.status == JobStatus.COMPLETED
    assert fetched.progress_percent == 100.0
    assert fetched.result == {"chunks": 42}
    assert fetched.completed_at is not None


def test_fail_job():
    reg = JobRegistry()
    job = reg.create_job("ingest")
    reg.fail_job(job.job_id, "parse error")
    fetched = reg.get_job(job.job_id)
    assert fetched.status == JobStatus.FAILED
    assert fetched.error == "parse error"
    assert fetched.completed_at is not None


def test_cancel_pending():
    reg = JobRegistry()
    job = reg.create_job("ingest")
    assert reg.cancel_job(job.job_id) is True
    assert reg.get_job(job.job_id).status == JobStatus.CANCELLED


def test_cancel_running():
    reg = JobRegistry()
    job = reg.create_job("ingest")
    reg.update_progress(job.job_id, 30.0)
    assert reg.cancel_job(job.job_id) is True
    assert reg.get_job(job.job_id).status == JobStatus.CANCELLED


def test_cancel_completed_fails():
    reg = JobRegistry()
    job = reg.create_job("ingest")
    reg.complete_job(job.job_id)
    assert reg.cancel_job(job.job_id) is False


def test_cancel_nonexistent():
    reg = JobRegistry()
    assert reg.cancel_job("nope") is False


def test_job_to_dict():
    reg = JobRegistry()
    job = reg.create_job("eval")
    d = job.to_dict()
    assert d["job_id"] == job.job_id
    assert d["status"] == "PENDING"
    assert d["job_type"] == "eval"


@pytest.mark.asyncio
async def test_subscriber_receives_progress():
    reg = JobRegistry()
    job = reg.create_job("ingest")
    events = []

    async def collect():
        async for event in reg.subscribe(job.job_id):
            events.append(event)
            if event["event"] == "completed":
                break

    task = asyncio.create_task(collect())
    await asyncio.sleep(0.01)
    reg.update_progress(job.job_id, 50.0, "halfway")
    reg.complete_job(job.job_id)
    await asyncio.wait_for(task, timeout=2.0)

    assert len(events) == 2
    assert events[0]["event"] == "progress"
    assert events[0]["data"]["progress_percent"] == 50.0
    assert events[1]["event"] == "completed"


@pytest.mark.asyncio
async def test_subscriber_fan_out():
    reg = JobRegistry()
    job = reg.create_job("ingest")
    events_a: list[dict] = []
    events_b: list[dict] = []

    async def collect(target):
        async for event in reg.subscribe(job.job_id):
            target.append(event)
            if event["event"] == "completed":
                break

    task_a = asyncio.create_task(collect(events_a))
    task_b = asyncio.create_task(collect(events_b))
    await asyncio.sleep(0.01)
    reg.update_progress(job.job_id, 100.0)
    reg.complete_job(job.job_id)
    await asyncio.wait_for(
        asyncio.gather(task_a, task_b),
        timeout=2.0,
    )

    assert len(events_a) == 2
    assert len(events_b) == 2


@pytest.mark.asyncio
async def test_subscriber_error_terminates():
    reg = JobRegistry()
    job = reg.create_job("ingest")
    events = []

    async def collect():
        async for event in reg.subscribe(job.job_id):
            events.append(event)

    task = asyncio.create_task(collect())
    await asyncio.sleep(0.01)
    reg.fail_job(job.job_id, "boom")
    await asyncio.wait_for(task, timeout=2.0)

    assert len(events) == 1
    assert events[0]["event"] == "error"
