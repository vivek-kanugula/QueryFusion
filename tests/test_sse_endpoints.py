"""Tests for SSE endpoints: job status, cancel, event streaming, heartbeat."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from hybrid_rag.api.jobs import router
from hybrid_rag.core.jobs import JobRegistry


@pytest.fixture()
def registry():
    return JobRegistry()


@pytest.fixture()
def client(registry):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    with patch("hybrid_rag.api.jobs.job_registry", registry):
        yield TestClient(app)


def test_get_job_status(client, registry):
    job = registry.create_job("ingest")
    resp = client.get(f"/api/jobs/{job.job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job.job_id
    assert data["status"] == "PENDING"


def test_get_job_not_found(client):
    resp = client.get("/api/jobs/nonexistent")
    assert resp.status_code == 404


def test_get_job_with_progress(client, registry):
    job = registry.create_job("ingest")
    registry.update_progress(job.job_id, 75.0, "Processing")
    resp = client.get(f"/api/jobs/{job.job_id}")
    data = resp.json()
    assert data["status"] == "RUNNING"
    assert data["progress_percent"] == 75.0
    assert data["status_message"] == "Processing"


def test_cancel_job(client, registry):
    job = registry.create_job("ingest")
    resp = client.post(f"/api/jobs/{job.job_id}/cancel")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "CANCELLED"


def test_cancel_job_not_found(client):
    resp = client.post("/api/jobs/nonexistent/cancel")
    assert resp.status_code == 404


def test_cancel_completed_job_409(client, registry):
    job = registry.create_job("ingest")
    registry.complete_job(job.job_id)
    resp = client.post(f"/api/jobs/{job.job_id}/cancel")
    assert resp.status_code == 409


def _fire_after_delay(registry, job_id, actions, delay=0.1):
    def _fire():
        time.sleep(delay)
        for action, args in actions:
            getattr(registry, action)(job_id, *args)

    threading.Thread(target=_fire, daemon=True).start()


def test_sse_event_format(client, registry):
    """SSE events follow event: type\\ndata: json\\n\\n format."""
    job = registry.create_job("ingest")
    _fire_after_delay(
        registry,
        job.job_id,
        [
            ("update_progress", (50.0, "half")),
            ("complete_job", ()),
        ],
    )

    with client.stream("GET", f"/api/jobs/{job.job_id}/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = b"".join(resp.iter_bytes()).decode()

    assert "event: progress" in raw
    assert "event: completed" in raw
    blocks = [b for b in raw.split("\n\n") if b.strip()]
    for block in blocks:
        lines = block.strip().split("\n")
        assert any(ln.startswith("event: ") for ln in lines)
        assert any(ln.startswith("data: ") for ln in lines)


def test_sse_events_not_found(client):
    resp = client.get("/api/jobs/nonexistent/events")
    assert resp.status_code == 404


def test_sse_stream_terminates_on_complete(client, registry):
    job = registry.create_job("ingest")
    _fire_after_delay(registry, job.job_id, [("complete_job", ())])

    with client.stream("GET", f"/api/jobs/{job.job_id}/events") as resp:
        chunks = list(resp.iter_bytes())

    combined = b"".join(chunks).decode()
    assert "event: completed" in combined


def test_sse_stream_terminates_on_error(client, registry):
    job = registry.create_job("ingest")
    _fire_after_delay(
        registry,
        job.job_id,
        [
            ("fail_job", ("boom",)),
        ],
    )

    with client.stream("GET", f"/api/jobs/{job.job_id}/events") as resp:
        combined = b"".join(resp.iter_bytes()).decode()

    assert "event: error" in combined
    assert "boom" in combined


def test_sse_data_is_valid_json(client, registry):
    job = registry.create_job("ingest")
    _fire_after_delay(
        registry,
        job.job_id,
        [
            ("update_progress", (42.0, "chunking")),
            ("complete_job", ({"chunks": 10},)),
        ],
    )

    with client.stream("GET", f"/api/jobs/{job.job_id}/events") as resp:
        raw = b"".join(resp.iter_bytes()).decode()

    blocks = [b for b in raw.split("\n\n") if b.strip()]
    for block in blocks:
        for line in block.strip().split("\n"):
            if line.startswith("data: "):
                payload = line[len("data: ") :]
                parsed = json.loads(payload)
                assert isinstance(parsed, dict)


def test_sse_heartbeat_interval_constant():
    from hybrid_rag.api.jobs import HEARTBEAT_INTERVAL

    assert HEARTBEAT_INTERVAL == 15


def test_sse_headers(client, registry):
    job = registry.create_job("ingest")
    _fire_after_delay(registry, job.job_id, [("complete_job", ())])

    with client.stream("GET", f"/api/jobs/{job.job_id}/events") as resp:
        list(resp.iter_bytes())
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"


def test_sse_late_subscriber_completed_job(client, registry):
    """Verify subscribing to completed job terminates immediately with completed event."""
    job = registry.create_job("ingest")
    registry.complete_job(job.job_id, {"result_info": "done"})

    with client.stream("GET", f"/api/jobs/{job.job_id}/events") as resp:
        assert resp.status_code == 200
        raw = b"".join(resp.iter_bytes()).decode()

    assert "event: completed" in raw
    assert "done" in raw


def test_sse_late_subscriber_failed_job(client, registry):
    """Verify subscribing to an already failed job terminates immediately with error event."""
    job = registry.create_job("ingest")
    registry.fail_job(job.job_id, "disk error")

    with client.stream("GET", f"/api/jobs/{job.job_id}/events") as resp:
        assert resp.status_code == 200
        raw = b"".join(resp.iter_bytes()).decode()

    assert "event: error" in raw
    assert "disk error" in raw
