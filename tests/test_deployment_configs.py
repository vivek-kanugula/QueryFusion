"""End-to-end validation of deployment configs, CORS middleware, and container scripts."""

from pathlib import Path

from fastapi.testclient import TestClient

import hybrid_rag.app as app_module
from hybrid_rag.settings import Settings

ROOT = Path(__file__).resolve().parent.parent


def test_production_app_blocks_stale_hf_cors_and_honors_configured_origins(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(_env_file=None, app_env="production", cors_origins=[]),
    )
    client = TestClient(app_module.create_app())
    resp = client.get("/health", headers={"Origin": "https://attacker.hf.space"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers

    monkeypatch.setattr(
        app_module,
        "settings",
        Settings(
            _env_file=None,
            app_env="production",
            cors_origins=["https://queryfusion.example.com"],
        ),
    )
    client_allowed = TestClient(app_module.create_app())
    resp_allowed = client_allowed.get(
        "/health", headers={"Origin": "https://queryfusion.example.com"}
    )
    assert resp_allowed.status_code == 200
    assert resp_allowed.headers.get("access-control-allow-origin") == "https://queryfusion.example.com"


def test_docker_compose_port_and_healthcheck_match_dockerfile():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "EXPOSE 8080" in dockerfile
    assert 'HF_HOME=/app/.cache/huggingface' in dockerfile
    assert 'FASTEMBED_CACHE_PATH=/app/.cache/fastembed' in dockerfile

    assert '"8080:8080"' in compose
    assert "PORT: 8080" in compose
    assert "http://localhost:8080/health" in compose
    assert '"8000:8000"' not in compose


def test_entrypoint_is_lf_and_fails_fast_on_migration_error():
    raw = (ROOT / "scripts" / "entrypoint.sh").read_bytes()
    assert b"\r\n" not in raw
    text = raw.decode("utf-8")
    assert "set -e" in text
    assert "alembic upgrade head" in text
    assert "|| echo" not in text
    assert 'PORT="${PORT:-8080}"' in text


def test_docker_run_helpers_override_windows_cache_and_temp_paths():
    ps1 = (ROOT / "scripts" / "docker_build_and_run.ps1").read_text(encoding="utf-8")
    sh_raw = (ROOT / "scripts" / "docker_build_and_run.sh").read_bytes()
    assert b"\r\n" not in sh_raw
    sh = sh_raw.decode("utf-8")

    required_overrides = [
        "APP_ENV=production",
        "PORT=8080",
        "HF_HOME=/app/.cache/huggingface",
        "FASTEMBED_CACHE_PATH=/app/.cache/fastembed",
        "TEMP=/tmp",
        "TMP=/tmp",
    ]
    for item in required_overrides:
        assert item in ps1, f"Missing {item} override in docker_build_and_run.ps1"
        assert item in sh, f"Missing {item} override in docker_build_and_run.sh"


def test_env_example_updated_for_neon_and_port_8080():
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "Neon" in env_example
    assert "DATABASE_URL_DIRECT" in env_example
    assert "AWS_ENDPOINT_URL_S3" in env_example
    assert "PORT=8080" in env_example
    assert "UVICORN_PORT=8080" in env_example
    assert "hf.space" not in env_example
    assert "Supabase" not in env_example

