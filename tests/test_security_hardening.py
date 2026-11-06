"""Security and Path Traversal Hardening Test Suite.

Verifies:
- Path Traversal defense in SPA fallback handler (app.py)
- API endpoint security and fail-closed handling
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import hybrid_rag.app as app_module
from hybrid_rag.app import app
from hybrid_rag.db.engine import get_db


@pytest.fixture
def mock_db():
    mock_session = MagicMock()
    mock_session.query.return_value.order_by.return_value.all.return_value = []
    app.dependency_overrides[get_db] = lambda: mock_session
    yield mock_session
    app.dependency_overrides.pop(get_db, None)


def _app_with_dist(dist_dir: Path):
    original = app_module.DIST_DIR
    app_module.DIST_DIR = dist_dir
    try:
        test_app = app_module.create_app()
        return TestClient(test_app)
    finally:
        app_module.DIST_DIR = original


class TestAdversarialPathTraversal:
    """Stress-test SPA fallback handler against aggressive path traversal vectors."""

    @pytest.fixture
    def traversal_sandbox(self, tmp_path):
        root = tmp_path
        dist = root / "dist"
        dist.mkdir()
        (dist / "index.html").write_text(
            "<!DOCTYPE html><html><body>SPA_INDEX_CANARY_7749</body></html>"
        )
        assets = dist / "assets"
        assets.mkdir()
        (assets / "app.js").write_text("console.log('valid_asset_marker_42');")

        # Plant sensitive secret files outside dist_dir
        secret_env = root / ".env"
        secret_env.write_text(
            "DATABASE_URL=postgresql://secret_user:secret_pass@db.example.com/prod\n"
            "API_SECRET_KEY=adversarial_canary_leaked"
        )
        pyproject = root / "pyproject.toml"
        toml_content = "[project]\nname = 'confidential-package-canary'"
        pyproject.write_text(toml_content)
        nested_dir = root / "nested" / "deep"
        nested_dir.mkdir(parents=True)
        (nested_dir / "private.key").write_text(
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEogIBAAKCAQEAnCanaryKey"
        )

        return {
            "root": root,
            "dist": dist,
            "secret_env": secret_env,
            "pyproject": pyproject,
        }

    @pytest.mark.parametrize(
        "payload_description,payload",
        [
            # URL encoded standard traversals
            ("URL encoded dot-dot-slash .env", "/..%2f..%2f.env"),
            ("URL encoded dot-dot-slash pyproject", "/..%2f..%2fpyproject.toml"),
            ("URL encoded double-dot encoded", "/%2e%2e%2f%2e%2e%2fpyproject.toml"),
            ("URL encoded uppercase hex", "/%2E%2E%2F%2E%2E%2F.env"),
            ("URL encoded backslash", "/..%5c..%5c.env"),
            ("URL encoded double-dot backslash", "/%2e%2e%5c%2e%2e%5c.env"),
            ("URL encoded uppercase backslash", "/%2E%2E%5C%2E%2E%5C.env"),
            # Double URL encoded traversals
            ("Double encoded dot-dot", "/%252e%252e%252f"),
            ("Double encoded full path", "/%252e%252e%252f%252e%252e%252f.env"),
            ("Double encoded backslash full path", "/%252e%252e%255c%252e%252e%255c.env"),
            # Null bytes
            ("Null byte traversal to test", "/..%00/test"),
            ("Null byte traversal to .env", "/..%00/.env"),
            ("Null byte at root", "/%00"),
            ("Null byte in html extension", "/index.html%00.txt"),
            # Nested directory traversals
            ("Assets mount traversal /assets/..%2f", "/assets/..%2f..%2f.env"),
            ("Assets mount backslash traversal", "/assets/..%5c..%5c.env"),
            ("Assets mount encoded traversal", "/assets/%2e%2e%2f%2e%2e%2f.env"),
            ("Client route traversal /qa/..%2f", "/qa/..%2f..%2f.env"),
            ("Client route traversal /eval/..%2f", "/eval/..%2f..%2f.env"),
            ("Deep path traversal", "/deep/nested/sub/..%2f..%2f..%2f.env"),
            # Absolute Windows paths & Drive Letters
            ("Windows drive letter forward slash", "/C:/Windows/win.ini"),
            ("Windows drive letter backslash", "/C:\\Windows\\win.ini"),
            ("Windows system32 hosts file", "/c:/windows/system32/drivers/etc/hosts"),
            ("UNC path forward slash", "//localhost/c$/Windows/win.ini"),
            ("UNC path backslash", "\\\\localhost\\c$\\Windows\\win.ini"),
            # Complex / Edge cases
            ("Multiple consecutive dots", "/....//....//.env"),
            ("Triple dots", "/.../.../.env"),
            ("Dot-encoded dot-slash", "/.%2e/.%2e/.env"),
            ("Semicolon traversal variant", "/..;/..;/.env"),
            ("Space padded dots", "/%20..%20/..%20/.env"),
            ("Direct root file name .env", "/.env"),
            ("Direct root file name pyproject.toml", "/pyproject.toml"),
        ],
    )
    def test_adversarial_traversal_never_leaks_and_serves_spa_or_error(
        self, traversal_sandbox, payload_description, payload
    ):
        """None of the traversal payloads should ever read files outside dist_dir."""
        client = _app_with_dist(traversal_sandbox["dist"])
        response = client.get(payload)

        # Status must be valid (200 SPA fallback or 404 from staticfiles mount)
        assert response.status_code in (200, 404), (
            f"Failed on {payload_description} ({payload}): unexpected status {response.status_code}"
        )

        # Response content must NEVER contain secrets or private files outside dist
        response_text = response.text
        assert "adversarial_canary_leaked" not in response_text, (
            f"LEAK DETECTED for {payload_description}: {payload}"
        )
        assert "confidential-package-canary" not in response_text, (
            f"LEAK DETECTED for {payload_description}: {payload}"
        )
        assert "MIIEogIBAAKCAQEAnCanaryKey" not in response_text, (
            f"LEAK DETECTED for {payload_description}: {payload}"
        )
        assert "[fonts]" not in response_text.lower(), (
            f"WINDOWS FILE LEAK for {payload_description}: {payload}"
        )
        assert "[extensions]" not in response_text.lower(), (
            f"WINDOWS FILE LEAK for {payload_description}: {payload}"
        )

        # If status is 200, it must be the legitimate SPA index fallback
        if response.status_code == 200 and not payload.startswith("/assets/app.js"):
            assert "SPA_INDEX_CANARY_7749" in response_text

    def test_legitimate_assets_still_accessible(self, traversal_sandbox):
        """Legitimate static assets inside dist_dir remain accessible."""
        client = _app_with_dist(traversal_sandbox["dist"])

        # Root index
        resp_root = client.get("/")
        assert resp_root.status_code == 200
        assert "SPA_INDEX_CANARY_7749" in resp_root.text

        # Direct asset
        resp_asset = client.get("/assets/app.js")
        assert resp_asset.status_code == 200
        assert "valid_asset_marker_42" in resp_asset.text

        # Valid client-side route
        resp_route = client.get("/qa/overview")
        assert resp_route.status_code == 200
        assert "SPA_INDEX_CANARY_7749" in resp_route.text
