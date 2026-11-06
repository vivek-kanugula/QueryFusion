"""Tests for SPA static serving with route fallback [DEPLOY-01]."""

from pathlib import Path

from fastapi.testclient import TestClient

import hybrid_rag.app as app_module


def _app_with_dist(dist_dir: Path):
    """Create fresh app with overridden DIST_DIR."""
    original = app_module.DIST_DIR
    app_module.DIST_DIR = dist_dir
    try:
        test_app = app_module.create_app()
        return TestClient(test_app)
    finally:
        app_module.DIST_DIR = original


class TestSPAServing:
    def test_index_html_at_root(self, tmp_path):
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html><body>SPA</body></html>")
        assets = dist / "assets"
        assets.mkdir()
        (assets / "main.js").write_text("console.log('app')")

        client = _app_with_dist(dist)
        response = client.get("/")
        assert response.status_code == 200
        assert "SPA" in response.text

    def test_spa_fallback_for_client_routes(self, tmp_path):
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html><body>SPA</body></html>")
        assets = dist / "assets"
        assets.mkdir()

        client = _app_with_dist(dist)

        for route in ["/qa", "/diagnostics", "/eval"]:
            response = client.get(route)
            assert response.status_code == 200
            assert "SPA" in response.text

    def test_static_asset_served_directly(self, tmp_path):
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>SPA</html>")
        assets = dist / "assets"
        assets.mkdir()
        (assets / "app.js").write_text("var x = 1;")

        client = _app_with_dist(dist)
        response = client.get("/assets/app.js")
        assert response.status_code == 200
        assert "var x = 1" in response.text

    def test_api_routes_not_caught_by_spa(self, tmp_path):
        """API routes still work when SPA mounted."""
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>SPA</html>")
        (dist / "assets").mkdir()

        client = _app_with_dist(dist)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_no_dist_dir_no_spa(self):
        """When dist/ doesn't exist, SPA fallback not mounted."""
        fake_dist = Path("/nonexistent/dist/path")
        client = _app_with_dist(fake_dist)

        response = client.get("/health")
        assert response.status_code == 200

        response = client.get("/some-spa-route")
        assert response.status_code in (404, 422)

    def test_path_traversal_attempts_return_spa_fallback(self, tmp_path):
        """Path traversal attempts (e.g. /../secret.txt) must never escape dist_dir
        and return SPA fallback."""
        root = tmp_path
        dist = root / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>SPA INDEX</html>")
        (dist / "assets").mkdir()
        (dist / "assets" / "valid.js").write_text("console.log('valid')")

        # Create sensitive files outside dist directory
        secret_file = root / "secret.env"
        secret_file.write_text("SUPER_SECRET_TOKEN=compromised")
        outer_file = root / "pyproject.toml"
        outer_file.write_text("[project]\nname = 'secret-pkg'")

        client = _app_with_dist(dist)

        spa_traversal_payloads = [
            "/..%2Fsecret.env",
            "/..%2F..%2Fsecret.env",
            "/..\\secret.env",
            "/..%2Fpyproject.toml",
            "/..%2F..%2Fpyproject.toml",
            "/qa/..%2F..%2Fsecret.env",
        ]

        for payload in spa_traversal_payloads:
            response = client.get(payload)
            assert response.status_code == 200
            assert "SPA INDEX" in response.text
            assert "SUPER_SECRET_TOKEN" not in response.text
            assert "secret-pkg" not in response.text

        # Traversal within static assets mount is blocked by StaticFiles and must not leak
        assets_traversal = client.get("/assets/..%2F..%2Fsecret.env")
        assert assets_traversal.status_code in (404, 200)
        assert "SUPER_SECRET_TOKEN" not in assets_traversal.text

    def test_malformed_and_absolute_paths_fallback_to_spa(self, tmp_path):
        """Malformed paths, null bytes, or absolute paths should gracefully return SPA fallback."""
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>SPA INDEX</html>")
        (dist / "assets").mkdir()

        client = _app_with_dist(dist)

        for path in ["/%00", "/..%00/secret", "/invalid|char<path>", "///double-slash"]:
            response = client.get(path)
            assert response.status_code == 200
            assert "SPA INDEX" in response.text
