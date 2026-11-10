"""Global pytest configuration and fixtures for Hybrid-RAG test suites."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _setup_test_environment():
    """Ensure mock credentials and test variables are configured across test sessions."""
    if not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = "gsk-mock-key-for-tests"
