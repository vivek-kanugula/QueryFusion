"""Tests for identifier-preserving BM25 tokenizer (20+ technical identifiers)."""

import re

import pytest

from hybrid_rag.retrieval.tokenizer import (
    expand_technical_tokens,
    tokenize_technical,
)


def _tokens(text: str) -> set[str]:
    """Extract lowercased alphanumeric tokens from expanded text."""
    return set(re.findall(r"[a-z0-9]+", expand_technical_tokens(text).lower()))


@pytest.mark.parametrize(
    "input_text, expected_composite, expected_parts",
    [
        ("DB-204", "db204", ["db", "204"]),
        ("PG_TRGM", "pgtrgm", ["pg", "trgm"]),
        ("error_code_500", "errorcode500", ["error", "code", "500"]),
        ("config.json", "configjson", ["config", "json"]),
        ("RFC-7231", "rfc7231", ["rfc", "7231"]),
        ("UTF-8", "utf8", ["utf", "8"]),
        ("x86-64", "x8664", ["x86", "64"]),
        ("TCP/IP", "tcpip", ["tcp", "ip"]),
        ("v2.0.1", "v201", ["v2", "0", "1"]),
        ("api/v1", "apiv1", ["api", "v1"]),
        ("sys.path", "syspath", ["sys", "path"]),
        ("MAX_RETRIES", "maxretries", ["max", "retries"]),
        ("node_modules", "nodemodules", ["node", "modules"]),
        ("docker-compose", "dockercompose", ["docker", "compose"]),
        ("git-bisect", "gitbisect", ["git", "bisect"]),
        ("k8s-cluster", "k8scluster", ["k8s", "cluster"]),
        ("file.tar.gz", "filetargz", ["file", "tar", "gz"]),
        ("192.168.1.1", "19216811", ["192", "168", "1"]),
        ("my-app.config", "myappconfig", ["my", "app", "config"]),
        ("UPPER_CASE_ID", "uppercaseid", ["upper", "case", "id"]),
        ("std::vector", "stdvector", ["std", "vector"]),
        ("path/to/file", "pathtofile", ["path", "to", "file"]),
    ],
    ids=lambda val: val if isinstance(val, str) and "/" not in val else None,
)
def test_expand_technical_token(input_text, expected_composite, expected_parts):
    tokens = _tokens(input_text)
    assert expected_composite in tokens, f"{expected_composite} not in {tokens}"
    for part in expected_parts:
        assert part in tokens, f"sub-token '{part}' not in {tokens}"


def test_plain_text_unchanged():
    text = "The quick brown fox jumps"
    expanded = expand_technical_tokens(text)
    assert expanded == text


def test_mixed_text_expands_only_technical():
    text = "Error DB-204 occurred in config.json file"
    expanded = expand_technical_tokens(text).lower()
    assert "db204" in expanded
    assert "configjson" in expanded
    assert "occurred" in expanded
    assert "file" in expanded


def test_empty_string():
    assert expand_technical_tokens("") == ""


def test_tokenize_technical_stems():
    tokens = tokenize_technical("searching databases efficiently")
    assert "search" in tokens
    assert "databas" in tokens
    assert "effici" in tokens


def test_tokenize_technical_removes_stopwords():
    tokens = tokenize_technical("the error is in the config")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "in" not in tokens


def test_tokenize_technical_preserves_composite():
    tokens = tokenize_technical("DB-204")
    assert "db204" in tokens
