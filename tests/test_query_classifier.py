"""Unit tests for query difficulty classifier and alpha recommendation."""

from hybrid_rag.retrieval.query_classifier import (
    CATEGORY_ALPHA,
    QueryCategory,
    classify_query,
)


def test_classify_empty_or_whitespace_query():
    """Empty or whitespace queries must default to factual with 0.5 alpha."""
    res1 = classify_query("")
    assert res1.category == QueryCategory.FACTUAL
    assert res1.alpha == 0.5
    assert res1.confidence == 0.5

    res2 = classify_query("   \n\t  ")
    assert res2.category == QueryCategory.FACTUAL
    assert res2.alpha == 0.5


def test_classify_exact_match_quoted_terms():
    """Queries with quoted terms should classify as EXACT_MATCH (alpha=0.2)."""
    res = classify_query('What does "ConnectionRefusedError" mean in production?')
    assert res.category == QueryCategory.EXACT_MATCH
    assert res.alpha == CATEGORY_ALPHA[QueryCategory.EXACT_MATCH]
    assert "quoted terms" in res.reason or "error code" in res.reason


def test_classify_exact_match_uuid_and_hex():
    """Queries with UUIDs or hex identifiers should classify as EXACT_MATCH."""
    uuid_query = "Find trace 12345678-1234-5678-1234-567812345678 in server logs"
    res = classify_query(uuid_query)
    assert res.category == QueryCategory.EXACT_MATCH
    assert res.alpha == 0.2

    hex_query = "What failed at address 0xdeadbeef?"
    res_hex = classify_query(hex_query)
    assert res_hex.category == QueryCategory.EXACT_MATCH


def test_classify_exact_match_version_numbers():
    """Queries with exact version numbers and error types."""
    res = classify_query('Why did "NullPointerException" happen in v2.4.1?')
    assert res.category == QueryCategory.EXACT_MATCH
    assert res.alpha == 0.2


def test_classify_factual_short_queries():
    """Simple factual questions (who/what/when/where) should classify as FACTUAL (alpha=0.5)."""
    res1 = classify_query("Who is the CEO of Anthropic?")
    assert res1.category == QueryCategory.FACTUAL
    assert res1.alpha == 0.5

    res2 = classify_query("What is the capital of France?")
    assert res2.category == QueryCategory.FACTUAL
    assert res2.alpha == 0.5

    res3 = classify_query("Where is PostgreSQL configuration stored?")
    assert res3.category == QueryCategory.FACTUAL
    assert res3.alpha == 0.5


def test_classify_multi_hop_comparison():
    """Questions comparing concepts with conjunctions should classify as MULTI_HOP (alpha=0.6)."""
    res = classify_query("Compare the difference between PostgreSQL and MongoDB for vector search")
    assert res.category == QueryCategory.MULTI_HOP
    assert res.alpha == 0.6
    assert "comparison" in res.reason.lower() or "multiple concepts" in res.reason.lower()

    res2 = classify_query("How does indexing affect query latency versus memory consumption?")
    assert res2.category == QueryCategory.MULTI_HOP
    assert res2.alpha == 0.6


def test_classify_semantic_conceptual():
    """Conceptual, architectural, or explanatory queries should classify as SEMANTIC (alpha=0.7)."""
    res = classify_query("Explain the underlying architecture and trade-offs of hybrid RAG")
    assert res.category == QueryCategory.SEMANTIC
    assert res.alpha == 0.7
    assert "conceptual" in res.reason.lower() or "explanatory" in res.reason.lower()

    res2 = classify_query("Describe the philosophy and paradigm of microservices design patterns")
    assert res2.category == QueryCategory.SEMANTIC
    assert res2.alpha == 0.7


def test_classify_default_fallback_without_strong_signals():
    """Queries without strong markers should fall back to factual gracefully."""
    res = classify_query("hello world")
    assert res.category in (QueryCategory.FACTUAL, QueryCategory.SEMANTIC)
    assert 0.0 <= res.confidence <= 1.0
    assert res.alpha in CATEGORY_ALPHA.values()

