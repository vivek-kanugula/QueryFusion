"""Integration test for semantic chunker (loads fastembed model)."""

import pytest

from hybrid_rag.ingest.chunkers import chunk_semantic


@pytest.mark.slow
def test_semantic_chunker_splits():
    text = (
        "Machine learning is a subset of artificial intelligence. "
        "It uses statistical methods to learn from data. "
        "Neural networks are a popular approach. "
        "They consist of layers of interconnected nodes. "
        "Cooking is a wonderful art form. "
        "Italian cuisine uses fresh ingredients like tomatoes and basil. "
        "Pasta can be made from scratch with flour and eggs. "
        "The best pizzas come from wood-fired ovens. "
        "Back to technology, databases store structured data. "
        "SQL is the standard query language for relational databases. "
    )
    chunks = chunk_semantic(text, max_tokens=500, similarity_threshold_percentile=30.0)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.text.strip()
        assert c.token_count > 0
        assert c.chunk_index >= 0


@pytest.mark.slow
def test_semantic_chunker_single_sentence():
    text = "Just one sentence here."
    chunks = chunk_semantic(text)
    assert len(chunks) == 1
    assert chunks[0].text == text


@pytest.mark.slow
def test_semantic_chunker_empty():
    assert chunk_semantic("") == []
    assert chunk_semantic("   ") == []
