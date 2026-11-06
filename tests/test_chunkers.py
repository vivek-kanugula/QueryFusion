"""Tests for tri-chunking engine."""

from hybrid_rag.ingest.chunkers import (
    chunk_fixed,
    chunk_semantic,
    chunk_structure_aware,
    count_tokens,
)


def test_count_tokens_nonempty():
    count = count_tokens("Hello world, this is a test sentence.")
    assert count > 0


def test_chunk_fixed_basic():
    text = "This is sentence one. This is sentence two. This is sentence three. " * 20
    chunks = chunk_fixed(text, token_size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count > 0
        assert c.text


def test_chunk_fixed_ordering():
    text = "Word " * 200
    chunks = chunk_fixed(text, token_size=30, overlap=5)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_chunk_fixed_overlap():
    text = "Word " * 200
    chunks = chunk_fixed(text, token_size=30, overlap=10)
    assert len(chunks) > 1
    if len(chunks) >= 2:
        assert chunks[0].char_offset_end > chunks[1].char_offset_start


def test_chunk_fixed_empty():
    assert chunk_fixed("") == []
    assert chunk_fixed("   ") == []


def test_chunk_fixed_small_text():
    chunks = chunk_fixed("Hello.", token_size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_chunk_structure_aware_basic():
    text = (
        "# Header One\n\nFirst paragraph with content."
        "\n\n# Header Two\n\nSecond paragraph with content."
    )
    chunks = chunk_structure_aware(text, max_tokens=500)
    assert len(chunks) >= 2
    headings = [c.section_heading for c in chunks if c.section_heading]
    assert "Header One" in headings


def test_chunk_structure_aware_with_sections():
    sections = [
        {"text": "First section content.", "heading": "Intro"},
        {"text": "Second section content.", "heading": "Body"},
    ]
    text = "First section content.\n\nSecond section content."
    chunks = chunk_structure_aware(text, max_tokens=500, sections=sections)
    assert len(chunks) == 2


def test_chunk_structure_aware_long_block():
    long_text = "This is a long sentence for testing. " * 100
    text = f"# Title\n\n{long_text}"
    chunks = chunk_structure_aware(text, max_tokens=50)
    assert len(chunks) > 1


def test_chunk_offsets_valid():
    text = "Hello world. " * 50
    chunks = chunk_fixed(text, token_size=20, overlap=5)
    for c in chunks:
        assert c.char_offset_start >= 0
        assert c.char_offset_end > c.char_offset_start
        assert c.char_offset_end <= len(text)


def test_chunk_offset_fidelity_irregular_whitespace():
    """Verify chunk text matches document slice with 100% fidelity on irregular whitespace."""
    text = (
        "Paragraph 1 sentence 1.   Paragraph 1 sentence 2.\n\n\n\t"
        "Paragraph 2 sentence 3. \n\nParagraph 2 sentence 4."
    )

    # Test structure-aware chunking
    struct_chunks = chunk_structure_aware(text, max_tokens=10)
    assert len(struct_chunks) >= 2
    for c in struct_chunks:
        assert text[c.char_offset_start : c.char_offset_end] == c.text
        assert not c.text.startswith(" ")
        assert not c.text.endswith(" ")

    # Test semantic chunking
    sem_chunks = chunk_semantic(text, max_tokens=10)
    assert len(sem_chunks) >= 2
    for c in sem_chunks:
        assert text[c.char_offset_start : c.char_offset_end] == c.text


def test_semantic_chunker_reuses_embedder_session():
    """Verify chunk_semantic reuses get_embedder singleton instead of instantiating per call."""
    from unittest.mock import patch

    import fastembed

    from hybrid_rag.ingest.embedder import get_embedder

    init_count = 0
    orig_init = fastembed.TextEmbedding.__init__

    def tracked_init(self, *args, **kwargs):
        nonlocal init_count
        init_count += 1
        orig_init(self, *args, **kwargs)

    # Warm up singleton first
    get_embedder()

    with patch.object(fastembed.TextEmbedding, "__init__", tracked_init):
        chunk_semantic("Sentence one. Sentence two. Sentence three.")
        chunk_semantic("Sentence four. Sentence five. Sentence six.")

    # With session reuse, TextEmbedding.__init__ should not be called again
    assert init_count == 0


def test_semantic_chunker_custom_embedder_injection():
    """Verify chunk_semantic accepts an injected embedder."""
    from unittest.mock import MagicMock

    import numpy as np

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [np.ones(384), np.zeros(384)]

    text = "Sentence alpha. Sentence beta."
    chunks = chunk_semantic(text, embedder=mock_embedder)
    assert len(chunks) >= 1
    mock_embedder.embed.assert_called_once()
