"""Tests for golden dataset loading, span mapper, dev/test split, and dataset hash."""

import pytest

from hybrid_rag.eval.dataset import (
    ChunkInfo,
    GoldenItem,
    SourceSpan,
    dataset_hash,
    load_golden_dataset,
    map_spans_to_chunks,
    save_golden_dataset,
    span_overlap_ratio,
    split_dev_test,
)

# ── Span overlap ratio ──


def test_overlap_full():
    assert span_overlap_ratio(100, 200, 100, 200) == pytest.approx(1.0)


def test_overlap_none():
    assert span_overlap_ratio(100, 200, 300, 400) == pytest.approx(0.0)


def test_overlap_half():
    assert span_overlap_ratio(100, 200, 150, 250) == pytest.approx(0.5)


def test_overlap_chunk_inside_span():
    assert span_overlap_ratio(100, 300, 150, 200) == pytest.approx(50 / 200)


def test_overlap_span_inside_chunk():
    assert span_overlap_ratio(150, 200, 100, 300) == pytest.approx(1.0)


def test_overlap_zero_length_span():
    assert span_overlap_ratio(100, 100, 50, 150) == pytest.approx(0.0)


def test_overlap_adjacent_no_overlap():
    assert span_overlap_ratio(100, 200, 200, 300) == pytest.approx(0.0)


def test_overlap_one_char():
    assert span_overlap_ratio(100, 200, 199, 300) == pytest.approx(1 / 100)


# ── Span-to-chunk mapper ──


def _make_chunks() -> list[ChunkInfo]:
    return [
        ChunkInfo(chunk_id="c1", document_id="doc-a", char_offset_start=0, char_offset_end=200),
        ChunkInfo(chunk_id="c2", document_id="doc-a", char_offset_start=180, char_offset_end=400),
        ChunkInfo(chunk_id="c3", document_id="doc-a", char_offset_start=380, char_offset_end=600),
        ChunkInfo(chunk_id="c4", document_id="doc-b", char_offset_start=0, char_offset_end=300),
    ]


def test_map_spans_single_match():
    spans = [SourceSpan(document_id="doc-a", start_char=50, end_char=150)]
    result = map_spans_to_chunks(spans, _make_chunks(), overlap_threshold=0.5)
    assert result == {"c1"}


def test_map_spans_multi_chunk():
    spans = [SourceSpan(document_id="doc-a", start_char=100, end_char=300)]
    result = map_spans_to_chunks(spans, _make_chunks(), overlap_threshold=0.3)
    assert "c1" in result
    assert "c2" in result


def test_map_spans_wrong_document():
    spans = [SourceSpan(document_id="doc-c", start_char=0, end_char=100)]
    result = map_spans_to_chunks(spans, _make_chunks())
    assert result == set()


def test_map_spans_empty_spans():
    result = map_spans_to_chunks([], _make_chunks())
    assert result == set()


def test_map_spans_empty_chunks():
    spans = [SourceSpan(document_id="doc-a", start_char=0, end_char=100)]
    result = map_spans_to_chunks(spans, [])
    assert result == set()


def test_map_spans_cross_document():
    spans = [
        SourceSpan(document_id="doc-a", start_char=50, end_char=150),
        SourceSpan(document_id="doc-b", start_char=50, end_char=250),
    ]
    result = map_spans_to_chunks(spans, _make_chunks(), overlap_threshold=0.5)
    assert "c1" in result
    assert "c4" in result


def test_map_spans_threshold_boundary():
    spans = [SourceSpan(document_id="doc-a", start_char=0, end_char=200)]
    result_low = map_spans_to_chunks(spans, _make_chunks(), overlap_threshold=0.05)
    result_high = map_spans_to_chunks(spans, _make_chunks(), overlap_threshold=0.9)
    assert len(result_low) >= len(result_high)


# ── Dataset hash ──


def test_dataset_hash_deterministic():
    items = [
        GoldenItem(id="q1", question="Q?", expected_answer="A", category="cat", source_spans=[]),
        GoldenItem(id="q2", question="R?", expected_answer="B", category="cat", source_spans=[]),
    ]
    h1 = dataset_hash(items)
    h2 = dataset_hash(items)
    assert h1 == h2
    assert len(h1) == 64


def test_dataset_hash_order_independent():
    items_a = [
        GoldenItem(id="q1", question="Q?", expected_answer="A", category="cat"),
        GoldenItem(id="q2", question="R?", expected_answer="B", category="cat"),
    ]
    items_b = list(reversed(items_a))
    assert dataset_hash(items_a) == dataset_hash(items_b)


def test_dataset_hash_different_content():
    items_a = [GoldenItem(id="q1", question="Q?", expected_answer="A", category="cat")]
    items_b = [GoldenItem(id="q1", question="Different?", expected_answer="A", category="cat")]
    assert dataset_hash(items_a) != dataset_hash(items_b)


# ── Dev/test split ──


def _make_golden(n: int) -> list[GoldenItem]:
    return [
        GoldenItem(
            id=f"q{i}",
            question=f"Q{i}?",
            expected_answer="A",
            category="cat",
        )
        for i in range(n)
    ]


def test_split_deterministic():
    items = _make_golden(50)
    dev1, test1 = split_dev_test(items)
    dev2, test2 = split_dev_test(items)
    assert [x.id for x in dev1] == [x.id for x in dev2]
    assert [x.id for x in test1] == [x.id for x in test2]


def test_split_covers_all():
    items = _make_golden(50)
    dev, test = split_dev_test(items)
    assert len(dev) + len(test) == 50
    all_ids = {x.id for x in dev} | {x.id for x in test}
    assert len(all_ids) == 50


def test_split_no_overlap():
    items = _make_golden(50)
    dev, test = split_dev_test(items)
    dev_ids = {x.id for x in dev}
    test_ids = {x.id for x in test}
    assert dev_ids & test_ids == set()


def test_split_roughly_balanced():
    items = _make_golden(100)
    dev, test = split_dev_test(items, dev_ratio=0.5)
    assert 30 <= len(dev) <= 70


# ── Save / load roundtrip ──


def test_save_load_roundtrip(tmp_path):
    items = [
        GoldenItem(
            id="q1",
            question="Q?",
            expected_answer="A",
            category="exact-identifier",
            source_spans=[SourceSpan(document_id="doc-a", start_char=10, end_char=50)],
        ),
    ]
    path = tmp_path / "test_golden.json"
    save_golden_dataset(items, path)
    loaded = load_golden_dataset(path)
    assert len(loaded) == 1
    assert loaded[0].id == "q1"
    assert loaded[0].source_spans[0].document_id == "doc-a"
    assert loaded[0].source_spans[0].start_char == 10


# ── Load bundled dataset ──


def test_load_bundled_golden_dataset():
    items = load_golden_dataset()
    assert len(items) >= 50
    categories = {item.category for item in items}
    assert len(categories) == 5
    for item in items:
        assert item.id
        assert item.question
        assert item.category
