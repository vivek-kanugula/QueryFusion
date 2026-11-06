"""Golden evaluation dataset with span-based ground truth and dynamic span-to-chunk mapper."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceSpan:
    document_id: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    expected_answer: str
    category: str
    source_spans: list[SourceSpan] = field(default_factory=list)


# ── Loading / saving ──


def load_golden_dataset(path: Path | None = None) -> list[GoldenItem]:
    if path is None:
        path = Path(__file__).parent / "golden_questions.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [_parse_item(item) for item in raw]


def _parse_item(d: dict) -> GoldenItem:
    spans = [
        SourceSpan(
            document_id=s["document_id"],
            start_char=s["start_char"],
            end_char=s["end_char"],
        )
        for s in d.get("source_spans", [])
    ]
    return GoldenItem(
        id=d["id"],
        question=d["question"],
        expected_answer=d["expected_answer"],
        category=d["category"],
        source_spans=spans,
    )


def save_golden_dataset(items: list[GoldenItem], path: Path) -> None:
    data = [
        {
            "id": item.id,
            "question": item.question,
            "expected_answer": item.expected_answer,
            "category": item.category,
            "source_spans": [
                {
                    "document_id": s.document_id,
                    "start_char": s.start_char,
                    "end_char": s.end_char,
                }
                for s in item.source_spans
            ],
        }
        for item in items
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Dataset hash (SHA-256 freeze) ──


def dataset_hash(items: list[GoldenItem]) -> str:
    canonical = json.dumps(
        [
            {
                "id": item.id,
                "question": item.question,
                "category": item.category,
                "spans": [(s.document_id, s.start_char, s.end_char) for s in item.source_spans],
            }
            for item in sorted(items, key=lambda x: x.id)
        ],
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Dev / test split ──


def split_dev_test(
    items: list[GoldenItem],
    dev_ratio: float = 0.5,
    seed_salt: str = "hybrid-rag-v1",
) -> tuple[list[GoldenItem], list[GoldenItem]]:
    dev: list[GoldenItem] = []
    test: list[GoldenItem] = []
    for item in items:
        h = hashlib.sha256(f"{seed_salt}:{item.id}".encode()).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        if bucket < dev_ratio:
            dev.append(item)
        else:
            test.append(item)
    return dev, test


# ── Span-to-chunk overlap mapper ──


def span_overlap_ratio(span_start: int, span_end: int, chunk_start: int, chunk_end: int) -> float:
    span_length = span_end - span_start
    if span_length <= 0:
        return 0.0
    intersection = max(0, min(span_end, chunk_end) - max(span_start, chunk_start))
    return intersection / span_length


@dataclass(frozen=True)
class ChunkInfo:
    chunk_id: str
    document_id: str
    char_offset_start: int
    char_offset_end: int
    document_name: str | None = None


def _normalize_doc_id(val: str | None) -> str:
    """Normalize document identifier or filename into a canonical slug."""
    if not val:
        return ""
    s = val.strip().lower()
    if "." in s:
        s = s.rsplit(".", 1)[0]
    if s.startswith("doc-"):
        s = s[4:]
    elif s.startswith("doc_"):
        s = s[4:]
    return s.replace("_", "-")


def _matches_document(
    span_doc_id: str,
    chunk: ChunkInfo,
    doc_id_map: dict[str, str] | None = None,
) -> bool:
    """Multi-tier document resolution between span document ID and chunk."""
    # Tier 1: Direct ID match
    if chunk.document_id == span_doc_id:
        return True

    # Tier 2: Explicit doc_id_map lookup
    if doc_id_map:
        if doc_id_map.get(span_doc_id) == chunk.document_id:
            return True
        if doc_id_map.get(chunk.document_id) == span_doc_id:
            return True
        norm_span = _normalize_doc_id(span_doc_id)
        if doc_id_map.get(norm_span) == chunk.document_id:
            return True
        norm_chunk = _normalize_doc_id(chunk.document_id)
        if doc_id_map.get(norm_chunk) == span_doc_id:
            return True

    # Tier 3: document_name comparison (if available)
    if chunk.document_name:
        if chunk.document_name == span_doc_id:
            return True
        if _normalize_doc_id(chunk.document_name) == _normalize_doc_id(span_doc_id):
            return True

    # Tier 4: Slug normalization between chunk.document_id and span_doc_id
    norm_span = _normalize_doc_id(span_doc_id)
    norm_chunk = _normalize_doc_id(chunk.document_id)
    if norm_span and norm_chunk and norm_span == norm_chunk:
        return True

    return False


def build_doc_id_map(db: Any) -> dict[str, str]:
    """Build a mapping from canonical slugs and filenames to document UUIDs.

    Accepts a SQLAlchemy Session or an iterable of Document objects/dictionaries.
    """
    doc_id_map: dict[str, str] = {}
    docs = []
    if hasattr(db, "query"):
        from hybrid_rag.models.document import Document

        docs = db.query(Document).all()
    elif isinstance(db, (list, tuple, set)):
        docs = list(db)
    elif db is not None:
        docs = [db]

    for doc in docs:
        doc_id = str(getattr(doc, "id", "") or (doc.get("id") if isinstance(doc, dict) else ""))
        filename = getattr(doc, "filename", "") or (
            doc.get("filename") if isinstance(doc, dict) else ""
        )
        if not doc_id:
            continue

        doc_id_map[doc_id] = doc_id
        if filename:
            doc_id_map[filename] = doc_id
            norm = _normalize_doc_id(filename)
            if norm:
                doc_id_map[norm] = doc_id
                doc_id_map[f"doc-{norm}"] = doc_id
                doc_id_map[f"doc_{norm}"] = doc_id

    return doc_id_map


def map_spans_to_chunks(
    spans: list[SourceSpan],
    chunks: list[ChunkInfo],
    overlap_threshold: float = 0.5,
    doc_id_map: dict[str, str] | None = None,
) -> set[str]:
    relevant_chunk_ids: set[str] = set()
    for span in spans:
        for chunk in chunks:
            if not _matches_document(span.document_id, chunk, doc_id_map=doc_id_map):
                continue
            ratio = span_overlap_ratio(
                span.start_char,
                span.end_char,
                chunk.char_offset_start,
                chunk.char_offset_end,
            )
            if ratio >= overlap_threshold:
                relevant_chunk_ids.add(chunk.chunk_id)
    return relevant_chunk_ids
