"""Tri-chunking engine: Fixed, Structure-Aware, and Semantic chunkers.

Each chunker takes full text and returns list[ChunkOutput].
Chunk IDs are assigned externally via make_chunk_id.
Token counting uses the BGE tokenizer for consistency with the embedder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tokenizers import Tokenizer


@dataclass
class ChunkOutput:
    text: str
    chunk_index: int
    token_count: int
    page_number: int | None = None
    section_heading: str | None = None
    char_offset_start: int = 0
    char_offset_end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SentenceSpan:
    text: str
    start: int
    end: int


_tokenizer: Tokenizer | None = None


def get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    return _tokenizer


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False).ids)


def _split_sentences_with_spans(text: str) -> list[SentenceSpan]:
    if not text:
        return []
    spans: list[SentenceSpan] = []
    last_pos = 0
    pattern = re.compile(r"(?<=[.!?])\s+")
    for m in pattern.finditer(text):
        raw = text[last_pos : m.start()]
        l_strip = len(raw) - len(raw.lstrip())
        r_strip = len(raw) - len(raw.rstrip())
        s_start = last_pos + l_strip
        s_end = m.start() - r_strip
        if s_start < s_end:
            spans.append(SentenceSpan(text=text[s_start:s_end], start=s_start, end=s_end))
        last_pos = m.end()
    if last_pos < len(text):
        raw = text[last_pos:]
        l_strip = len(raw) - len(raw.lstrip())
        r_strip = len(raw) - len(raw.rstrip())
        s_start = last_pos + l_strip
        s_end = len(text) - r_strip
        if s_start < s_end:
            spans.append(SentenceSpan(text=text[s_start:s_end], start=s_start, end=s_end))
    return spans


def _split_sentences(text: str) -> list[str]:
    return [s.text for s in _split_sentences_with_spans(text)]


def chunk_fixed(
    text: str,
    token_size: int = 500,
    overlap: int = 100,
    page_number: int | None = None,
    section_heading: str | None = None,
) -> list[ChunkOutput]:
    if not text.strip():
        return []

    tokenizer = get_tokenizer()
    encoding = tokenizer.encode(text, add_special_tokens=False)
    token_ids = encoding.ids
    offsets = encoding.offsets

    if not token_ids:
        return []

    chunks: list[ChunkOutput] = []
    start = 0
    idx = 0

    while start < len(token_ids):
        end = min(start + token_size, len(token_ids))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk_text = text[char_start:char_end]

        if chunk_text.strip():
            chunks.append(
                ChunkOutput(
                    text=chunk_text,
                    chunk_index=idx,
                    token_count=end - start,
                    page_number=page_number,
                    section_heading=section_heading,
                    char_offset_start=char_start,
                    char_offset_end=char_end,
                )
            )
            idx += 1

        step = token_size - overlap
        if step <= 0:
            step = max(1, token_size // 2)
        start += step

    return chunks


def _split_by_headers(text: str) -> list[tuple[str, str | None, int, int]]:
    pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    blocks: list[tuple[str, str | None, int, int]] = []
    last_end = 0
    current_heading: str | None = None

    for match in pattern.finditer(text):
        if match.start() > last_end:
            block_slice = text[last_end : match.start()]
            if block_slice.strip():
                l_strip = len(block_slice) - len(block_slice.lstrip())
                r_strip = len(block_slice) - len(block_slice.rstrip())
                b_start = last_end + l_strip
                b_end = match.start() - r_strip
                blocks.append((text[b_start:b_end], current_heading, b_start, b_end))
        current_heading = match.group(2).strip()
        last_end = match.end()

    if last_end < len(text):
        block_slice = text[last_end:]
        if block_slice.strip():
            l_strip = len(block_slice) - len(block_slice.lstrip())
            r_strip = len(block_slice) - len(block_slice.rstrip())
            b_start = last_end + l_strip
            b_end = len(text) - r_strip
            blocks.append((text[b_start:b_end], current_heading, b_start, b_end))

    if not blocks and text.strip():
        l_strip = len(text) - len(text.lstrip())
        r_strip = len(text) - len(text.rstrip())
        blocks.append((text[l_strip : len(text) - r_strip], None, l_strip, len(text) - r_strip))

    return blocks


def chunk_structure_aware(
    text: str,
    max_tokens: int = 500,
    sections: list[dict[str, Any]] | None = None,
) -> list[ChunkOutput]:
    """Split respecting markdown headers, paragraphs, and sentences.

    If sections (from parser) provided, use them as initial splits.
    Then subdivide any that exceed max_tokens while preserving exact character spans.
    """
    if sections:
        raw_blocks: list[tuple[str, str | None, int, int]] = []
        char_pos = 0
        for s in sections:
            stext = s.get("text", "")
            if not stext.strip():
                continue
            heading = s.get("heading")
            offset = text.find(stext, char_pos)
            if offset == -1:
                offset = text.find(stext)
            if offset == -1:
                offset = char_pos
            end = offset + len(stext)
            raw_blocks.append((stext, heading, offset, end))
            char_pos = end
    else:
        raw_blocks = _split_by_headers(text)

    chunks: list[ChunkOutput] = []
    idx = 0

    for block_text, heading, b_start, b_end in raw_blocks:
        if not block_text.strip():
            continue

        token_count = count_tokens(block_text)
        if token_count <= max_tokens:
            chunks.append(
                ChunkOutput(
                    text=block_text,
                    chunk_index=idx,
                    token_count=token_count,
                    section_heading=heading,
                    char_offset_start=b_start,
                    char_offset_end=b_end,
                )
            )
            idx += 1
        else:
            sent_spans = _split_sentences_with_spans(block_text)
            if not sent_spans:
                sub_fixed = chunk_fixed(block_text, token_size=max_tokens, overlap=50)
                for sc in sub_fixed:
                    sc.chunk_index = idx
                    sc.section_heading = heading
                    sc.char_offset_start = b_start + sc.char_offset_start
                    sc.char_offset_end = b_start + sc.char_offset_end
                    chunks.append(sc)
                    idx += 1
                continue

            current_start_idx = 0
            current_tokens = 0

            for s_idx, span in enumerate(sent_spans):
                sent_tokens = count_tokens(span.text)
                if sent_tokens > max_tokens:
                    if s_idx > current_start_idx:
                        chunk_start = b_start + sent_spans[current_start_idx].start
                        chunk_end = b_start + sent_spans[s_idx - 1].end
                        chunk_text = text[chunk_start:chunk_end]
                        chunks.append(
                            ChunkOutput(
                                text=chunk_text,
                                chunk_index=idx,
                                token_count=count_tokens(chunk_text),
                                section_heading=heading,
                                char_offset_start=chunk_start,
                                char_offset_end=chunk_end,
                            )
                        )
                        idx += 1
                    sub_fixed = chunk_fixed(span.text, token_size=max_tokens, overlap=50)
                    for sc in sub_fixed:
                        sc.chunk_index = idx
                        sc.section_heading = heading
                        sc.char_offset_start = b_start + span.start + sc.char_offset_start
                        sc.char_offset_end = b_start + span.start + sc.char_offset_end
                        chunks.append(sc)
                        idx += 1
                    current_start_idx = s_idx + 1
                    current_tokens = 0
                elif current_tokens + sent_tokens > max_tokens and s_idx > current_start_idx:
                    chunk_start = b_start + sent_spans[current_start_idx].start
                    chunk_end = b_start + sent_spans[s_idx - 1].end
                    chunk_text = text[chunk_start:chunk_end]
                    chunks.append(
                        ChunkOutput(
                            text=chunk_text,
                            chunk_index=idx,
                            token_count=count_tokens(chunk_text),
                            section_heading=heading,
                            char_offset_start=chunk_start,
                            char_offset_end=chunk_end,
                        )
                    )
                    idx += 1
                    current_start_idx = s_idx
                    current_tokens = sent_tokens
                else:
                    current_tokens += sent_tokens

            if current_start_idx < len(sent_spans):
                chunk_start = b_start + sent_spans[current_start_idx].start
                chunk_end = b_start + sent_spans[-1].end
                chunk_text = text[chunk_start:chunk_end]
                if chunk_text.strip():
                    chunks.append(
                        ChunkOutput(
                            text=chunk_text,
                            chunk_index=idx,
                            token_count=count_tokens(chunk_text),
                            section_heading=heading,
                            char_offset_start=chunk_start,
                            char_offset_end=chunk_end,
                        )
                    )
                    idx += 1

    return chunks


def chunk_semantic(
    text: str,
    max_tokens: int = 500,
    similarity_threshold_percentile: float = 25.0,
    window_size: int = 3,
    embedder: Any | None = None,
) -> list[ChunkOutput]:
    """Split where cosine similarity between adjacent sentence windows drops below threshold.

    Reuses embedder session via get_embedder() to avoid memory exhaustion and re-instantiation.
    Uses exact character spans to preserve 100% slice fidelity.
    """
    sent_spans = _split_sentences_with_spans(text)
    if len(sent_spans) <= 1:
        if not sent_spans:
            return []
        s = sent_spans[0]
        chunk_text = text[s.start : s.end]
        token_count = count_tokens(chunk_text)
        if token_count == 0:
            return []
        return [
            ChunkOutput(
                text=chunk_text,
                chunk_index=0,
                token_count=token_count,
                char_offset_start=s.start,
                char_offset_end=s.end,
            )
        ]

    if embedder is None:
        from hybrid_rag.ingest.embedder import get_embedder

        embedder = get_embedder()

    sentences = [s.text for s in sent_spans]
    embeddings = list(embedder.embed(sentences))
    emb_array = np.array(embeddings)

    windows: list[np.ndarray] = []
    for i in range(len(sentences)):
        start = max(0, i - window_size // 2)
        end = min(len(sentences), i + window_size // 2 + 1)
        windows.append(emb_array[start:end].mean(axis=0))

    similarities: list[float] = []
    for i in range(len(windows) - 1):
        a, b = windows[i], windows[i + 1]
        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
        similarities.append(cos_sim)

    if not similarities:
        s_first = sent_spans[0]
        s_last = sent_spans[-1]
        chunk_text = text[s_first.start : s_last.end]
        return [
            ChunkOutput(
                text=chunk_text,
                chunk_index=0,
                token_count=count_tokens(chunk_text),
                char_offset_start=s_first.start,
                char_offset_end=s_last.end,
            )
        ]

    threshold = float(np.percentile(similarities, similarity_threshold_percentile))

    split_indices = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            split_indices.append(i + 1)
    split_indices.append(len(sentences))

    chunks: list[ChunkOutput] = []

    for idx in range(len(split_indices) - 1):
        start_sent = split_indices[idx]
        end_sent = split_indices[idx + 1]
        chunk_start = sent_spans[start_sent].start
        chunk_end = sent_spans[end_sent - 1].end
        chunk_text = text[chunk_start:chunk_end]

        if not chunk_text.strip():
            continue

        token_count = count_tokens(chunk_text)

        if token_count > max_tokens:
            sub_chunks = chunk_fixed(chunk_text, token_size=max_tokens, overlap=50)
            for sc in sub_chunks:
                sc.chunk_index = len(chunks)
                sc.char_offset_start = chunk_start + sc.char_offset_start
                sc.char_offset_end = chunk_start + sc.char_offset_end
                chunks.append(sc)
        else:
            chunks.append(
                ChunkOutput(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    token_count=token_count,
                    char_offset_start=chunk_start,
                    char_offset_end=chunk_end,
                )
            )

    return chunks


CHUNKERS = {
    "fixed": chunk_fixed,
    "structure": chunk_structure_aware,
    "semantic": chunk_semantic,
}
