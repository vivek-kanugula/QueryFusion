"""Dense embedding via fastembed (ONNX, no torch)."""

from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from hybrid_rag.settings import settings

_embedder: TextEmbedding | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                cpu_limit = float(os.getenv("CONTAINER_CPU_LIMIT", "1.0"))
                threads = 2 if cpu_limit >= 2.0 else 1
                _embedder = TextEmbedding(
                    model_name=settings.embedding_model,
                    cache_dir=settings.fastembed_cache_path,
                    threads=threads,
                )
    return _embedder


def reset_embedder(embedder: TextEmbedding | None = None) -> None:
    global _embedder
    with _embedder_lock:
        _embedder = embedder
        _embed_query_cached.cache_clear()


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    embedder = get_embedder()
    return list(embedder.embed(texts))


QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=512)
def _embed_query_cached(clean_query: str) -> tuple[float, ...]:
    embedder = get_embedder()
    vec = list(embedder.embed([QUERY_INSTRUCTION + clean_query]))[0]
    return tuple(float(x) for x in vec)


def embed_query(query: str) -> np.ndarray:
    clean = query.strip()
    return np.array(_embed_query_cached(clean), dtype=np.float32)

