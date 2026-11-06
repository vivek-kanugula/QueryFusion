"""Dense embedding via fastembed (ONNX, no torch)."""

from __future__ import annotations

import threading

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
                _embedder = TextEmbedding(
                    model_name=settings.embedding_model,
                    cache_dir=settings.fastembed_cache_path,
                )
    return _embedder


def reset_embedder(embedder: TextEmbedding | None = None) -> None:
    global _embedder
    with _embedder_lock:
        _embedder = embedder


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    embedder = get_embedder()
    return list(embedder.embed(texts))


QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def embed_query(query: str) -> np.ndarray:
    embedder = get_embedder()
    return list(embedder.embed([QUERY_INSTRUCTION + query]))[0]
