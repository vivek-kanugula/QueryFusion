"""BM25 index store using bm25s with PyStemmer.

One index per chunking strategy. Atomic swap pattern ensures concurrent
reads never see a half-built index.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import bm25s
import Stemmer as PyStemmer

from hybrid_rag.retrieval.tokenizer import expand_technical_tokens

logger = logging.getLogger(__name__)

_bm25_store: Bm25Store | None = None


class Bm25Store:
    def __init__(self) -> None:
        self._indices: dict[str, bm25s.BM25] = {}
        self._chunk_ids: dict[str, list[str]] = {}
        self._chunk_doc_ids: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def _stemmer(self) -> PyStemmer.Stemmer:
        if not hasattr(self._local, "stemmer") or self._local.stemmer is None:
            self._local.stemmer = PyStemmer.Stemmer("english")
        return self._local.stemmer

    @_stemmer.setter
    def _stemmer(self, val: Any) -> None:
        self._local.stemmer = val

    def _tokenize(self, texts: list[str]):
        expanded = [expand_technical_tokens(t) for t in texts]
        return bm25s.tokenize(expanded, stemmer=self._stemmer, stopwords="en")

    def rebuild(
        self,
        texts: list[str],
        chunk_ids: list[str],
        strategy: str,
        doc_ids: list[str] | None = None,
    ) -> int:
        """Build a new BM25 index and atomically swap it in."""
        if not texts:
            with self._lock:
                self._indices.pop(strategy, None)
                self._chunk_ids.pop(strategy, None)
                self._chunk_doc_ids.pop(strategy, None)
            return 0

        corpus_tokens = self._tokenize(texts)
        if not getattr(corpus_tokens, "vocab", None):
            with self._lock:
                self._indices.pop(strategy, None)
                self._chunk_ids.pop(strategy, None)
                self._chunk_doc_ids.pop(strategy, None)
            logger.warning(
                "BM25 index skipped (empty vocabulary after tokenization): strategy=%s, docs=%d",
                strategy,
                len(texts),
            )
            return 0

        new_index = bm25s.BM25()
        try:
            new_index.index(corpus_tokens)
        except ValueError as exc:
            logger.warning(
                "BM25 index failed due to empty/invalid vocabulary: strategy=%s, error=%s",
                strategy,
                exc,
            )
            with self._lock:
                self._indices.pop(strategy, None)
                self._chunk_ids.pop(strategy, None)
                self._chunk_doc_ids.pop(strategy, None)
            return 0

        doc_id_list = list(doc_ids) if doc_ids is not None else [""] * len(chunk_ids)

        with self._lock:
            self._indices[strategy] = new_index
            self._chunk_ids[strategy] = list(chunk_ids)
            self._chunk_doc_ids[strategy] = doc_id_list

        logger.info("BM25 index rebuilt: strategy=%s, docs=%d", strategy, len(texts))
        return len(texts)

    def rebuild_from_db(self, db, strategy: str) -> int:
        """Restore BM25 index for a strategy from PostgreSQL chunks."""
        from hybrid_rag.models.document import Document, DocumentChunk

        chunks = (
            db.query(DocumentChunk)
            .join(Document)
            .filter(Document.chunking_strategy == strategy)
            .order_by(DocumentChunk.id)
            .all()
        )

        texts = [c.content for c in chunks]
        ids = [str(c.id) for c in chunks]
        doc_ids = [str(c.document_id) for c in chunks]
        return self.rebuild(texts, ids, strategy, doc_ids=doc_ids)

    def search(
        self,
        query: str,
        strategy: str,
        top_k: int = 50,
        document_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> list[tuple[str, float]]:
        """Search BM25 index. Returns [(chunk_id, score)]."""
        with self._lock:
            index = self._indices.get(strategy)
            chunk_ids = self._chunk_ids.get(strategy)
            chunk_doc_ids = self._chunk_doc_ids.get(strategy)

        if index is None or not chunk_ids:
            return []

        doc_filter_set = {str(d) for d in document_ids} if document_ids else None

        query_tokens = self._tokenize([query])
        k = (
            min(len(chunk_ids), max(top_k * 10, 500))
            if doc_filter_set
            else min(top_k, len(chunk_ids))
        )
        if k == 0:
            return []

        results, scores = index.retrieve(query_tokens, k=k)

        output: list[tuple[str, float]] = []
        for idx, score in zip(results[0], scores[0]):
            idx = int(idx)
            if 0 <= idx < len(chunk_ids) and score > 0:
                if doc_filter_set is not None:
                    did = chunk_doc_ids[idx] if chunk_doc_ids and idx < len(chunk_doc_ids) else None
                    if not did or did not in doc_filter_set:
                        continue
                output.append((chunk_ids[idx], float(score)))
                if len(output) >= top_k:
                    break

        return output

    def has_index(self, strategy: str) -> bool:
        with self._lock:
            return strategy in self._indices

    def clear(self, strategy: str | None = None) -> None:
        with self._lock:
            if strategy:
                self._indices.pop(strategy, None)
                self._chunk_ids.pop(strategy, None)
                self._chunk_doc_ids.pop(strategy, None)
            else:
                self._indices.clear()
                self._chunk_ids.clear()
                self._chunk_doc_ids.clear()


_bm25_store_lock = threading.Lock()


def get_bm25_store() -> Bm25Store:
    global _bm25_store
    if _bm25_store is None:
        with _bm25_store_lock:
            if _bm25_store is None:
                _bm25_store = Bm25Store()
    return _bm25_store
