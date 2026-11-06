"""Tests for BM25 store: thread-local PyStemmer isolation, indexing, and search concurrency."""

import threading
from concurrent.futures import ThreadPoolExecutor

from hybrid_rag.retrieval.bm25_store import Bm25Store, get_bm25_store


def test_bm25_store_singleton():
    """Verify get_bm25_store returns a singleton instance."""
    store1 = get_bm25_store()
    store2 = get_bm25_store()
    assert store1 is store2


def test_thread_local_stemmer_isolation():
    """Verify different threads receive distinct Stemmer instances to prevent C-buffer races."""
    store = Bm25Store()
    stemmer_refs = []
    barrier = threading.Barrier(5)
    lock = threading.Lock()

    def worker():
        barrier.wait()
        s = store._stemmer
        with lock:
            stemmer_refs.append(s)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(stemmer_refs) == 5
    # Every thread must have its own distinct stemmer instance
    assert len({id(s) for s in stemmer_refs}) == 5


def test_bm25_indexing_and_search():
    """Verify basic indexing and searching works accurately."""
    store = Bm25Store()
    texts = [
        "Python is a popular programming language.",
        "FastAPI is a modern web framework for Python.",
        "PostgreSQL and Qdrant provide vector search.",
    ]
    chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
    count = store.rebuild(texts, chunk_ids, "fixed")
    assert count == 3
    assert store.has_index("fixed")

    # Search for Python
    results = store.search("Python", "fixed", top_k=5)
    assert len(results) >= 2
    matched_ids = [cid for cid, score in results]
    assert "chunk-1" in matched_ids
    assert "chunk-2" in matched_ids


def test_concurrent_search_and_tokenize():
    """Verify multiple concurrent threads can search BM25 without stemmer collisions."""
    store = Bm25Store()
    texts = [
        f"Document {i} discusses machine learning and data science topic {i}." for i in range(50)
    ]
    chunk_ids = [f"c-{i}" for i in range(50)]
    store.rebuild(texts, chunk_ids, "fixed")

    queries = ["machine learning", "data science", "topic 10", "Document 25", "algorithm"]

    def perform_searches(thread_idx: int):
        results = []
        for q in queries:
            res = store.search(q, "fixed", top_k=10)
            results.append(len(res))
        return results

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(perform_searches, i) for i in range(20)]
        all_results = [f.result() for f in futures]

    assert len(all_results) == 20
    for r in all_results:
        assert len(r) == len(queries)


def test_bm25_clear_and_empty():
    """Verify clearing indexes works cleanly."""
    store = Bm25Store()
    store.rebuild(["test doc"], ["id-1"], "fixed")
    assert store.has_index("fixed")

    store.clear("fixed")
    assert not store.has_index("fixed")
    assert store.search("test", "fixed") == []


def test_stemmer_setter_override():
    """Verify thread-local stemmer setter allows custom override."""
    store = Bm25Store()
    custom_mock = object()
    store._stemmer = custom_mock
    assert store._stemmer is custom_mock
