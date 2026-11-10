"""Adversarial stress-test harness for Milestone 1: Retrieval & Core Engine.

Specifically targets:
- CR-03: BM25 Index Crash on Empty / Stopword-Only Vocabulary
- LO-02: Non-ASCII Characters Stripped by BM25 Technical Tokenizer
"""

from __future__ import annotations

import threading
import time

import pytest

from hybrid_rag.core.jobs import JobRegistry
from hybrid_rag.retrieval.bm25_store import Bm25Store
from hybrid_rag.retrieval.tokenizer import (
    expand_technical_tokens,
    tokenize_technical,
)

# =====================================================================
# CR-03 Adversarial Suite: Empty / Stopword / Edge Inputs & Concurrency
# =====================================================================

class TestCR03Adversarial:
    """Stress testing BM25 index against empty, stopword-only, malformed inputs and concurrency."""

    @pytest.mark.parametrize(
        "empty_corpus,cids",
        [
            ([], []),
            ([""], ["c1"]),
            (["", "", ""], ["c1", "c2", "c3"]),
            (["   "], ["c1"]),
            (["\t\r\n\v\f"], ["c1"]),
            (["   ", "\t\n", "   \r "], ["c1", "c2", "c3"]),
            ([" ", "  ", "   ", "    "], ["c1", "c2", "c3", "c4"]),
        ],
    )
    def test_empty_and_whitespace_inputs(self, empty_corpus, cids):
        """Confirm no ValueError is thrown and index is safely empty."""
        store = Bm25Store()
        indexed = store.rebuild(empty_corpus, cids, "fixed")
        assert indexed == 0
        assert not store.has_index("fixed")
        assert store.search("anything", "fixed") == []
        assert store.search("", "fixed") == []

    @pytest.mark.parametrize(
        "stopword_corpus,cids",
        [
            (["the"], ["c1"]),
            (["is"], ["c1"]),
            (["a"], ["c1"]),
            (["the and is"], ["c1"]),
            (["at by for", "of on with", "it that this"], ["c1", "c2", "c3"]),
            (["", "   ", "the", "and is", "\n\t", "for"], ["c1", "c2", "c3", "c4", "c5", "c6"]),
            (["the, and, is!", "at... by? for;"], ["c1", "c2"]),
            (["a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with"], ["c1"]),
        ],
    )
    def test_stopword_variations_no_value_error(self, stopword_corpus, cids):
        """Confirm all stopword variants return 0 and never raise ValueError."""
        store = Bm25Store()
        indexed = store.rebuild(stopword_corpus, cids, "hierarchical")
        assert indexed == 0
        assert not store.has_index("hierarchical")
        assert store.search("the", "hierarchical") == []
        assert store.search("random_query", "hierarchical") == []

    def test_pure_symbols_and_emojis(self):
        """Confirm non-letter symbol and emoji sets return 0 without crashing."""
        store = Bm25Store()
        symbols = ["!@#$%^&*()_+", "<>?/|\\~`", "--- === ..."]
        indexed = store.rebuild(symbols, ["s1", "s2", "s3"], "fixed")
        assert indexed == 0
        assert not store.has_index("fixed")
        assert store.search("---", "fixed") == []

        emojis = ["😀 🎉 🚀", "🔥 💥 ⚡", "🌟 💡 🎯"]
        indexed_emoji = store.rebuild(emojis, ["e1", "e2", "e3"], "fixed")
        assert indexed_emoji == 0
        assert not store.has_index("fixed")
        assert store.search("🚀", "fixed") == []

    def test_repeated_rebuild_state_flapping(self):
        """Test transitioning back and forth between valid and empty/stopword texts."""
        store = Bm25Store()
        strategy = "flapping"

        valid_texts = ["PostgreSQL pgvector extension", "BM25 keyword search index"]
        valid_ids = ["v1", "v2"]
        empty_texts = ["the and is", "at by for"]
        empty_ids = ["e1", "e2"]

        for cycle in range(5):
            # 1. Rebuild with valid docs
            count = store.rebuild(valid_texts, valid_ids, strategy)
            assert count == 2
            assert store.has_index(strategy)
            hits = store.search("pgvector", strategy)
            assert len(hits) == 1
            assert hits[0][0] == "v1"

            # 2. Rebuild with stopword docs -> must wipe index
            count = store.rebuild(empty_texts, empty_ids, strategy)
            assert count == 0
            assert not store.has_index(strategy)
            assert store.search("pgvector", strategy) == []

            # 3. Rebuild with whitespace docs -> must remain empty
            count = store.rebuild(["   ", "\n\t"], ["w1", "w2"], strategy)
            assert count == 0
            assert not store.has_index(strategy)
            assert store.search("pgvector", strategy) == []

            # 4. Rebuild with empty list -> must remain empty
            count = store.rebuild([], [], strategy)
            assert count == 0
            assert not store.has_index(strategy)
            assert store.search("pgvector", strategy) == []

    def test_search_edge_cases(self):
        """Test search under diverse adversarial queries and bounds."""
        store = Bm25Store()
        strategy = "edge_search"

        # Search against non-existent strategy
        assert store.search("test", "non_existent") == []

        # Populate with valid docs
        store.rebuild(
            ["Vector search engine with embeddings", "Keyword BM25 full-text indexing"],
            ["c1", "c2"],
            strategy,
            doc_ids=["doc_1", "doc_2"],
        )
        assert store.has_index(strategy)

        # 1. Empty query
        assert store.search("", strategy) == []

        # 2. Whitespace query
        assert store.search("   \t\n  ", strategy) == []

        # 3. Stopwords only query
        assert store.search("the and is", strategy) == []

        # 4. Punctuation only query
        assert store.search("!@#$%^&*()", strategy) == []

        # 5. Emoji only query
        assert store.search("🚀 🔥", strategy) == []

        # 6. Non-existent term
        assert store.search("xyznonexistentterm12345", strategy) == []

        # 7. Boundary top_k = 0
        assert store.search("vector", strategy, top_k=0) == []

        # 8. Large top_k exceeding corpus size
        res = store.search("vector", strategy, top_k=1000)
        assert len(res) == 1
        assert res[0][0] == "c1"

        # 9. Document filter matching
        res = store.search("vector", strategy, document_ids={"doc_1"})
        assert len(res) == 1
        assert res[0][0] == "c1"

        # 10. Document filter not matching
        res = store.search("vector", strategy, document_ids={"doc_other"})
        assert res == []

        # 11. Clear index and search
        store.clear(strategy)
        assert not store.has_index(strategy)
        assert store.search("vector", strategy) == []

    def test_concurrent_index_rebuilds(self):
        """Stress test concurrent rebuilds with mixed valid and empty/stopword texts."""
        store = Bm25Store()
        strategy = "concurrent_strat"
        errors: list[Exception] = []

        valid_pool = [
            ["Machine learning model deployment", "Continuous integration pipeline"],
            ["Distributed database sharding", "RAFT consensus algorithm"],
            ["FastAPI web service routing", "Pytest automated verification"],
        ]
        empty_pool = [
            ["the and is", "at by for"],
            ["", "   "],
            ["\t\n\r", "of on with"],
            [],
        ]

        def worker_task(thread_id: int):
            try:
                for i in range(25):
                    if (thread_id + i) % 2 == 0:
                        texts = valid_pool[(thread_id + i) % len(valid_pool)]
                        cids = [f"t{thread_id}_c{j}" for j in range(len(texts))]
                        count = store.rebuild(texts, cids, strategy)
                        assert count in (0, len(texts))
                    else:
                        texts = empty_pool[(thread_id + i) % len(empty_pool)]
                        cids = [f"t{thread_id}_e{j}" for j in range(len(texts))]
                        count = store.rebuild(texts, cids, strategy)
                        assert count == 0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker_task, args=(tid,)) for tid in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent rebuild errors encountered: {errors}"

    def test_concurrent_rebuilds_and_searches(self):
        """Stress test concurrent rebuilds simultaneous with active searches."""
        store = Bm25Store()
        strategy = "concurrent_rw"
        errors: list[Exception] = []
        stop_event = threading.Event()

        # Seed valid index
        store.rebuild(["Initial database document", "Search index chunk"], ["c1", "c2"], strategy)

        def rebuilder(tid: int):
            counter = 0
            while not stop_event.is_set():
                try:
                    if counter % 3 == 0:
                        store.rebuild(["the and is", "at by for"], ["s1", "s2"], strategy)
                    elif counter % 3 == 1:
                        store.rebuild(["", "   "], ["e1", "e2"], strategy)
                    else:
                        store.rebuild([f"Document version {counter}", "BM25 retrieval candidate"], [f"c_{counter}_1", f"c_{counter}_2"], strategy)
                    counter += 1
                    time.sleep(0.005)
                except Exception as exc:
                    errors.append(exc)
                    break

        def searcher(tid: int):
            queries = ["document", "retrieval", "the", "", "nonexistent", "index"]
            q_idx = 0
            while not stop_event.is_set():
                try:
                    query = queries[q_idx % len(queries)]
                    results = store.search(query, strategy, top_k=10)
                    assert isinstance(results, list)
                    q_idx += 1
                    time.sleep(0.003)
                except Exception as exc:
                    errors.append(exc)
                    break

        rebuild_threads = [threading.Thread(target=rebuilder, args=(i,)) for i in range(4)]
        search_threads = [threading.Thread(target=searcher, args=(i,)) for i in range(6)]

        all_threads = rebuild_threads + search_threads
        for t in all_threads:
            t.start()

        # Run stress test for 1.5 seconds
        time.sleep(1.5)
        stop_event.set()

        for t in all_threads:
            t.join(timeout=3.0)

        assert errors == [], f"Concurrent search/rebuild errors: {errors}"


# =====================================================================
# LO-02 Adversarial Suite: Non-ASCII, Unicode, Scripts & Identifiers
# =====================================================================

class TestLO02Adversarial:
    """Stress testing technical tokenizer with international characters, symbols, and composite tokens."""

    @pytest.mark.parametrize(
        "word,expected_part",
        [
            ("naïve", "ï"),
            ("résumé", "é"),
            ("café", "é"),
            ("façade", "ç"),
            ("año", "ñ"),
            ("über", "ü"),
            ("château", "â"),
            ("coöperation", "ö"),
            ("mañana", "ñ"),
            ("señor", "ñ"),
        ],
    )
    def test_accented_latin_characters_preserved(self, word, expected_part):
        """Confirm accented letters are preserved and not stripped to ASCII fragments."""
        tokens = tokenize_technical(word)
        assert len(tokens) >= 1
        combined = " ".join(tokens)
        assert expected_part in combined, f"Expected '{expected_part}' in tokens '{combined}'"

    def test_german_eszett_and_umlauts(self):
        """Verify German letters (ß, ä, ö, ü) are preserved intact in composite tokens and words."""
        text = "München-2024.config in Groß_Straße/Abschnitt_4"
        expanded = expand_technical_tokens(text)

        # Composite and sub-tokens should retain German letters
        assert "München2024config" in expanded
        assert "GroßStraßeAbschnitt4" in expanded
        assert "München" in expanded
        assert "Groß" in expanded
        assert "Straße" in expanded

        tokens = tokenize_technical(text)
        assert "münchen2024config" in tokens
        assert "großstraßeabschnitt4" in tokens
        assert "straße" in tokens

    def test_cyrillic_preservation_and_search(self):
        """Verify Cyrillic script words and technical identifiers are indexed and retrievable."""
        store = Bm25Store()
        strategy = "cyrillic"

        doc1 = "Москва — столица России. Настройка сервера-1.2.3 в файле база_данных.sql"
        doc2 = "Привет мир! Тестирование распределенной системы."
        store.rebuild([doc1, doc2], ["c1", "c2"], strategy)

        # Search for pure Cyrillic words
        hits_moscow = store.search("Москва", strategy)
        assert len(hits_moscow) == 1
        assert hits_moscow[0][0] == "c1"

        hits_privet = store.search("Привет", strategy)
        assert len(hits_privet) == 1
        assert hits_privet[0][0] == "c2"

        # Search for Cyrillic technical identifiers (expanded sub-token & composite)
        hits_server = store.search("сервера", strategy)
        assert len(hits_server) == 1
        assert hits_server[0][0] == "c1"

        hits_db = store.search("базаданныхsql", strategy)
        assert len(hits_db) == 1
        assert hits_db[0][0] == "c1"

    def test_cjk_preservation_and_search(self):
        """Verify CJK ideographs in technical tokens and spaced text are preserved."""
        # Technical token with delimiters
        text = "東京-v2.0 上海_db:3306 서울/서버-1"
        expanded = expand_technical_tokens(text)

        assert "東京v20" in expanded
        assert "東京" in expanded
        assert "上海db3306" in expanded
        assert "上海" in expanded
        assert "서울서버1" in expanded
        assert "서울" in expanded

        store = Bm25Store()
        store.rebuild(
            ["東京 サーバー クラウド 設定", "上海 数据库 分布式 架构"],
            ["c_jp", "c_cn"],
            "cjk",
        )
        jp_hits = store.search("東京", "cjk")
        assert len(jp_hits) >= 1
        assert jp_hits[0][0] == "c_jp"

        cn_hits = store.search("上海", "cjk")
        assert len(cn_hits) >= 1
        assert cn_hits[0][0] == "c_cn"

    def test_mixed_scripts_in_single_technical_token(self):
        """Verify composite tokens combining Latin, Cyrillic, Greek, and CJK with numbers."""
        token = "café-привет-東京-v1.0.tar.gz"
        expanded = expand_technical_tokens(token)
        # Separators (- and .) stripped in composite
        assert "caféпривет東京v10targz" in expanded
        assert "café" in expanded
        assert "привет" in expanded
        assert "東京" in expanded
        assert "v1" in expanded
        assert "tar" in expanded
        assert "gz" in expanded

    def test_technical_delimiter_coverage(self):
        """Verify all supported delimiters: -, _, ., :, /, \\"""
        delims = [
            ("dash-token-123", "dashtoken123", ["dash", "token", "123"]),
            ("under_score_token_456", "underscoretoken456", ["under", "score", "token", "456"]),
            ("dot.delimited.config", "dotdelimitedconfig", ["dot", "delimited", "config"]),
            ("colon:port:8080", "colonport8080", ["colon", "port", "8080"]),
            ("slash/path/to/resource", "slashpathtoresource", ["slash", "path", "to", "resource"]),
            (r"backslash\path\to\file", "backslashpathtofile", ["backslash", "path", "to", "file"]),
        ]
        for raw, expected_comp, expected_sub in delims:
            expanded = expand_technical_tokens(raw)
            assert expected_comp in expanded
            for sub in expected_sub:
                assert sub in expanded

    def test_emojis_adjacent_to_technical_tokens(self):
        """Verify emojis adjacent to tokens do not crash or corrupt extraction."""
        text = "🔥[ERROR-500] Critical failure in DB_CLUSTER:PRIMARY 🚀"
        expanded = expand_technical_tokens(text)
        assert "ERROR500" in expanded
        assert "DBCLUSTERPRIMARY" in expanded

        tokens = tokenize_technical(text)
        assert "error500" in tokens
        # Stemmer transforms primary -> primari
        assert "dbclusterprimari" in tokens
        assert "db" in tokens
        assert "cluster" in tokens
        assert "critic" in tokens  # stemmed from critical
        assert "failur" in tokens  # stemmed from failure

    def test_extreme_and_unusual_unicode_inputs(self):
        """Verify handling of RTL (Hebrew/Arabic), zero-width characters, null bytes, long tokens."""
        # RTL text
        rtl_text = "שלום-1.0 and مرحبا_v2"
        expanded_rtl = expand_technical_tokens(rtl_text)
        assert "שלום10" in expanded_rtl
        assert "مرحباv2" in expanded_rtl

        # Long token
        long_tok = "part_" + ("x" * 500) + "-v1"
        expanded_long = expand_technical_tokens(long_tok)
        assert "part" in expanded_long

        # String with null bytes
        null_text = "clean\x00technical-id_v1\x00after"
        expanded_null = expand_technical_tokens(null_text)
        assert "technicalidv1" in expanded_null

        # Ensure tokenize_technical never throws on unusual characters
        toks = tokenize_technical("👨‍👩‍👧‍👦 \u200b\u200c\u200d\ufeff zero width")
        assert "zero" in toks
        assert "width" in toks


# =====================================================================
# Additional Adversarial Suites: Multithreaded Tokenization & Jobs Churn
# =====================================================================

def test_concurrent_multithreaded_tokenization():
    """Verify thread-safety of PyStemmer and tokenizer across 16 threads."""
    test_texts = [
        "café München naïve résumé façade año Über-сервер:8080/api",
        "Привет мир! Москва — столица России. Настройка сервера-1.2.3",
        "東京-v2.0 上海_db:3306 서울/서버-1 [ERROR-500] Critical failure",
        "DB-204 error in config.prod.yaml at redis://127.0.0.1:6379/0",
    ]
    errors: list[Exception] = []

    def worker(tid: int):
        try:
            for _ in range(50):
                text = test_texts[tid % len(test_texts)]
                expanded = expand_technical_tokens(text)
                assert isinstance(expanded, str)
                toks = tokenize_technical(text)
                assert isinstance(toks, list)
                assert len(toks) > 0
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Multithreaded tokenization errors: {errors}"


def test_concurrent_job_registry_churn():
    """Stress test JobRegistry bounded capacity under heavy concurrent read/write churn."""
    max_cap = 20
    registry = JobRegistry(max_jobs=max_cap)
    errors: list[Exception] = []
    stop_event = threading.Event()

    def job_creator(tid: int):
        counter = 0
        while not stop_event.is_set():
            try:
                job = registry.create_job(f"churn_task_{tid}")
                if counter % 3 == 0:
                    registry.complete_job(job.job_id, {"status": "ok"})
                elif counter % 3 == 1:
                    registry.fail_job(job.job_id, "simulated_error")
                else:
                    registry.update_progress(job.job_id, 50.0, "running")
                counter += 1
                assert registry.count() <= max_cap
                time.sleep(0.002)
            except Exception as exc:
                errors.append(exc)
                break

    threads = [threading.Thread(target=job_creator, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()

    time.sleep(1.0)
    stop_event.set()

    for t in threads:
        t.join(timeout=3.0)

    assert errors == [], f"JobRegistry churn errors: {errors}"
    assert registry.count() <= max_cap
