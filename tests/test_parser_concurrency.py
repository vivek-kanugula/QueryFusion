"""Chunk Offset, Parser Hardening, and Concurrency Stress Test Suite.

Verifies character offset fidelity across whitespace variations,
HTML/Markdown parser edge cases, and multi-threaded job queue safety.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from hybrid_rag.core.jobs import JobRegistry, JobStatus
from hybrid_rag.ingest.chunkers import (
    chunk_fixed,
    chunk_semantic,
    chunk_structure_aware,
)
from hybrid_rag.ingest.parsers import (
    parse_html,
    parse_markdown,
)
from hybrid_rag.retrieval.bm25_store import Bm25Store

# ==============================================================================
# SECTION 1: Chunk Character Offset Fidelity on Whitespace
# ==============================================================================

PATHOLOGICAL_DOCUMENTS = [
    # 1. Mixed \r\n and \n line breaks with trailing and leading newlines
    (
        "crlf_mixed",
        "\r\n\r\nParagraph 1 line 1.\r\nParagraph 1 line 2.\n\n"
        "Paragraph 2 line 1.\r\n\r\n\r\nParagraph 3 line 1.\nParagraph 3 line 2.\r\n",
    ),
    # 2. Extreme consecutive spaces (5 to 25 spaces) between words and sentences
    (
        "consecutive_spaces",
        "Sentence one begins here.     Sentence two has five spaces.          "
        "Sentence three has ten spaces.                    Sentence four has twenty spaces.",
    ),
    # 3. Tabs mixed with spaces and irregular indentation
    (
        "tabs_and_spaces",
        "\t\tFirst tabbed sentence.\t   \tSecond sentence with mixed tabs and spaces.\n"
        "\t\t\tThird sentence deeply indented with tabs.\t   Fourth sentence.\n",
    ),
    # 4. Multiple blank lines between paragraphs and sentences
    (
        "multiple_blank_lines",
        "Section Alpha.\n\n\n\n\n\nSection Beta after six blank lines.\n\n\n\n"
        "Section Gamma after four blank lines.\n\n\n\n\n\n\n\nSection Delta.",
    ),
    # 5. Leading and trailing whitespace around headers and code
    (
        "headers_and_code_whitespace",
        "   #   Whitespace Header One   \n\n"
        "Content paragraph under header one.\n\n"
        "    def sample_code():\n        return 42\n\n"
        "  ##   Header Two With Trailing Spaces   \n\n"
        "Content under header two.\n",
    ),
    # 6. Unicode whitespace, special characters, emojis, CJK, and accents
    (
        "unicode_whitespace_and_special",
        "Unicode\u00a0non-breaking\u00a0space.\u2003Em\u2003space\u2003sentence.\u3000"
        "Ideographic\u3000space.\u2009Thin\u2009space.\u200bZero-width space.\n\n"
        "Emoji text: 🚀 Rocket launched! ✨ Sparkles. 🔥 Performance. 💡 Idea.\n\n"
        "Multilingual: 自然语言处理。 한국어. Accents: déjà vu, naïve café, façade.\r\n",
    ),
    # 7. Massive composite document combining all pathological vectors
    (
        "massive_composite",
        "   # Pathological Document Analysis   \r\n\r\n"
        "First sentence in section one.     Multiple spaces here. \t Tabs here.\r\n\r\n\r\n\r\n"
        "## Subsection 1.1: Irregularities \n\n"
        "Here is text with \u00a0 non-breaking spaces and \u2003 em spaces.\n"
        "Next line has unicode emojis: 🎯 Target acquired. 🏆 Victory achieved.\r\n\r\n"
        "    indented_code_line_1()\n    indented_code_line_2()\n\n"
        "```python\n@decorator\ndef func():\n    return 'fenced'\n```\r\n\r\n\r\n"
        "### Subsection 1.1.1: Deep Structure\r\n"
        "Paragraph Alpha with many sentences. Sentence A. Sentence B! Sentence C? Sentence D.\n\n\n"
        "Paragraph Beta with trailing whitespace and accents: resume, résumé, protégé.\t\t\n"
        "Final sentence concluding the test document.",
    ),
]


@pytest.mark.parametrize("doc_name,text", PATHOLOGICAL_DOCUMENTS)
@pytest.mark.parametrize("token_size,overlap", [(20, 5), (50, 15), (200, 50)])
def test_fixed_chunking_exact_slice_fidelity(
    doc_name: str, text: str, token_size: int, overlap: int
):
    """Stress-test chunk_fixed: 100% of chunks must match document_text[start:end] exactly."""
    chunks = chunk_fixed(text, token_size=token_size, overlap=overlap)
    assert len(chunks) > 0, f"Failed to produce chunks for {doc_name}"

    prev_start = -1
    for i, c in enumerate(chunks):
        # 1. Exact slice equality
        slice_text = text[c.char_offset_start : c.char_offset_end]
        assert slice_text == c.text, (
            f"[{doc_name}] Chunk {i} slice mismatch: expected {c.text!r}, got {slice_text!r} "
            f"at offsets ({c.char_offset_start}, {c.char_offset_end})"
        )
        # 2. Boundary guarantees
        assert c.char_offset_start >= 0, f"Negative start offset {c.char_offset_start}"
        assert c.char_offset_end <= len(text), (
            f"End offset {c.char_offset_end} exceeds text length {len(text)}"
        )
        assert c.char_offset_start < c.char_offset_end, (
            f"Inverted offsets: {c.char_offset_start} >= {c.char_offset_end}"
        )
        # 3. Monotonically non-decreasing start offsets
        assert c.char_offset_start >= prev_start, (
            f"Start offset {c.char_offset_start} decreased from {prev_start}"
        )
        prev_start = c.char_offset_start
        # 4. Non-empty stripped text and positive tokens
        assert c.text.strip() != ""
        assert c.token_count > 0


@pytest.mark.parametrize("doc_name,text", PATHOLOGICAL_DOCUMENTS)
@pytest.mark.parametrize("max_tokens", [10, 25, 60, 500])
def test_structure_aware_chunking_exact_slice_fidelity(
    doc_name: str, text: str, max_tokens: int
):
    """Stress-test chunk_structure_aware: 100% chunks match slice exactly."""
    chunks = chunk_structure_aware(text, max_tokens=max_tokens)
    assert len(chunks) > 0, f"Failed to produce chunks for {doc_name}"

    prev_start = -1
    for i, c in enumerate(chunks):
        slice_text = text[c.char_offset_start : c.char_offset_end]
        assert slice_text == c.text, (
            f"[{doc_name}] Structure chunk {i} slice mismatch: expected {c.text!r}, "
            f"got {slice_text!r} at offsets ({c.char_offset_start}, {c.char_offset_end})"
        )
        assert c.char_offset_start >= 0
        assert c.char_offset_end <= len(text)
        assert c.char_offset_start < c.char_offset_end
        assert c.char_offset_start >= prev_start, (
            f"Structure chunk {i} start {c.char_offset_start} < previous {prev_start}"
        )
        prev_start = c.char_offset_start
        assert c.text.strip() != ""
        assert c.token_count > 0


@pytest.mark.parametrize("doc_name,text", PATHOLOGICAL_DOCUMENTS)
@pytest.mark.parametrize("max_tokens", [15, 50, 500])
def test_semantic_chunker_exact_slice_fidelity(
    doc_name: str, text: str, max_tokens: int
):
    """Stress-test chunk_semantic: 100% of chunks must match document_text[start:end] exactly."""
    chunks = chunk_semantic(text, max_tokens=max_tokens)
    assert len(chunks) > 0, f"Failed to produce semantic chunks for {doc_name}"

    prev_start = -1
    for i, c in enumerate(chunks):
        slice_text = text[c.char_offset_start : c.char_offset_end]
        assert slice_text == c.text, (
            f"[{doc_name}] Semantic chunk {i} slice mismatch: expected {c.text!r}, "
            f"got {slice_text!r} at offsets ({c.char_offset_start}, {c.char_offset_end})"
        )
        assert c.char_offset_start >= 0
        assert c.char_offset_end <= len(text)
        assert c.char_offset_start < c.char_offset_end
        assert c.char_offset_start >= prev_start, (
            f"Semantic chunk {i} start {c.char_offset_start} < previous {prev_start}"
        )
        prev_start = c.char_offset_start
        assert c.text.strip() != ""
        assert c.token_count > 0


def test_structure_aware_with_parsed_sections_fidelity():
    """Verify chunk_structure_aware with sections accurately preserves offsets."""
    text = (
        "Heading One\n\n"
        "First section text with irregular    spaces and \t tabs.\n\n"
        "Heading Two\n\n"
        "Second section text with emoji 🌟 and accents: café.\n\n"
        "Heading Three\n\n"
        "Third section content."
    )
    sections = [
        {
            "text": "First section text with irregular    spaces and \t tabs.",
            "heading": "Heading One",
        },
        {"text": "Second section text with emoji 🌟 and accents: café.", "heading": "Heading Two"},
        {"text": "Third section content.", "heading": "Heading Three"},
    ]
    chunks = chunk_structure_aware(text, max_tokens=15, sections=sections)
    assert len(chunks) >= 3
    for c in chunks:
        assert text[c.char_offset_start : c.char_offset_end] == c.text
        assert c.char_offset_start >= 0
        assert c.char_offset_end <= len(text)


# ==============================================================================
# SECTION 2: HTML Nested Tag Deduplication & Markdown Hardening
# ==============================================================================


def test_html_deeply_nested_dom_deduplication():
    """Verify parse_html extracts each text fragment once without parent/child duplication."""
    html = b"""
    <!DOCTYPE html>
    <html>
    <head><title>Deep DOM Test</title></head>
    <body>
        <table>
            <thead>
                <tr>
                    <th>Col 1 Header</th>
                    <th>Col 2 Header</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><p><span><b>Cell Text 1</b></span></p></td>
                    <td><p>Cell Text 2</p></td>
                </tr>
                <tr>
                    <td><div><section><p><span>Cell Text 3</span></p></section></div></td>
                    <td><p><span><em><b>Cell Text 4</b></em></span></p></td>
                </tr>
            </tbody>
        </table>
        <ul>
            <li>Item 1
                <ul>
                    <li>Nested 1A</li>
                    <li>Nested 1B
                        <ul>
                            <li>Deep Leaf 1B-i</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li>Item 2</li>
        </ul>
    </body>
    </html>
    """
    doc = parse_html(html, "deep_dom.html")
    texts = [s.text for s in doc.sections]

    # Verify no section text is duplicated
    assert len(texts) == len(set(texts)), f"Duplicate sections found: {texts}"

    # Verify table cell text extracted
    assert any("Cell Text 1" in t for t in texts)
    assert any("Cell Text 2" in t for t in texts)
    assert any("Cell Text 3" in t for t in texts)
    assert any("Cell Text 4" in t for t in texts)

    # Crucial check: Cell Text 1 must not appear twice (e.g. once from td and once from p)
    occurrences_cell1 = sum(t.count("Cell Text 1") for t in texts)
    assert occurrences_cell1 == 1, f"Cell Text 1 appeared {occurrences_cell1} times!"

    occurrences_cell2 = sum(t.count("Cell Text 2") for t in texts)
    assert occurrences_cell2 == 1, f"Cell Text 2 appeared {occurrences_cell2} times!"

    # In nested lists, child li text should not be extracted twice
    occurrences_deep_leaf = sum(t.count("Deep Leaf 1B-i") for t in texts)
    assert occurrences_deep_leaf == 1, f"Deep Leaf appeared {occurrences_deep_leaf} times!"


def test_html_identical_text_in_distinct_siblings_preserved():
    """Verify that identical text in sibling cells is preserved (not falsely deduplicated)."""
    html = b"""
    <table>
        <tr>
            <td>Status: OK</td>
            <td>Status: OK</td>
            <td>Status: OK</td>
        </tr>
    </table>
    """
    doc = parse_html(html, "status.html")
    texts = [s.text for s in doc.sections]
    # Sibling td elements are distinct tags, so each td should yield its section
    assert len(texts) == 3
    assert all(t == "Status: OK" for t in texts)


def test_html_strips_scripts_styles_nav_footer_header():
    """Verify dangerous / non-content tags are stripped completely even when nested."""
    html = b"""
    <div>
        <nav><p>Navigation link</p></nav>
        <header><h1>Banner Header</h1></header>
        <article>
            <p>Legitimate article content.</p>
            <script>var x = "malicious payload";</script>
            <style>body { background: red; }</style>
        </article>
        <footer><p>Copyright 2026</p></footer>
    </div>
    """
    doc = parse_html(html, "clean.html")
    texts = [s.text for s in doc.sections]
    assert len(texts) == 1
    assert "Legitimate article content." in texts[0]
    assert not any("Navigation link" in t for t in texts)
    assert not any("Banner Header" in t for t in texts)
    assert not any("malicious payload" in t for t in texts)
    assert not any("Copyright 2026" in t for t in texts)


def test_markdown_indented_code_blocks_and_empty_headers():
    """Stress-test parse_markdown with indented code blocks (4 sp, 8 sp, tabs) and empty headers."""
    md = (
        b"# \n\n## \n\n### Valid Heading 3\n\nParagraph under heading 3.\n\n"
        b"    def indented_four_spaces():\n        x = 10\n        return x * 2\n\n"
        b"Paragraph between code blocks.\n\n"
        b'        def indented_eight_spaces():\n            return "deep"\n\n'
        b"\tdef tab_indented_block():\n\t\tpass\n\n#### \n\n"
        b"##### Final Heading 5\n\nFinal paragraph content.\n"
    )
    doc = parse_markdown(md, "adversarial.md")
    texts = [s.text for s in doc.sections]

    # Verify heading was extracted
    headings = [s.heading for s in doc.sections if s.heading]
    assert "Valid Heading 3" in headings
    assert "Final Heading 5" in headings

    # Verify indented code blocks were preserved
    assert any("def indented_four_spaces():" in t for t in texts)
    assert any("def indented_eight_spaces():" in t for t in texts)
    assert any("def tab_indented_block():" in t for t in texts)

    # Verify paragraphs were not dropped by empty headers
    assert any("Paragraph under heading 3." in t for t in texts)
    assert any("Paragraph between code blocks." in t for t in texts)
    assert any("Final paragraph content." in t for t in texts)


def test_markdown_alternating_fences_and_indented_blocks():
    """Verify parser preserves alternating code fences and indented blocks without state leakage."""
    md = (
        b"# Code Showcase\n\n```python\ndef fence_one():\n    return 1\n```\n\n"
        b"    def indent_one():\n        return 2\n\n```javascript\n"
        b"function fenceTwo() {\n    return 3;\n}\n```\n\n"
        b"    def indent_two():\n        return 4\n\nFinal remarks.\n"
    )
    doc = parse_markdown(md, "alternating.md")
    texts = [s.text for s in doc.sections]

    assert any("def fence_one():" in t for t in texts)
    assert any("def indent_one():" in t for t in texts)
    assert any("function fenceTwo()" in t for t in texts)
    assert any("def indent_two():" in t for t in texts)
    assert any("Final remarks." in t for t in texts)


# ==============================================================================
# SECTION 3: Async Queue Thread Safety & PyStemmer Concurrency
# ==============================================================================


def test_pystemmer_high_concurrency_stress():
    """Spawn 10 concurrent threads repeatedly tokenizing, indexing, and searching with Bm25Store.

    Verifies thread-local PyStemmer isolation prevents C-buffer corruption or memory errors.
    """
    store = Bm25Store()
    num_threads = 10
    iterations_per_thread = 50
    barrier = threading.Barrier(num_threads)
    errors: list[Exception] = []
    thread_stemmer_ids: list[int] = []
    lock = threading.Lock()

    # Pre-populate index with multiple documents
    initial_docs = [
        (
            f"Document {i} covers machine learning, neural networks, "
            f"retrieval models, and NLP concepts {i * 7}."
        )
        for i in range(100)
    ]
    initial_ids = [f"doc-{i}" for i in range(100)]
    store.rebuild(initial_docs, initial_ids, "stress_strategy")

    def worker_stress(thread_id: int):
        try:
            barrier.wait()
            # Capture thread-local stemmer id
            s_id = id(store._stemmer)
            with lock:
                thread_stemmer_ids.append(s_id)

            queries = [
                "machine learning models",
                "retrieval concepts",
                "neural networks NLP",
                "Document 42",
                "nonexistent query term xyz",
            ]

            for it in range(iterations_per_thread):
                # 1. Tokenize directly
                sample_texts = [
                    f"Text sample {it} with technical tokens API_V1 and stemming {thread_id}."
                ]
                tokens = store._tokenize(sample_texts)
                assert len(tokens.ids) == 1

                # 2. Search index
                q = queries[it % len(queries)]
                results = store.search(q, "stress_strategy", top_k=10)
                assert isinstance(results, list)

                # 3. Periodically trigger rebuild in parallel from subset of threads
                if thread_id in (0, 5) and it % 10 == 0:
                    subset_docs = [f"Rebuilt doc {j} iteration {it}" for j in range(20)]
                    subset_ids = [f"rebuilt-{j}" for j in range(20)]
                    count = store.rebuild(subset_docs, subset_ids, "stress_strategy")
                    assert count == 20

        except Exception as e:
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker_stress, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    # Assert all threads completed cleanly with zero errors
    assert len(errors) == 0, f"Concurrency errors occurred: {errors}"
    assert len(thread_stemmer_ids) == num_threads
    # Each thread must have its own isolated PyStemmer instance
    assert len(set(thread_stemmer_ids)) == num_threads, (
        f"Stemmer instance collision detected! Expected {num_threads} distinct IDs, "
        f"got {len(set(thread_stemmer_ids))}"
    )


@pytest.mark.asyncio
async def test_job_queue_massive_multithread_fanout_and_subscribers():
    """Stress-test JobRegistry with 20 worker threads publishing SSE events.

    Verifies thread-safe loop.call_soon_threadsafe dispatch, no queue deadlocks, and clean shutdown.
    """
    registry = JobRegistry()
    job = registry.create_job("concurrency_fanout_test")
    num_subscribers = 10
    num_worker_threads = 20
    events_per_worker = 15

    subscriber_event_counts = [0] * num_subscribers
    subscriber_received_terminal = [False] * num_subscribers

    async def subscriber_task(sub_idx: int):
        async for event in registry.subscribe(job.job_id):
            subscriber_event_counts[sub_idx] += 1
            if event["event"] in ("completed", "error"):
                subscriber_received_terminal[sub_idx] = True
                break

    # Launch 10 concurrent subscriber tasks on the asyncio event loop
    tasks = [asyncio.create_task(subscriber_task(i)) for i in range(num_subscribers)]
    await asyncio.sleep(0.02)  # Give subscribers time to register

    # Verify all 10 subscribers registered in registry under lock
    with registry._lock:
        assert len(registry._subscribers[job.job_id]) == num_subscribers

    worker_errors: list[Exception] = []
    lock = threading.Lock()
    barrier = threading.Barrier(num_worker_threads)

    def worker_thread_fn(worker_idx: int):
        try:
            barrier.wait()
            for step in range(events_per_worker):
                progress = (
                    (worker_idx * events_per_worker + step)
                    / (num_worker_threads * events_per_worker)
                    * 90.0
                )
                registry.update_progress(job.job_id, progress, f"worker-{worker_idx}-step-{step}")
                time.sleep(0.001)
        except Exception as e:
            with lock:
                worker_errors.append(e)

    # Spawn 20 concurrent worker threads blasting progress events
    threads = [
        threading.Thread(target=worker_thread_fn, args=(i,)) for i in range(num_worker_threads)
    ]
    for t in threads:
        t.start()

    # Wait for all worker threads to finish their progress updates
    for t in threads:
        t.join(timeout=10.0)

    assert len(worker_errors) == 0, f"Worker thread errors: {worker_errors}"

    # Terminate the job
    registry.complete_job(job.job_id, {"status": "success", "workers": num_worker_threads})

    # Wait for all subscribers to receive terminal event and terminate
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

    # Verify all subscribers received events and terminated properly
    assert all(subscriber_received_terminal), "Some subscribers did not receive the terminal event"
    assert all(c > 0 for c in subscriber_event_counts), "Some subscribers received 0 events"

    # Verify subscriber cleanup
    with registry._lock:
        assert len(registry._subscribers[job.job_id]) == 0, "Subscriber queues were not cleaned up"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_state,result_data,expected_event",
    [
        ("completed", {"result": "success"}, "completed"),
        ("failed", "Fatal error occurred", "error"),
        ("cancelled", None, "cancelled"),
    ],
)
async def test_late_subscribers_receive_immediate_terminal_event(
    terminal_state: str, result_data: Any, expected_event: str
):
    """Verify that late subscribers connecting AFTER a job reaches a terminal state
    immediately receive the terminal event without hanging or deadlocking.
    """
    registry = JobRegistry()
    job = registry.create_job(f"late_sub_{terminal_state}")

    if terminal_state == "completed":
        registry.complete_job(job.job_id, result_data)
    elif terminal_state == "failed":
        registry.fail_job(job.job_id, result_data)
    elif terminal_state == "cancelled":
        registry.cancel_job(job.job_id)

    # Verify job is terminal
    assert job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    # 10 late subscribers connect concurrently
    async def late_subscriber():
        events = []
        async for event in registry.subscribe(job.job_id):
            events.append(event)
        return events

    # Wrap in timeout to detect any hanging / deadlock
    sub_tasks = [asyncio.create_task(late_subscriber()) for _ in range(10)]
    results = await asyncio.wait_for(asyncio.gather(*sub_tasks), timeout=2.0)

    assert len(results) == 10
    for events in results:
        assert len(events) == 1, f"Expected 1 event, got {len(events)}"
        assert events[0]["event"] == expected_event
        if terminal_state == "completed":
            assert events[0]["data"] == result_data
        elif terminal_state == "failed":
            assert events[0]["data"]["error"] == result_data

    # Verify registry has zero leaked subscribers
    with registry._lock:
        assert len(registry._subscribers.get(job.job_id, [])) == 0


@pytest.mark.asyncio
async def test_late_subscriber_race_during_state_transition():
    """Verify no race condition when a subscriber joins at the exact instant of job completion."""
    registry = JobRegistry()
    job = registry.create_job("race_test")

    start_event = threading.Event()
    race_subscriber_events = []

    def completer():
        start_event.wait()
        time.sleep(0.005)
        registry.complete_job(job.job_id, {"winner": "completer"})

    t = threading.Thread(target=completer)
    t.start()

    start_event.set()
    # Subscribe concurrently right as completion happens
    async for event in registry.subscribe(job.job_id):
        race_subscriber_events.append(event)

    t.join(timeout=2.0)

    assert len(race_subscriber_events) >= 1
    assert race_subscriber_events[-1]["event"] == "completed"
    assert race_subscriber_events[-1]["data"] == {"winner": "completer"}
