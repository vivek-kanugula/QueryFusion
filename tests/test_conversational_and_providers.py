"""Tests for conversational RAG contextualization and multi-provider LLM dispatcher."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hybrid_rag.api.query import router
from hybrid_rag.generation.contextualizer import contextualize_query, format_history
from hybrid_rag.generation.generator import (
    build_generation_prompt,
    generate_answer,
)
from hybrid_rag.generation.llm_client import (
    _call_groq,
    _call_ollama,
    resolve_provider,
)
from hybrid_rag.schemas.pipeline import CandidateResult


def _make_candidate(chunk_id: str = "c1", content: str = "Doc content") -> CandidateResult:
    return CandidateResult(
        chunk_id=chunk_id,
        score=0.9,
        rank=1,
        content=content,
        metadata={"document_id": "doc1"},
    )


# ── Provider Resolution ──


def test_resolve_provider_gemini_defaults():
    prov, model = resolve_provider(None)
    assert prov == "gemini"
    assert "gemini" in model

    prov, model = resolve_provider("gemini-3.8-flash")
    assert prov == "gemini"
    assert model == "gemini-3.8-flash"


def test_resolve_provider_groq():
    prov, model = resolve_provider("llama-3.3-70b-versatile")
    assert prov == "groq"
    assert model == "llama-3.3-70b-versatile"

    prov, model = resolve_provider("groq/llama-3.1-8b-instant")
    assert prov == "groq"
    assert model == "llama-3.1-8b-instant"


def test_resolve_provider_ollama():
    prov, model = resolve_provider("ollama/llama3.2")
    assert prov == "ollama"
    assert model == "llama3.2"


def test_resolve_provider_explicit():
    prov, model = resolve_provider("custom-model", explicit_provider="groq")
    assert prov == "groq"
    assert model == "custom-model"


# ── LLM Dispatching Mocks ──


@patch("urllib.request.urlopen")
def test_call_groq_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        {"choices": [{"message": {"content": '{"answer": "Groq response"}'}}]}
    ).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test-key"}):
        result = _call_groq(
            model_name="llama-3.3-70b-versatile",
            prompt="Hello",
            system_instruction="Be helpful",
            temperature=0.1,
        )
    assert '{"answer": "Groq response"}' in result
    assert mock_urlopen.called


@patch("urllib.request.urlopen")
def test_call_ollama_success(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"response": '{"answer": "Ollama response"}'}).encode(
        "utf-8"
    )
    mock_resp.__enter__.return_value = mock_resp
    mock_urlopen.return_value = mock_resp

    result = _call_ollama(
        model_name="llama3.2",
        prompt="Hello",
        system_instruction="Be helpful",
        temperature=0.1,
    )
    assert '{"answer": "Ollama response"}' in result
    assert mock_urlopen.called


def test_call_groq_missing_key():
    with patch("hybrid_rag.settings.settings.groq_api_key", None), patch.dict(
        "os.environ", {}, clear=True
    ):
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            _call_groq("llama-3.3-70b-versatile", "p", "s", 0.0)


# ── Contextualizer & History ──


def test_format_history():
    history = [
        {"role": "user", "content": "What is dense retrieval?"},
        {"role": "assistant", "content": "Dense retrieval uses vector embeddings."},
    ]
    formatted = format_history(history)
    assert "User: What is dense retrieval?" in formatted
    assert "Assistant: Dense retrieval uses vector embeddings." in formatted


def test_contextualize_query_empty_history():
    standalone = contextualize_query("What is BM25?", chat_history=[])
    assert standalone == "What is BM25?"


@patch("hybrid_rag.generation.contextualizer._default_cache.get", return_value=None)
@patch("hybrid_rag.generation.contextualizer.call_llm")
def test_contextualize_query_resolves_pronoun(mock_call, mock_cache_get):
    mock_call.return_value = json.dumps(
        {"standalone_query": "How does BM25 calculate document frequency?"}
    )
    history = [{"role": "user", "content": "Tell me about BM25 algorithm."}]
    standalone = contextualize_query("How does it calculate document frequency?", history)
    assert standalone == "How does BM25 calculate document frequency?"
    assert mock_call.called


@patch("hybrid_rag.generation.contextualizer.call_llm")
def test_contextualize_query_fallback_on_error(mock_call):
    mock_call.side_effect = RuntimeError("API unavailable")
    history = [{"role": "user", "content": "Tell me about BM25."}]
    standalone = contextualize_query("How does it work?", history)
    assert standalone == "How does it work?"


# ── Generator with History & Dispatch ──


def test_build_generation_prompt_with_history():
    history = [{"role": "user", "content": "First turn."}]
    prompt = build_generation_prompt("Second turn?", [_make_candidate()], chat_history=history)
    assert "<conversation_history>" in prompt
    assert "User: First turn." in prompt
    assert "<user_question>" in prompt
    assert "Second turn?" in prompt


@patch("hybrid_rag.generation.generator._call_gemini")
def test_generate_answer_gemini_dispatch(mock_call):
    mock_call.return_value = json.dumps(
        {
            "answer": "Grounded answer [C1].",
            "citations": [{"citation_id": "[C1]", "evidence_quote": "evidence"}],
            "has_sufficient_context": True,
        }
    )
    ans, cached = generate_answer(
        "Follow up?",
        [_make_candidate()],
        chat_history=[{"role": "user", "content": "Prior"}],
        model="gemini-3.5-flash-lite",
        provider="gemini",
    )
    assert "Grounded answer" in ans.answer
    assert mock_call.called


@patch("hybrid_rag.generation.llm_client._call_groq")
def test_generate_answer_groq_dispatch(mock_groq):
    mock_groq.return_value = json.dumps(
        {
            "answer": "Groq grounded answer [C1].",
            "citations": [{"citation_id": "[C1]", "evidence_quote": "evidence"}],
            "has_sufficient_context": True,
        }
    )
    ans, cached = generate_answer(
        "Follow up?",
        [_make_candidate()],
        model="llama-3.3-70b-versatile",
        provider="groq",
    )
    assert "Groq grounded answer" in ans.answer
    assert mock_groq.called


# ── Streaming API with History and Provider ──


@patch("hybrid_rag.api.query.trace_store")
def test_stream_query_with_chat_history_and_provider(mock_trace_store):
    from hybrid_rag.schemas.generation import (
        Citation,
        CompositeConfidence,
        GenerationResponse,
        GroundedAnswer,
    )
    from hybrid_rag.schemas.pipeline import PipelineConfig, QueryTrace, StageTrace

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_trace = QueryTrace(
        trace_id="conv-trace",
        query="What is BM25 formula?",
        config=PipelineConfig(retrieval_strategy="hybrid_rerank", chunking_strategy="fixed"),
        stages=[StageTrace(stage="dense", candidates=[_make_candidate()], duration_ms=10.0)],
    )
    mock_gen = GenerationResponse(
        query="What is BM25 formula?",
        answer=GroundedAnswer(
            answer="BM25 is a ranking function [C1].",
            citations=[Citation(citation_id="[C1]", chunk_id="c1", evidence_quote="quote")],
            has_sufficient_context=True,
        ),
        confidence=CompositeConfidence(
            overall_score=0.9,
            retrieval_score=0.9,
            citation_coverage=1.0,
            completeness_score=0.8,
        ),
        trace_id="conv-trace",
        model="llama-3.3-70b-versatile",
    )

    with (
        patch(
            "hybrid_rag.generation.contextualizer.contextualize_query",
            return_value="What is BM25 ranking function formula?",
        ) as mock_ctx,
        patch(
            "hybrid_rag.retrieval.retriever.full_retrieve",
            return_value=mock_trace,
        ) as mock_retrieve,
        patch(
            "hybrid_rag.generation.pipeline.generate_grounded_answer",
            return_value=mock_gen,
        ) as mock_generate,
        patch("hybrid_rag.retrieval.bm25_store.get_bm25_store"),
    ):
        resp = client.post(
            "/api/query/stream",
            json={
                "question": "Can you explain its formula?",
                "chat_history": [{"role": "user", "content": "What is BM25?"}],
                "model": "llama-3.3-70b-versatile",
                "provider": "groq",
            },
        )

    assert resp.status_code == 200
    assert mock_ctx.called
    assert mock_retrieve.called
    # Verified full_retrieve received the contextualized query
    assert mock_retrieve.call_args[0][0] == "What is BM25 ranking function formula?"
    assert mock_generate.called

    # Check SSE events
    events = []
    for block in resp.text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        evt_type = "message"
        data = {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                evt_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        events.append((evt_type, data))

    stage_names = [d.get("name") for t, d in events if t == "stage"]
    assert "contextualize" in stage_names

    completed_events = [d for t, d in events if t == "completed"]
    assert len(completed_events) == 1
    assert completed_events[0]["contextualized_query"] == "What is BM25 ranking function formula?"
