"""Query contextualization using conversation history for conversational multi-turn RAG."""

from __future__ import annotations

import json

from hybrid_rag.generation.llm_client import call_llm

CONTEXTUALIZE_SYSTEM_INSTRUCTION = """\
You are an expert search query reformulation assistant. \
Given a chat conversation history and a follow-up user question, \
rephrase the follow-up question into a standalone, keyword-rich search query \
suitable for a hybrid document retrieval engine.

Rules:
1. Do NOT answer the question.
2. Resolve all ambiguous pronouns ("it", "they", "this", "these", "the second one", "that point").
3. If the question is already fully self-contained, return it unchanged.
4. Output strictly valid JSON with a single key "standalone_query".
"""

CONTEXTUALIZE_PROMPT_TEMPLATE = """\
Conversation History:
{history_text}

Follow-up Question:
{question}

Return strictly JSON:
{{"standalone_query": "rephrased standalone query"}}
"""


def format_history(chat_history: list[dict[str, str]] | None, max_turns: int = 4) -> str:
    """Format recent conversation turns into readable text."""
    if not chat_history:
        return ""

    recent = chat_history[-max_turns:]
    lines = []
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def contextualize_query(
    question: str,
    chat_history: list[dict[str, str]] | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> str:
    """Contextualize a follow-up question into a standalone query using conversation history."""
    if not chat_history:
        return question.strip()

    history_text = format_history(chat_history)
    if not history_text.strip():
        return question.strip()

    prompt = CONTEXTUALIZE_PROMPT_TEMPLATE.format(
        history_text=history_text,
        question=question.strip(),
    )

    try:
        raw = call_llm(
            prompt=prompt,
            system_instruction=CONTEXTUALIZE_SYSTEM_INSTRUCTION,
            model=model,
            provider=provider,
            temperature=0.0,
        )
        data = json.loads(raw)
        standalone = str(data.get("standalone_query", "")).strip()
        if standalone:
            return standalone
    except Exception:
        # Graceful fallback to original question on any failure
        pass

    return question.strip()
