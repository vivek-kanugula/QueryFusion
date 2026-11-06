"""Query Expansion and Hypothetical Document Embeddings (HyDE) for multi-query retrieval."""

from __future__ import annotations

import json

from hybrid_rag.generation.llm_client import call_llm

EXPANSION_SYSTEM_INSTRUCTION = """\
You are an expert search query formulation specialist for technical information retrieval.
Given a user query, output:
1. Two to three alternative, high-recall search queries covering synonyms and acronyms.
2. A brief hypothetical answer passage (1-2 sentences) containing plausible terminology (HyDE).

Output strictly valid JSON:
{
  "queries": ["alternative query 1", "alternative query 2"],
  "hypothetical_passage": "concise hypothetical passage"
}
"""

EXPANSION_PROMPT_TEMPLATE = """\
User Search Query:
{query}

Generate search expansions and a hypothetical passage.
Strictly return JSON:
{{"queries": ["..."], "hypothetical_passage": "..."}}
"""


def expand_query(
    query: str,
    model: str | None = None,
    provider: str | None = None,
    max_expansions: int = 3,
) -> list[str]:
    """Generate expanded query variations including the original query.

    Returns a list of unique query strings with the original query first.
    """
    clean_query = query.strip()
    if not clean_query:
        return [query]

    prompt = EXPANSION_PROMPT_TEMPLATE.format(query=clean_query)

    try:
        raw = call_llm(
            prompt=prompt,
            system_instruction=EXPANSION_SYSTEM_INSTRUCTION,
            model=model,
            provider=provider,
            temperature=0.3,
        )
        data = json.loads(raw)
        expansions = [clean_query]

        for q in data.get("queries", []):
            q_clean = str(q).strip()
            if q_clean and q_clean.lower() != clean_query.lower() and q_clean not in expansions:
                expansions.append(q_clean)

        hypo = str(data.get("hypothetical_passage", "")).strip()
        if hypo and hypo not in expansions:
            expansions.append(hypo)

        return expansions[: max_expansions + 1]
    except Exception:
        # Resilient fallback on any model or parsing error
        return [clean_query]
