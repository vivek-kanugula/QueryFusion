"""Query Expansion and Hypothetical Document Embeddings (HyDE) for multi-query retrieval."""

from __future__ import annotations

import json

from hybrid_rag.generation.cache import (
    LLMCache,
    TokenBucketRateLimiter,
    cache_key,
    retry_with_backoff,
)
from hybrid_rag.generation.llm_client import call_llm
from hybrid_rag.settings import settings

_default_cache = LLMCache()
_default_limiter = TokenBucketRateLimiter(rate=15.0 / 60.0, capacity=15)

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


def _format_expansions(clean_query: str, data: dict, max_expansions: int) -> list[str]:
    expansions = [clean_query]
    for q in data.get("queries", []):
        q_clean = str(q).strip()
        if q_clean and q_clean.lower() != clean_query.lower() and q_clean not in expansions:
            expansions.append(q_clean)

    hypo = str(data.get("hypothetical_passage", "")).strip()
    if hypo and hypo not in expansions:
        expansions.append(hypo)

    return expansions[: max_expansions + 1]


def expand_query(
    query: str,
    model: str | None = None,
    provider: str | None = None,
    max_expansions: int = 3,
    llm_cache: LLMCache | None = None,
    rate_limiter: TokenBucketRateLimiter | None = None,
) -> list[str]:
    """Generate expanded query variations with caching and rate limiting.

    Returns a list of unique query strings with the original query first.
    """
    clean_query = query.strip()
    if not clean_query:
        return [query]

    cache = llm_cache or _default_cache
    limiter = rate_limiter or _default_limiter

    resolved_model = model or settings.gemini_model
    prompt = EXPANSION_PROMPT_TEMPLATE.format(query=clean_query)
    cache_tag = f"{provider}:{resolved_model}" if provider else resolved_model
    key = cache_key(
        cache_tag,
        prompt,
        "query_expansion",
        0.3,
        system_instruction=EXPANSION_SYSTEM_INSTRUCTION,
    )

    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            try:
                data = json.loads(cached)
                return _format_expansions(clean_query, data, max_expansions)
            except Exception:
                pass

    if limiter is not None and not limiter.acquire():
        return [clean_query]

    try:
        raw = retry_with_backoff(
            lambda: call_llm(
                prompt=prompt,
                system_instruction=EXPANSION_SYSTEM_INSTRUCTION,
                model=resolved_model,
                provider=provider,
                temperature=0.3,
            )
        )
        if cache is not None and raw:
            cache.put(key, raw, model=resolved_model)

        data = json.loads(raw)
        return _format_expansions(clean_query, data, max_expansions)
    except Exception:
        # Resilient fallback on any model or parsing error
        return [clean_query]
