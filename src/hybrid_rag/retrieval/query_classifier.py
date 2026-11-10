"""Heuristic query difficulty classifier.

Categorizes queries into difficulty classes and recommends retrieval alpha.
No LLM calls — pure regex + token heuristics for zero-latency classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class QueryCategory(StrEnum):
    EXACT_MATCH = "exact_match"
    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    SEMANTIC = "semantic"


CATEGORY_ALPHA: dict[QueryCategory, float] = {
    QueryCategory.EXACT_MATCH: 0.2,
    QueryCategory.FACTUAL: 0.5,
    QueryCategory.MULTI_HOP: 0.6,
    QueryCategory.SEMANTIC: 0.7,
}

CATEGORY_DESCRIPTION: dict[QueryCategory, str] = {
    QueryCategory.EXACT_MATCH: "Looks for specific identifiers, error codes, or quoted terms",
    QueryCategory.FACTUAL: "Simple factual question (who/what/when/where)",
    QueryCategory.MULTI_HOP: "Complex question requiring multiple pieces of information",
    QueryCategory.SEMANTIC: "Conceptual or explanatory question needing semantic understanding",
}


@dataclass(frozen=True)
class ClassificationResult:
    category: QueryCategory
    confidence: float
    alpha: float
    reason: str


_QUOTED = re.compile(r'"[^"]{2,}"')
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_ERROR_CODE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}(?:Error|Exception|Warning|Code)\b")
_IDENTIFIER = re.compile(r"\b(?:0x[0-9a-fA-F]+|[A-Z][A-Z0-9_]{3,})\b")
_VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
_FACTUAL_START = re.compile(
    r"^\s*(?:what|who|when|where|which|how\s+(?:many|much|long|often))\b", re.I
)
_MULTI_HOP_MARKERS = re.compile(
    r"\b(?:compare|differ(?:ence)?|versus|vs\.?|between|relationship|"
    r"both|all|each|every|combined|together|affect|impact|cause)\b",
    re.I,
)
_CONJUNCTION_CHAIN = re.compile(r"\b(?:and|or|but|while|whereas|although)\b", re.I)
_SEMANTIC_START = re.compile(
    r"^\s*(?:explain|describe|why|how\s+does|how\s+do|how\s+can|how\s+should|"
    r"what\s+is\s+the\s+(?:purpose|reason|concept|idea|meaning)|"
    r"in\s+what\s+way)\b",
    re.I,
)
_ABSTRACT_TERMS = re.compile(
    r"\b(?:concept|principle|approach|strategy|pattern|methodology|"
    r"architecture|philosophy|paradigm|trade-?off|best\s+practice|"
    r"advantage|disadvantage|implication|overview)\b",
    re.I,
)


def classify_query(query: str) -> ClassificationResult:
    """Classify a query into a difficulty category with recommended alpha."""
    q = query.strip()
    if not q:
        return ClassificationResult(
            category=QueryCategory.FACTUAL,
            confidence=0.5,
            alpha=CATEGORY_ALPHA[QueryCategory.FACTUAL],
            reason="Empty query, defaulting to factual",
        )

    scores: dict[QueryCategory, float] = {c: 0.0 for c in QueryCategory}

    has_quotes = bool(_QUOTED.search(q))
    has_uuid = bool(_UUID.search(q))
    has_error_code = bool(_ERROR_CODE.search(q))
    has_identifier = bool(_IDENTIFIER.search(q))
    has_version = bool(_VERSION.search(q))

    exact_signals = sum([has_quotes, has_uuid, has_error_code, has_identifier, has_version])
    if exact_signals >= 2:
        scores[QueryCategory.EXACT_MATCH] += 0.9
    elif exact_signals == 1:
        scores[QueryCategory.EXACT_MATCH] += 0.75
    if has_quotes:
        scores[QueryCategory.EXACT_MATCH] += 0.2
    if has_uuid:
        scores[QueryCategory.EXACT_MATCH] += 0.2

    if _FACTUAL_START.search(q):
        scores[QueryCategory.FACTUAL] += 0.5
    word_count = len(q.split())
    if word_count <= 8:
        scores[QueryCategory.FACTUAL] += 0.2

    conjunctions = len(_CONJUNCTION_CHAIN.findall(q))
    multi_hop_markers = len(_MULTI_HOP_MARKERS.findall(q))
    if multi_hop_markers >= 2 or conjunctions >= 2:
        scores[QueryCategory.MULTI_HOP] += 0.7
    elif multi_hop_markers >= 1:
        scores[QueryCategory.MULTI_HOP] += 0.4
    if "?" in q and q.count("?") > 1:
        scores[QueryCategory.MULTI_HOP] += 0.3

    if _SEMANTIC_START.search(q):
        scores[QueryCategory.SEMANTIC] += 0.6
    abstract_count = len(_ABSTRACT_TERMS.findall(q))
    if abstract_count >= 2:
        scores[QueryCategory.SEMANTIC] += 0.5
    elif abstract_count == 1:
        scores[QueryCategory.SEMANTIC] += 0.3
    if word_count >= 12:
        scores[QueryCategory.SEMANTIC] += 0.1

    best = max(scores, key=lambda c: scores[c])
    best_score = scores[best]

    if best_score < 0.3:
        best = QueryCategory.FACTUAL
        best_score = 0.4
        reason = "No strong signals; defaulting to factual"
    else:
        reasons = []
        if best == QueryCategory.EXACT_MATCH:
            parts = []
            if has_quotes:
                parts.append("quoted terms")
            if has_uuid:
                parts.append("UUID")
            if has_error_code:
                parts.append("error code")
            if has_identifier:
                parts.append("identifier")
            if has_version:
                parts.append("version number")
            reasons.append(f"Contains {', '.join(parts)}")
        elif best == QueryCategory.FACTUAL:
            reasons.append("Simple factual question pattern")
        elif best == QueryCategory.MULTI_HOP:
            reasons.append(f"Multiple concepts ({multi_hop_markers} comparison markers, {conjunctions} conjunctions)")
        elif best == QueryCategory.SEMANTIC:
            reasons.append("Conceptual/explanatory question pattern")
        reason = "; ".join(reasons)

    confidence = min(1.0, best_score / 1.0)

    return ClassificationResult(
        category=best,
        confidence=round(confidence, 2),
        alpha=CATEGORY_ALPHA[best],
        reason=reason,
    )
