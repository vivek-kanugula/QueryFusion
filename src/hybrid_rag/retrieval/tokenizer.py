"""Identifier-preserving BM25 tokenizer for technical documentation.

Expands technical identifiers (DB-204, config.json, error_code_500) into
both a composite token and constituent sub-tokens, enabling exact-phrase
and partial matching in BM25 indices.
"""

from __future__ import annotations

import re

import Stemmer as PyStemmer

TECH_PATTERN = re.compile(r"\b([A-Za-z0-9]+(?:[-_.:/\\]+[A-Za-z0-9]+)+)\b")
WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")

ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "or",
        "our",
        "own",
        "s",
        "she",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "up",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

_stemmer: PyStemmer.Stemmer | None = None


def get_stemmer() -> PyStemmer.Stemmer:
    global _stemmer
    if _stemmer is None:
        _stemmer = PyStemmer.Stemmer("english")
    return _stemmer


def expand_technical_tokens(text: str) -> str:
    """Expand technical identifiers to composite + sub-tokens.

    "Found DB-204 error" -> "Found DB204 DB 204 error"

    Composite token has separators stripped so BM25 tokenizers keep it
    as one token. Sub-tokens enable partial matching.
    """

    def _expand(match: re.Match) -> str:
        token = match.group(1)
        parts = re.split(r"[-_.:/\\]+", token)
        parts = [p for p in parts if p]
        if len(parts) <= 1:
            return token
        composite = "".join(parts)
        return f"{composite} {' '.join(parts)}"

    return TECH_PATTERN.sub(_expand, text)


def tokenize_technical(text: str) -> list[str]:
    """Full tokenization: expand, split, lowercase, stem, remove stopwords."""
    expanded = expand_technical_tokens(text)
    stemmer = get_stemmer()
    words = WORD_PATTERN.findall(expanded.lower())
    tokens = []
    for w in words:
        if w not in ENGLISH_STOPWORDS:
            stemmed = stemmer.stemWord(w)
            if stemmed:
                tokens.append(stemmed)
    return tokens
