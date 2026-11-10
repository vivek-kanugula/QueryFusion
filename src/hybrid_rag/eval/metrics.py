"""Standard IR evaluation metrics with binary relevance.

All functions treat retrieved as an ordered list of document/chunk IDs
and relevant as a set of IDs that are ground-truth relevant.
"""

from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    top_k = retrieved[:k]
    dcg = 0.0
    seen_relevant: set[str] = set()
    for i, doc_id in enumerate(top_k):
        if doc_id in relevant and doc_id not in seen_relevant:
            dcg += 1.0 / math.log2(i + 2)
            seen_relevant.add(doc_id)
    n_relevant_in_ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant_in_ideal))
    if idcg == 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


def compute_all(retrieved: list[str], relevant: set[str], k: int) -> dict[str, float]:
    return {
        "precision_at_k": precision_at_k(retrieved, relevant, k),
        "recall_at_k": recall_at_k(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant),
        "ndcg_at_k": ndcg_at_k(retrieved, relevant, k),
    }
