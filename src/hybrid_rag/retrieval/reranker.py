"""Cross-encoder reranker with sigmoid normalization and Lost-in-the-Middle reordering."""

from __future__ import annotations

import asyncio
import math
import threading
import time

from hybrid_rag.schemas.pipeline import CandidateResult, StageTrace
from hybrid_rag.settings import settings

_reranker = None
_lock = threading.Lock()
_inference_lock = threading.Lock()
_reranker_semaphore: asyncio.Semaphore | None = None


def _get_reranker_semaphore() -> asyncio.Semaphore:
    """Lazy initialize the asyncio semaphore on the active running loop."""
    global _reranker_semaphore
    if _reranker_semaphore is None:
        _reranker_semaphore = asyncio.Semaphore(1)
    return _reranker_semaphore


def _trim_memory() -> None:
    """Trigger Python garbage collection and trim glibc heap to prevent RSS creep."""
    import gc

    gc.collect()
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def get_reranker():
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                _reranker = TextCrossEncoder(
                    model_name=settings.reranker_model,
                    cache_dir=settings.fastembed_cache_path,
                    threads=1,
                )
    return _reranker


def warmup_reranker() -> None:
    """Pre-load model so first request avoids cold-start latency."""
    reranker = get_reranker()
    list(reranker.rerank("warmup", ["warmup passage"], top_k=1))


def sigmoid(x: float) -> float:
    """Normalize cross-encoder logit to [0.0, 1.0]."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def lost_in_middle_reorder(
    candidates: list[CandidateResult],
) -> list[CandidateResult]:
    """Place best results at context edges for LLM attention.

    [Rank 1, Rank 3, Rank 5, ..., Rank 4, Rank 2]
    """
    if len(candidates) <= 2:
        return [
            CandidateResult(
                chunk_id=c.chunk_id,
                score=c.score,
                rank=i,
                content=c.content,
                metadata=dict(c.metadata),
            )
            for i, c in enumerate(candidates, 1)
        ]

    front = candidates[::2]
    back = list(reversed(candidates[1::2]))
    reordered = front + back

    return [
        CandidateResult(
            chunk_id=c.chunk_id,
            score=c.score,
            rank=i,
            content=c.content,
            metadata=dict(c.metadata),
        )
        for i, c in enumerate(reordered, 1)
    ]


def rerank_candidates(
    query: str,
    candidates: list[CandidateResult],
    top_k: int = 5,
    apply_lost_in_middle: bool = True,
) -> StageTrace:
    """Rerank with cross-encoder, sigmoid-normalize scores, optionally LitM reorder."""
    start = time.perf_counter()

    if not candidates:
        duration_ms = (time.perf_counter() - start) * 1000
        return StageTrace(
            stage="rerank",
            candidates=[],
            duration_ms=duration_ms,
            metadata={"rerank_latency_ms": duration_ms},
        )

    reranker = get_reranker()
    documents = [c.content if c.content else c.chunk_id for c in candidates]

    # Serialize inference across threads to cap memory spikes
    with _inference_lock:
        raw_scores = list(reranker.rerank(query, documents))

    scored: list[tuple[CandidateResult, float, float]] = []
    for i, raw_score in enumerate(raw_scores):
        norm_score = sigmoid(float(raw_score))
        scored.append((candidates[i], norm_score, float(raw_score)))

    scored.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for rank, (original, norm_score, raw_score) in enumerate(scored[:top_k], 1):
        reranked.append(
            CandidateResult(
                chunk_id=original.chunk_id,
                score=norm_score,
                rank=rank,
                content=original.content,
                metadata={
                    **original.metadata,
                    "raw_rerank_score": raw_score,
                    "rerank_score": norm_score,
                    "pre_rerank_rank": original.rank,
                },
            )
        )

    if apply_lost_in_middle:
        reranked = lost_in_middle_reorder(reranked)

    duration_ms = (time.perf_counter() - start) * 1000
    return StageTrace(
        stage="rerank",
        candidates=reranked,
        duration_ms=duration_ms,
        metadata={
            "rerank_latency_ms": duration_ms,
            "lost_in_middle": apply_lost_in_middle,
            "candidates_in": len(candidates),
            "candidates_out": len(reranked),
        },
    )


async def arerank_candidates(
    query: str,
    candidates: list[CandidateResult],
    top_k: int = 5,
    apply_lost_in_middle: bool = True,
) -> StageTrace:
    """Async wrapper — serializes cross-encoder inference via semaphore to prevent OOM RAM spikes."""
    sem = _get_reranker_semaphore()
    async with sem:
        result = await asyncio.to_thread(
            rerank_candidates, query, candidates, top_k, apply_lost_in_middle
        )
    _trim_memory()
    return result
