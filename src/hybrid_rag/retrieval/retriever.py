"""Dense, Sparse, and Hybrid retrievers with alpha-RRF fusion.

Dense: pgvector cosine search (bge-small-en-v1.5 with query instruction prefix)
Sparse: BM25 via Bm25Store (identifier-preserving tokenizer)
Hybrid: concurrent Dense + Sparse, fused with weighted Reciprocal Rank Fusion
"""

from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sqlalchemy.orm import Session

from hybrid_rag.retrieval.bm25_store import Bm25Store
from hybrid_rag.schemas.pipeline import (
    CandidateResult,
    PipelineConfig,
    QueryTrace,
    StageTrace,
)

logger = logging.getLogger(__name__)

_retrieval_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")


def dense_retrieve(
    query: str,
    config: PipelineConfig,
    db: Any = None,
) -> StageTrace:
    start = time.perf_counter()

    from hybrid_rag.ingest.embedder import embed_query

    query_vec = embed_query(query)

    from hybrid_rag.db.engine import SessionLocal
    from hybrid_rag.models.document import Document, DocumentChunk

    def _query_pgvector(session):
        distance_col = DocumentChunk.embedding.cosine_distance(query_vec.tolist())
        filter_conditions = [
            Document.chunking_strategy == config.chunking_strategy,
            DocumentChunk.embedding.isnot(None),
        ]
        if config.document_ids:
            doc_uuids = []
            for d in config.document_ids:
                try:
                    doc_uuids.append(uuid.UUID(str(d)))
                except (ValueError, AttributeError):
                    pass
            if not doc_uuids:
                return []
            filter_conditions.append(DocumentChunk.document_id.in_(doc_uuids))

        rows = (
            session.query(DocumentChunk, distance_col.label("distance"))
            .join(Document)
            .filter(*filter_conditions)
            .order_by(distance_col.asc())
            .limit(config.dense_top_k)
            .all()
        )
        items = []
        for rank, (chunk, distance) in enumerate(rows, 1):
            sim_score = max(0.0, 1.0 - float(distance)) if distance is not None else 0.0
            items.append(
                CandidateResult(
                    chunk_id=str(chunk.id),
                    score=sim_score,
                    rank=rank,
                    content=chunk.content,
                    metadata={
                        "document_id": str(chunk.document_id),
                        "source": "dense",
                    },
                )
            )
        return items

    if db is not None and hasattr(db, "query"):
        candidates = _query_pgvector(db)
    else:
        with SessionLocal() as session:
            candidates = _query_pgvector(session)

    duration_ms = (time.perf_counter() - start) * 1000
    return StageTrace(
        stage="dense",
        candidates=candidates,
        duration_ms=duration_ms,
        metadata={"dense_latency_ms": duration_ms},
    )


def sparse_retrieve(query: str, config: PipelineConfig, bm25_store: Bm25Store) -> StageTrace:
    start = time.perf_counter()

    results = bm25_store.search(
        query,
        config.chunking_strategy,
        config.sparse_top_k,
        document_ids=config.document_ids,
    )

    candidates = []
    for rank, (chunk_id, score) in enumerate(results, 1):
        candidates.append(
            CandidateResult(
                chunk_id=chunk_id,
                score=score,
                rank=rank,
                metadata={"source": "sparse"},
            )
        )

    duration_ms = (time.perf_counter() - start) * 1000
    return StageTrace(
        stage="sparse",
        candidates=candidates,
        duration_ms=duration_ms,
        metadata={"sparse_latency_ms": duration_ms},
    )


def rrf_fuse(
    dense_candidates: list[CandidateResult],
    sparse_candidates: list[CandidateResult],
    alpha: float,
    k: int,
    fusion_top_k: int,
) -> StageTrace:
    """Weighted Reciprocal Rank Fusion.

    score(d) = sum_{m in {dense, sparse} : d in tau_m} (weight_m / (k + rank_m(d)))
    alpha=1.0 -> pure dense, alpha=0.0 -> pure sparse.
    """
    start = time.perf_counter()

    if alpha == 1.0:
        fused = _passthrough(dense_candidates, fusion_top_k, "dense")
        return _rrf_stage(fused, start)

    if alpha == 0.0:
        fused = _passthrough(sparse_candidates, fusion_top_k, "sparse")
        return _rrf_stage(fused, start)

    dense_ranks = {c.chunk_id: c.rank for c in dense_candidates}
    sparse_ranks = {c.chunk_id: c.rank for c in sparse_candidates}

    # Deterministic sequence: preserve dense order followed by sparse order
    seen: set[str] = set()
    ordered_ids: list[str] = []
    for c in dense_candidates:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            ordered_ids.append(c.chunk_id)
    for c in sparse_candidates:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            ordered_ids.append(c.chunk_id)

    # Retain source candidate references to preserve content and metadata
    source_candidates: dict[str, CandidateResult] = {}
    for c in sparse_candidates:
        source_candidates[c.chunk_id] = c
    for c in dense_candidates:
        if c.chunk_id in source_candidates:
            existing = source_candidates[c.chunk_id]
            if not existing.content and c.content:
                source_candidates[c.chunk_id] = c
        else:
            source_candidates[c.chunk_id] = c

    scored: list[tuple[str, float, int | None, int | None]] = []
    for chunk_id in ordered_ids:
        d_rank = dense_ranks.get(chunk_id)
        s_rank = sparse_ranks.get(chunk_id)
        score = 0.0
        if d_rank is not None:
            score += alpha / (k + d_rank)
        if s_rank is not None:
            score += (1.0 - alpha) / (k + s_rank)
        scored.append((chunk_id, score, d_rank, s_rank))

    scored.sort(key=lambda x: x[1], reverse=True)

    fused = []
    for rank, (chunk_id, score, d_rank, s_rank) in enumerate(scored[:fusion_top_k], 1):
        in_dense = chunk_id in dense_ranks
        in_sparse = chunk_id in sparse_ranks
        src_c = source_candidates.get(chunk_id)
        base_meta = dict(src_c.metadata) if src_c and src_c.metadata else {}
        base_meta["dense_rank"] = d_rank if in_dense else None
        base_meta["sparse_rank"] = s_rank if in_sparse else None
        base_meta["rank_delta"] = (
            abs(d_rank - s_rank)
            if in_dense and in_sparse and d_rank is not None and s_rank is not None
            else None
        )
        content = src_c.content if src_c else ""
        fused.append(
            CandidateResult(
                chunk_id=chunk_id,
                score=score,
                rank=rank,
                content=content,
                metadata=base_meta,
            )
        )

    return _rrf_stage(fused, start)


def _passthrough(
    candidates: list[CandidateResult], top_k: int, source: str
) -> list[CandidateResult]:
    rank_key = "dense_rank" if source == "dense" else "sparse_rank"
    other_key = "sparse_rank" if source == "dense" else "dense_rank"
    return [
        CandidateResult(
            chunk_id=c.chunk_id,
            score=c.score,
            rank=i,
            content=c.content,
            metadata={
                **c.metadata,
                rank_key: c.rank,
                other_key: None,
                "rank_delta": None,
            },
        )
        for i, c in enumerate(candidates[:top_k], 1)
    ]


def _rrf_stage(fused: list[CandidateResult], start: float) -> StageTrace:
    duration_ms = (time.perf_counter() - start) * 1000
    return StageTrace(
        stage="rrf",
        candidates=fused,
        duration_ms=duration_ms,
        metadata={"rrf_latency_ms": duration_ms},
    )


def hybrid_retrieve(
    query: str,
    config: PipelineConfig,
    db: Any = None,
    bm25_store: Bm25Store | None = None,
) -> QueryTrace:
    """Full hybrid retrieval: concurrent dense + sparse, then RRF fusion."""
    if bm25_store is None:
        from hybrid_rag.retrieval.bm25_store import get_bm25_store

        bm25_store = get_bm25_store()

    trace = QueryTrace(query=query, config=config)
    total_start = time.perf_counter()

    dense_future = _retrieval_pool.submit(dense_retrieve, query, config, None)
    sparse_future = _retrieval_pool.submit(sparse_retrieve, query, config, bm25_store)

    dense_stage = dense_future.result()
    sparse_stage = sparse_future.result()

    trace.add_stage(dense_stage)
    trace.add_stage(sparse_stage)

    rrf_stage = rrf_fuse(
        dense_stage.candidates,
        sparse_stage.candidates,
        alpha=config.alpha,
        k=config.rrf_k,
        fusion_top_k=config.fusion_top_k,
    )
    trace.add_stage(rrf_stage)

    trace.total_duration_ms = (time.perf_counter() - total_start) * 1000
    return trace


def full_retrieve(
    query: str,
    config: PipelineConfig,
    bm25_store: Bm25Store | None = None,
    db: Session | None = None,
) -> QueryTrace:
    """Full pipeline: route by strategy -> hydrate -> rerank -> Lost-in-the-Middle."""
    if config.expand_queries:
        from hybrid_rag.retrieval.expansion import expand_query

        expanded = expand_query(query)
        if len(expanded) > 1:
            sub_config = PipelineConfig(
                retrieval_strategy=config.retrieval_strategy,
                chunking_strategy=config.chunking_strategy,
                alpha=config.alpha,
                rrf_k=config.rrf_k,
                dense_top_k=config.dense_top_k,
                sparse_top_k=config.sparse_top_k,
                fusion_top_k=config.fusion_top_k,
                rerank_top_k=config.rerank_top_k,
                rerank_enabled=config.rerank_enabled,
                lost_in_middle=config.lost_in_middle,
                temperature=config.temperature,
                document_ids=config.document_ids,
                expand_queries=False,
            )
            return multi_query_retrieve(expanded, sub_config, bm25_store=bm25_store, db=db)

    strat = (config.retrieval_strategy or "hybrid_rerank").lower()

    if strat in ("dense_only", "dense"):
        trace = QueryTrace(query=query, config=config)
        start = time.perf_counter()
        stage = dense_retrieve(query, config, db)
        trace.add_stage(stage)
        trace.total_duration_ms = (time.perf_counter() - start) * 1000
        candidates = stage.candidates
    elif strat in ("bm25_only", "bm25", "sparse"):
        if bm25_store is None:
            from hybrid_rag.retrieval.bm25_store import get_bm25_store

            bm25_store = get_bm25_store()
        trace = QueryTrace(query=query, config=config)
        start = time.perf_counter()
        stage = sparse_retrieve(query, config, bm25_store)
        trace.add_stage(stage)
        trace.total_duration_ms = (time.perf_counter() - start) * 1000
        candidates = stage.candidates
    else:
        if bm25_store is None:
            from hybrid_rag.retrieval.bm25_store import get_bm25_store

            bm25_store = get_bm25_store()
        trace = hybrid_retrieve(query, config, db, bm25_store)
        candidates = trace.stages[-1].candidates if trace.stages else []

    if candidates:
        hydrate_db = db
        if hydrate_db is not None:
            hydrate_candidates(candidates, hydrate_db)
        else:
            try:
                from hybrid_rag.db.engine import SessionLocal

                with SessionLocal() as fallback_db:
                    hydrate_candidates(candidates, fallback_db)
            except Exception:
                logger.warning(
                    "Candidate hydration failed; candidates may lack content",
                    exc_info=True,
                )

        if config.document_ids:
            allowed = {str(d) for d in config.document_ids}
            candidates = [c for c in candidates if c.metadata.get("document_id") in allowed]
            trace.stages[-1].candidates = candidates

    if not config.rerank_enabled or not candidates:
        return trace

    try:
        from hybrid_rag.retrieval.reranker import rerank_candidates

        rerank_stage = rerank_candidates(
            query=query,
            candidates=candidates,
            top_k=config.rerank_top_k,
            apply_lost_in_middle=config.lost_in_middle,
        )
        trace.add_stage(rerank_stage)
    except Exception:
        logger.warning("Reranker failed; returning unranked candidates", exc_info=True)

    return trace


def hydrate_candidates(candidates: list[CandidateResult], db: Session) -> None:
    """Fill content field and document_id metadata from PostgreSQL for candidates missing it."""
    from hybrid_rag.models.document import DocumentChunk

    ids_to_fetch = [
        c.chunk_id for c in candidates if not c.content or not c.metadata.get("document_id")
    ]
    if not ids_to_fetch:
        return

    uuids = []
    for cid in ids_to_fetch:
        try:
            uuids.append(uuid.UUID(str(cid)))
        except (ValueError, AttributeError):
            pass
    if not uuids:
        return

    try:
        chunks = db.query(DocumentChunk).filter(DocumentChunk.id.in_(uuids)).all()
    except Exception:
        return

    chunk_map = {str(c.id): c for c in chunks}
    for candidate in candidates:
        if candidate.chunk_id in chunk_map:
            chunk = chunk_map[candidate.chunk_id]
            if not candidate.content and chunk.content:
                candidate.content = chunk.content
            if not candidate.metadata.get("document_id") and chunk.document_id:
                candidate.metadata["document_id"] = str(chunk.document_id)


def multi_query_retrieve(
    queries: list[str],
    config: PipelineConfig,
    bm25_store: Bm25Store | None = None,
    db: Session | None = None,
) -> QueryTrace:
    """Retrieve and fuse candidates across multiple expanded query variations."""
    if not queries:
        return QueryTrace(query="", config=config)

    if len(queries) == 1:
        return full_retrieve(queries[0], config, bm25_store=bm25_store, db=db)

    primary_query = queries[0]
    traces: list[QueryTrace] = []
    strat = (config.retrieval_strategy or "hybrid_rerank").lower()
    if strat not in ("dense_only", "dense") and bm25_store is None:
        from hybrid_rag.retrieval.bm25_store import get_bm25_store

        bm25_store = get_bm25_store()

    for q in queries:
        if strat in ("dense_only", "dense"):
            t = QueryTrace(query=q, config=config)
            start = time.perf_counter()
            s = dense_retrieve(q, config, db)
            t.add_stage(s)
            t.total_duration_ms = (time.perf_counter() - start) * 1000
            traces.append(t)
        elif strat in ("bm25_only", "bm25", "sparse"):
            t = QueryTrace(query=q, config=config)
            start = time.perf_counter()
            s = sparse_retrieve(q, config, bm25_store)
            t.add_stage(s)
            t.total_duration_ms = (time.perf_counter() - start) * 1000
            traces.append(t)
        else:
            traces.append(hybrid_retrieve(q, config, db=db, bm25_store=bm25_store))

    rrf_k = config.rrf_k
    scores: dict[str, float] = {}
    candidate_map: dict[str, CandidateResult] = {}

    for trace in traces:
        final_candidates = trace.stages[-1].candidates if trace.stages else []
        for rank, c in enumerate(final_candidates, 1):
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + (1.0 / (rrf_k + rank))
            if c.chunk_id not in candidate_map or c.content:
                candidate_map[c.chunk_id] = c

    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    fused_candidates: list[CandidateResult] = []
    for rank, cid in enumerate(sorted_chunk_ids[: config.fusion_top_k], 1):
        orig = candidate_map[cid]
        fused_candidates.append(
            CandidateResult(
                chunk_id=cid,
                score=scores[cid],
                rank=rank,
                content=orig.content,
                metadata={
                    **orig.metadata,
                    "multi_query_fused": True,
                },
            )
        )

    if db is not None:
        hydrate_candidates(fused_candidates, db)
    else:
        try:
            from hybrid_rag.db.engine import SessionLocal

            with SessionLocal() as fallback_db:
                hydrate_candidates(fused_candidates, fallback_db)
        except Exception:
            logger.warning(
                "Candidate hydration failed; candidates may lack content",
                exc_info=True,
            )

    if config.document_ids:
        allowed = {str(d) for d in config.document_ids}
        fused_candidates = [
            c for c in fused_candidates if c.metadata.get("document_id") in allowed
        ]

    merged_trace = QueryTrace(
        query=primary_query,
        config=config,
    )

    merged_trace.add_stage(
        StageTrace(
            stage="multi_query_rrf",
            candidates=fused_candidates,
            duration_ms=sum(t.total_duration_ms for t in traces),
            metadata={"expanded_queries": queries},
        )
    )

    if config.rerank_enabled and fused_candidates:
        try:
            from hybrid_rag.retrieval.reranker import rerank_candidates

            rerank_stage = rerank_candidates(
                query=primary_query,
                candidates=fused_candidates,
                top_k=config.rerank_top_k,
                apply_lost_in_middle=config.lost_in_middle,
            )
            merged_trace.add_stage(rerank_stage)
        except Exception:
            logger.warning("Reranker failed in multi-query; returning unranked", exc_info=True)

    return merged_trace
