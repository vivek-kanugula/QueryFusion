"""Core pipeline configuration and trace schemas.

PipelineConfig drives every stage of the retrieval/generation pipeline.
QueryTrace captures per-stage outputs for diagnostics, evaluation, and the 4-column UI.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    retrieval_strategy: str = "hybrid_rerank"  # dense_only | bm25_only | hybrid | hybrid_rerank
    chunking_strategy: str = "fixed"  # fixed | structure | semantic
    alpha: float = 0.5  # RRF weight: 1.0 = dense only, 0.0 = BM25 only
    rrf_k: int = 60
    dense_top_k: int = 50
    sparse_top_k: int = 50
    fusion_top_k: int = 20
    rerank_top_k: int = 5
    rerank_enabled: bool = True
    lost_in_middle: bool = True
    temperature: float = 0.0
    document_ids: tuple[str, ...] | None = None


@dataclass
class CandidateResult:
    chunk_id: str
    score: float
    rank: int
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageTrace:
    stage: str
    candidates: list[CandidateResult] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    config: PipelineConfig = field(default_factory=PipelineConfig)
    stages: list[StageTrace] = field(default_factory=list)
    total_duration_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def add_stage(self, stage: StageTrace) -> None:
        self.stages.append(stage)
        self.total_duration_ms = sum(s.duration_ms for s in self.stages)
