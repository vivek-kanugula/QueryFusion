"""Per-stage latency profiler with percentile statistics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    p50: float
    p90: float
    p95: float
    min: float
    max: float
    mean: float
    count: int


@dataclass
class StageLatencies:
    stage: str
    durations_ms: list[float] = field(default_factory=list)

    def stats(self) -> LatencyStats:
        if not self.durations_ms:
            return LatencyStats(
                p50=0.0,
                p90=0.0,
                p95=0.0,
                min=0.0,
                max=0.0,
                mean=0.0,
                count=0,
            )
        arr = np.array(self.durations_ms, dtype=float)
        return LatencyStats(
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            count=len(arr),
        )


STAGE_NAMES = [
    "query_embedding",
    "vector_search",
    "bm25_search",
    "rrf_fusion",
    "cross_encoder_rerank",
    "generation",
    "verification",
]


@dataclass
class LatencyProfiler:
    stages: dict[str, StageLatencies] = field(default_factory=dict)
    environment: str = "local"
    warmup_runs: int = 0

    def __post_init__(self) -> None:
        for name in STAGE_NAMES:
            if name not in self.stages:
                self.stages[name] = StageLatencies(stage=name)

    def record(self, stage: str, duration_ms: float) -> None:
        if stage not in self.stages:
            self.stages[stage] = StageLatencies(stage=stage)
        self.stages[stage].durations_ms.append(duration_ms)

    def record_from_trace(
        self,
        stage_durations: dict[str, float],
        is_warmup: bool = False,
    ) -> None:
        if is_warmup:
            self.warmup_runs += 1
            return
        for stage, duration_ms in stage_durations.items():
            self.record(stage, duration_ms)

    def summary(self) -> dict[str, LatencyStats]:
        return {name: sl.stats() for name, sl in self.stages.items() if sl.durations_ms}

    def total_latency_stats(self) -> LatencyStats:
        all_totals: list[float] = []
        stage_lists = [sl.durations_ms for sl in self.stages.values() if sl.durations_ms]
        if not stage_lists:
            return LatencyStats(
                p50=0.0,
                p90=0.0,
                p95=0.0,
                min=0.0,
                max=0.0,
                mean=0.0,
                count=0,
            )
        n_runs = max(len(sl) for sl in stage_lists)
        for i in range(n_runs):
            total = 0.0
            for sl in stage_lists:
                if i < len(sl):
                    total += sl[i]
            all_totals.append(total)

        arr = np.array(all_totals, dtype=float)
        return LatencyStats(
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            count=len(arr),
        )

    def to_dict(self) -> dict:
        summary = self.summary()
        return {
            "environment": self.environment,
            "warmup_runs_excluded": self.warmup_runs,
            "stages": {
                name: {
                    "p50": s.p50,
                    "p90": s.p90,
                    "p95": s.p95,
                    "min": s.min,
                    "max": s.max,
                    "mean": s.mean,
                    "count": s.count,
                }
                for name, s in summary.items()
            },
        }
