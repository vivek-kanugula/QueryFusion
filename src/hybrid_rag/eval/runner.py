"""Experiment runner with reproducibility manifests and per-question traces."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from hybrid_rag.eval.dataset import ChunkInfo, GoldenItem, map_spans_to_chunks
from hybrid_rag.eval.metrics import compute_all
from hybrid_rag.eval.stats import (
    ConfidenceInterval,
    bootstrap_ci,
    paired_significance,
    win_loss_tie,
)
from hybrid_rag.schemas.pipeline import PipelineConfig


@dataclass
class RunManifest:
    git_sha: str
    config_hash: str
    dataset_hash: str
    corpus_snapshot_id: str
    model_names: dict[str, str]
    timestamp: str
    python_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "corpus_snapshot_id": self.corpus_snapshot_id,
            "model_names": self.model_names,
            "timestamp": self.timestamp,
            "python_version": self.python_version,
        }


@dataclass
class QuestionResult:
    question_id: str
    category: str = ""
    retrieved_ids: list[str] = field(default_factory=list)
    relevant_ids: set[str] = field(default_factory=set)
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    question: str = ""


@dataclass
class ExperimentResult:
    name: str
    manifest: RunManifest = field(default_factory=lambda: RunManifest("", "", "", "", {}, ""))
    config: PipelineConfig = field(default_factory=PipelineConfig)
    overall_metrics: dict[str, ConfidenceInterval] = field(default_factory=dict)
    per_category: dict[str, dict[str, ConfidenceInterval]] = field(default_factory=dict)
    per_question: list[QuestionResult] = field(default_factory=list)
    strategy: str = ""
    chunking_strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "manifest": self.manifest.to_dict(),
            "config": {
                "retrieval_strategy": self.config.retrieval_strategy,
                "chunking_strategy": self.config.chunking_strategy,
                "alpha": self.config.alpha,
                "rrf_k": self.config.rrf_k,
                "rerank_enabled": self.config.rerank_enabled,
                "rerank_top_k": self.config.rerank_top_k,
            },
            "overall_metrics": {
                k: {"mean": v.mean, "lower": v.lower, "upper": v.upper}
                for k, v in self.overall_metrics.items()
            },
            "n_questions": len(self.per_question),
        }


# ── Helpers ──


def config_hash(config: PipelineConfig) -> str:
    canonical = json.dumps(
        {
            "retrieval_strategy": config.retrieval_strategy,
            "chunking_strategy": config.chunking_strategy,
            "alpha": config.alpha,
            "rrf_k": config.rrf_k,
            "dense_top_k": config.dense_top_k,
            "sparse_top_k": config.sparse_top_k,
            "fusion_top_k": config.fusion_top_k,
            "rerank_top_k": config.rerank_top_k,
            "rerank_enabled": config.rerank_enabled,
            "lost_in_middle": config.lost_in_middle,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_manifest(
    config: PipelineConfig,
    dataset_hash_value: str,
    corpus_snapshot_id: str,
    model_names: dict[str, str] | None = None,
) -> RunManifest:
    import sys

    return RunManifest(
        git_sha=_git_sha(),
        config_hash=config_hash(config),
        dataset_hash=dataset_hash_value,
        corpus_snapshot_id=corpus_snapshot_id,
        model_names=model_names or {},
        timestamp=datetime.now(UTC).isoformat(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )


# ── Single-question evaluation ──


def run_single_question(
    item: GoldenItem,
    config: PipelineConfig,
    retrieve_fn: Any,
    chunks: list[ChunkInfo],
    k: int = 5,
    overlap_threshold: float = 0.5,
    doc_id_map: dict[str, str] | None = None,
) -> QuestionResult:
    relevant_ids = map_spans_to_chunks(
        item.source_spans, chunks, overlap_threshold, doc_id_map=doc_id_map
    )

    start = time.perf_counter()
    retrieved_ids = retrieve_fn(item.question, config)
    latency_ms = (time.perf_counter() - start) * 1000

    metrics = compute_all(retrieved_ids, relevant_ids, k)

    return QuestionResult(
        question_id=item.id,
        category=item.category,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        metrics=metrics,
        latency_ms=latency_ms,
    )


# ── Full experiment run ──


def run_experiment(
    name: str,
    config: PipelineConfig,
    dataset: list[GoldenItem],
    retrieve_fn: Any,
    chunks: list[ChunkInfo],
    dataset_hash_value: str,
    corpus_snapshot_id: str = "",
    k: int = 5,
    overlap_threshold: float = 0.5,
    model_names: dict[str, str] | None = None,
    doc_id_map: dict[str, str] | None = None,
) -> ExperimentResult:
    manifest = build_manifest(config, dataset_hash_value, corpus_snapshot_id, model_names)
    question_results: list[QuestionResult] = []

    for item in dataset:
        result = run_single_question(
            item, config, retrieve_fn, chunks, k, overlap_threshold, doc_id_map=doc_id_map
        )
        question_results.append(result)

    overall_metrics = _aggregate_metrics(question_results)
    category_metrics = _aggregate_by_category(question_results)

    return ExperimentResult(
        name=name,
        manifest=manifest,
        config=config,
        overall_metrics=overall_metrics,
        per_category=category_metrics,
        per_question=question_results,
    )


def _aggregate_metrics(
    results: list[QuestionResult],
) -> dict[str, ConfidenceInterval]:
    if not results:
        return {}
    metric_names = list(results[0].metrics.keys())
    out: dict[str, ConfidenceInterval] = {}
    for metric in metric_names:
        scores = np.array([r.metrics[metric] for r in results])
        out[metric] = bootstrap_ci(scores)
    latencies = np.array([r.latency_ms for r in results])
    out["latency_ms"] = bootstrap_ci(latencies)
    return out


def _aggregate_by_category(
    results: list[QuestionResult],
) -> dict[str, dict[str, ConfidenceInterval]]:
    categories: dict[str, list[QuestionResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)
    out: dict[str, dict[str, ConfidenceInterval]] = {}
    for cat, cat_results in categories.items():
        out[cat] = _aggregate_metrics(cat_results)
    return out


# ── Comparative experiments ──


def compare_experiments(
    result_a: ExperimentResult,
    result_b: ExperimentResult,
    metric: str = "ndcg_at_k",
) -> dict[str, Any]:
    map_a = {r.question_id: r.metrics.get(metric, 0.0) for r in result_a.per_question}
    map_b = {r.question_id: r.metrics.get(metric, 0.0) for r in result_b.per_question}

    common_ids = [qid for qid in map_a if qid in map_b]
    if not common_ids:
        raise ValueError("No common questions found between experiments")

    scores_a = np.array([map_a[qid] for qid in common_ids], dtype=float)
    scores_b = np.array([map_b[qid] for qid in common_ids], dtype=float)

    stat, p_value = paired_significance(scores_a, scores_b)
    wlt = win_loss_tie(scores_a, scores_b)
    delta_ci = bootstrap_ci(scores_a - scores_b)

    return {
        "metric": metric,
        "experiment_a": result_a.name,
        "experiment_b": result_b.name,
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "delta_mean": float(np.mean(scores_a) - np.mean(scores_b)),
        "delta_ci": {"lower": delta_ci.lower, "upper": delta_ci.upper},
        "wilcoxon_statistic": stat,
        "wilcoxon_p_value": p_value,
        "statistic": stat,
        "p_value": p_value,
        "wins_losses_ties": wlt,
        "win_loss_tie": (wlt["wins"], wlt["losses"], wlt["ties"]),
        "significant_at_005": p_value < 0.05,
    }
