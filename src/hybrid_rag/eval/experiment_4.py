"""Experiment 4: end-to-end quality benchmark across retrieval configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hybrid_rag.eval.citation_metrics import CitationScores, compute_citation_metrics
from hybrid_rag.eval.latency import LatencyProfiler
from hybrid_rag.eval.metrics import compute_all
from hybrid_rag.eval.runner import (
    RunManifest,
    build_manifest,
)
from hybrid_rag.eval.stats import ConfidenceInterval, bootstrap_ci
from hybrid_rag.schemas.generation import GenerationResponse
from hybrid_rag.schemas.pipeline import PipelineConfig

EXPERIMENT_CONFIGS = {
    "dense_only": PipelineConfig(
        retrieval_strategy="dense_only",
        alpha=1.0,
        rerank_enabled=False,
        lost_in_middle=False,
    ),
    "hybrid": PipelineConfig(
        retrieval_strategy="hybrid",
        alpha=0.5,
        rerank_enabled=False,
        lost_in_middle=False,
    ),
    "hybrid_rerank": PipelineConfig(
        retrieval_strategy="hybrid_rerank",
        alpha=0.5,
        rerank_enabled=True,
        lost_in_middle=True,
    ),
}


@dataclass
class Exp4QuestionResult:
    question_id: str
    category: str
    retrieved_ids: list[str]
    relevant_ids: set[str]
    ir_metrics: dict[str, float]
    generation_metrics: dict[str, float]
    latency_breakdown: dict[str, float]
    latency_ms: float = 0.0


@dataclass
class Exp4Result:
    name: str
    manifest: RunManifest
    config: PipelineConfig
    ir_overall: dict[str, ConfidenceInterval]
    gen_overall: dict[str, ConfidenceInterval]
    per_question: list[Exp4QuestionResult]
    latency_profiler: LatencyProfiler = field(
        default_factory=LatencyProfiler,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "manifest": self.manifest.to_dict(),
            "config": {
                "retrieval_strategy": self.config.retrieval_strategy,
                "alpha": self.config.alpha,
                "rerank_enabled": self.config.rerank_enabled,
            },
            "ir_overall": {
                k: {"mean": v.mean, "lower": v.lower, "upper": v.upper}
                for k, v in self.ir_overall.items()
            },
            "gen_overall": {
                k: {"mean": v.mean, "lower": v.lower, "upper": v.upper}
                for k, v in self.gen_overall.items()
            },
            "n_questions": len(self.per_question),
            "latency": self.latency_profiler.to_dict(),
        }


def extract_generation_metrics(
    gen_response: GenerationResponse | None,
    citation_scores: CitationScores | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    if gen_response is None:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "citation_accuracy": 0.0,
            "composite_confidence": 0.0,
        }

    if citation_scores is not None:
        metrics["citation_accuracy"] = citation_scores.citation_accuracy
        metrics["citation_precision"] = citation_scores.citation_precision

    if gen_response.confidence is not None:
        metrics["composite_confidence"] = gen_response.confidence.overall_score
    else:
        metrics["composite_confidence"] = 0.0

    return metrics


def run_experiment_4_single(
    question_id: str,
    category: str,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    gen_response: GenerationResponse | None,
    stage_durations: dict[str, float] | None = None,
    k: int = 5,
) -> Exp4QuestionResult:
    ir = compute_all(retrieved_ids, relevant_ids, k)

    citation_scores = None
    if gen_response is not None and gen_response.verification is not None:
        citation_scores = compute_citation_metrics(
            gen_response.verification,
        )

    gen_metrics = extract_generation_metrics(gen_response, citation_scores)
    latency = stage_durations or {}
    total_latency = sum(latency.values())

    return Exp4QuestionResult(
        question_id=question_id,
        category=category,
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        ir_metrics=ir,
        generation_metrics=gen_metrics,
        latency_breakdown=latency,
        latency_ms=total_latency,
    )


def aggregate_exp4(
    results: list[Exp4QuestionResult],
) -> tuple[dict[str, ConfidenceInterval], dict[str, ConfidenceInterval]]:
    if not results:
        return {}, {}

    ir_keys = list(results[0].ir_metrics.keys())
    ir_overall: dict[str, ConfidenceInterval] = {}
    for key in ir_keys:
        scores = np.array([r.ir_metrics[key] for r in results])
        ir_overall[key] = bootstrap_ci(scores)

    all_gen_keys: set[str] = set()
    for r in results:
        all_gen_keys.update(r.generation_metrics.keys())
    gen_overall: dict[str, ConfidenceInterval] = {}
    for key in sorted(all_gen_keys):
        scores = np.array([r.generation_metrics.get(key, 0.0) for r in results])
        gen_overall[key] = bootstrap_ci(scores)

    return ir_overall, gen_overall


def run_experiment_4(
    name: str,
    config: PipelineConfig,
    question_results: list[Exp4QuestionResult],
    dataset_hash_value: str,
    corpus_snapshot_id: str = "",
    model_names: dict[str, str] | None = None,
    environment: str = "local",
) -> Exp4Result:
    manifest = build_manifest(
        config,
        dataset_hash_value,
        corpus_snapshot_id,
        model_names,
    )

    profiler = LatencyProfiler(environment=environment)
    for qr in question_results:
        profiler.record_from_trace(qr.latency_breakdown)

    ir_overall, gen_overall = aggregate_exp4(question_results)

    return Exp4Result(
        name=name,
        manifest=manifest,
        config=config,
        ir_overall=ir_overall,
        gen_overall=gen_overall,
        per_question=question_results,
        latency_profiler=profiler,
    )


def compare_exp4(
    result_a: Exp4Result,
    result_b: Exp4Result,
    ir_metric: str = "ndcg_at_k",
    gen_metric: str = "citation_accuracy",
) -> dict[str, Any]:
    ir_a = np.array([r.ir_metrics.get(ir_metric, 0.0) for r in result_a.per_question])
    ir_b = np.array([r.ir_metrics.get(ir_metric, 0.0) for r in result_b.per_question])

    gen_a = np.array([r.generation_metrics.get(gen_metric, 0.0) for r in result_a.per_question])
    gen_b = np.array([r.generation_metrics.get(gen_metric, 0.0) for r in result_b.per_question])

    from hybrid_rag.eval.stats import paired_significance

    ir_stat, ir_p = paired_significance(ir_a, ir_b)
    gen_stat, gen_p = paired_significance(gen_a, gen_b)

    ir_delta_ci = bootstrap_ci(ir_a - ir_b)
    gen_delta_ci = bootstrap_ci(gen_a - gen_b)

    return {
        "experiment_a": result_a.name,
        "experiment_b": result_b.name,
        "ir": {
            "metric": ir_metric,
            "mean_a": float(np.mean(ir_a)),
            "mean_b": float(np.mean(ir_b)),
            "delta_ci": {
                "lower": ir_delta_ci.lower,
                "upper": ir_delta_ci.upper,
            },
            "wilcoxon_p": ir_p,
            "significant": ir_p < 0.05,
        },
        "gen": {
            "metric": gen_metric,
            "mean_a": float(np.mean(gen_a)),
            "mean_b": float(np.mean(gen_b)),
            "delta_ci": {
                "lower": gen_delta_ci.lower,
                "upper": gen_delta_ci.upper,
            },
            "wilcoxon_p": gen_p,
            "significant": gen_p < 0.05,
        },
    }
