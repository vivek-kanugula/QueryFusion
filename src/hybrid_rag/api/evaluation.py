"""Evaluation results API: serves stored experiment metrics to the dashboard."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from hybrid_rag.db.engine import get_db
from hybrid_rag.models.evaluation import EvaluationRun

router = APIRouter(
    prefix="/api/v1/evaluation",
    tags=["evaluation"],
)


def _avg(values: list[float | None]) -> float:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else 0.0


@router.get("/results")
def get_evaluation_results(db: Session = Depends(get_db)):
    runs = (
        db.query(EvaluationRun)
        .order_by(EvaluationRun.created_at.desc())
        .all()
    )

    if not runs:
        return {"has_data": False, "strategies": []}

    latest_by_strategy: dict[str, EvaluationRun] = {}
    for run in runs:
        if run.retrieval_strategy not in latest_by_strategy:
            latest_by_strategy[run.retrieval_strategy] = run

    strategies = []
    for strategy, run in latest_by_strategy.items():
        metrics = run.metrics
        if not metrics:
            continue

        agg = {
            "precision_at_k": _avg([m.precision_at_k for m in metrics]),
            "recall_at_k": _avg([m.recall_at_k for m in metrics]),
            "mrr": _avg([m.mrr for m in metrics]),
            "ndcg_at_k": _avg([m.ndcg_at_k for m in metrics]),
            "faithfulness_score": _avg([m.faithfulness_score for m in metrics]),
            "relevancy_score": _avg([m.relevancy_score for m in metrics]),
            "citation_accuracy": _avg([m.citation_accuracy for m in metrics]),
            "citation_coverage": _avg([m.citation_coverage for m in metrics]),
        }

        latency = None
        if run.latency_breakdown_json:
            try:
                latency = json.loads(run.latency_breakdown_json)
            except (json.JSONDecodeError, TypeError):
                pass

        strategies.append({
            "id": str(run.id),
            "experiment_name": run.experiment_name,
            "retrieval_strategy": run.retrieval_strategy,
            "chunking_strategy": run.chunking_strategy,
            "created_at": run.created_at.isoformat(),
            "n_questions": len(metrics),
            "metrics": agg,
            "latency_breakdown": latency,
        })

    return {"has_data": bool(strategies), "strategies": strategies}
