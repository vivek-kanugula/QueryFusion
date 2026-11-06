"""RAGAS offline adapter: faithfulness + answer relevancy with local embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from hybrid_rag.generation.cache import LLMCache, TokenBucketRateLimiter


@dataclass
class RagasScores:
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    nan_count: int = 0
    total_count: int = 0

    @property
    def nan_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.nan_count / self.total_count


@dataclass
class RagasRunResult:
    per_question: list[dict[str, Any]] = field(default_factory=list)
    aggregate: RagasScores = field(default_factory=RagasScores)


def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def aggregate_scores(
    per_question: list[dict[str, float | None]],
) -> RagasScores:
    faith_values: list[float] = []
    relevancy_values: list[float] = []
    nan_count = 0
    total = len(per_question)

    for item in per_question:
        faith = item.get("faithfulness")
        relevancy = item.get("answer_relevancy")

        item_has_nan = False
        if not _is_nan(faith):
            faith_values.append(float(faith))
        else:
            item_has_nan = True

        if not _is_nan(relevancy):
            relevancy_values.append(float(relevancy))
        else:
            item_has_nan = True

        if item_has_nan:
            nan_count += 1

    faith_avg = sum(faith_values) / len(faith_values) if faith_values else None
    rel_avg = sum(relevancy_values) / len(relevancy_values) if relevancy_values else None

    return RagasScores(
        faithfulness=faith_avg,
        answer_relevancy=rel_avg,
        nan_count=nan_count,
        total_count=total,
    )


class LocalEmbeddingsWrapper:
    """Wraps fastembed bge-small-en-v1.5 for RAGAS answer relevancy."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model_name = model_name
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        embeddings = list(model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
    llm_cache: LLMCache | None = None,
    rate_limiter: TokenBucketRateLimiter | None = None,
) -> RagasRunResult:
    """Run RAGAS faithfulness + answer_relevancy offline.

    Imports ragas lazily (heavy deps, eval-only).
    """
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    from hybrid_rag.settings import settings

    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths is not None:
        dataset_dict["ground_truth"] = ground_truths

    try:
        from datasets import Dataset

        dataset = Dataset.from_dict(dataset_dict)
    except ImportError:
        raise ImportError("datasets package required for RAGAS evaluation")

    embeddings = LocalEmbeddingsWrapper(settings.embedding_model)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        embeddings=embeddings,
    )

    per_question: list[dict[str, float | None]] = []
    result_df = result.to_pandas()
    for _, row in result_df.iterrows():
        scores: dict[str, float | None] = {}
        for metric_name in ["faithfulness", "answer_relevancy"]:
            val = row.get(metric_name)
            if _is_nan(val):
                scores[metric_name] = None
            else:
                scores[metric_name] = float(val)
        per_question.append(scores)

    agg = aggregate_scores(per_question)

    return RagasRunResult(per_question=per_question, aggregate=agg)
