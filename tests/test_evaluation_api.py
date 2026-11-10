"""Unit tests for the evaluation results API endpoint."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from hybrid_rag.app import app
from hybrid_rag.db.engine import get_db
from hybrid_rag.models.evaluation import EvaluationMetric, EvaluationRun


@pytest.fixture
def client_and_mock_db():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    client = TestClient(app)
    yield client, mock_db
    app.dependency_overrides.clear()


def test_get_evaluation_results_empty(client_and_mock_db):
    client, mock_db = client_and_mock_db
    mock_db.query.return_value.order_by.return_value.all.return_value = []

    response = client.get("/api/v1/evaluation/results")
    assert response.status_code == 200
    data = response.json()
    assert data["has_data"] is False
    assert data["strategies"] == []


def test_get_evaluation_results_with_data(client_and_mock_db):
    client, mock_db = client_and_mock_db

    mock_metric1 = MagicMock(spec=EvaluationMetric)
    mock_metric1.precision_at_k = 0.8
    mock_metric1.recall_at_k = 0.7
    mock_metric1.mrr = 0.9
    mock_metric1.ndcg_at_k = 0.85
    mock_metric1.faithfulness_score = 0.95
    mock_metric1.relevancy_score = 0.88
    mock_metric1.citation_accuracy = 1.0
    mock_metric1.citation_coverage = 0.9

    mock_metric2 = MagicMock(spec=EvaluationMetric)
    mock_metric2.precision_at_k = 0.6
    mock_metric2.recall_at_k = 0.5
    mock_metric2.mrr = 0.7
    mock_metric2.ndcg_at_k = 0.65
    mock_metric2.faithfulness_score = 0.85
    mock_metric2.relevancy_score = 0.72
    mock_metric2.citation_accuracy = 0.8
    mock_metric2.citation_coverage = 0.7

    mock_run = MagicMock(spec=EvaluationRun)
    mock_run.id = "run-uuid-1234"
    mock_run.experiment_name = "test_experiment"
    mock_run.retrieval_strategy = "hybrid"
    mock_run.chunking_strategy = "semantic"
    mock_run.created_at.isoformat.return_value = "2026-11-10T00:00:00"
    mock_run.metrics = [mock_metric1, mock_metric2]
    mock_run.latency_breakdown_json = json.dumps({"p50": 120, "p90": 250})

    mock_db.query.return_value.order_by.return_value.all.return_value = [mock_run]

    response = client.get("/api/v1/evaluation/results")
    assert response.status_code == 200
    data = response.json()
    assert data["has_data"] is True
    assert len(data["strategies"]) == 1
    strat = data["strategies"][0]
    assert strat["retrieval_strategy"] == "hybrid"
    assert strat["n_questions"] == 2
    assert pytest.approx(strat["metrics"]["precision_at_k"], 0.01) == 0.7
    assert pytest.approx(strat["metrics"]["mrr"], 0.01) == 0.8
    assert strat["latency_breakdown"] == {"p50": 120, "p90": 250}

