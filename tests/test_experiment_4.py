"""Tests for Experiment 4 pipeline: orchestration, aggregation, comparison."""

from hybrid_rag.eval.experiment_4 import (
    EXPERIMENT_CONFIGS,
    Exp4QuestionResult,
    aggregate_exp4,
    compare_exp4,
    extract_generation_metrics,
    run_experiment_4,
    run_experiment_4_single,
)
from hybrid_rag.schemas.generation import (
    Citation,
    ClaimVerification,
    CompositeConfidence,
    GenerationResponse,
    GroundedAnswer,
    VerificationResult,
    VerificationVerdict,
)


def _make_gen_response(
    confidence: float = 0.8,
    supported: int = 2,
    unsupported: int = 0,
) -> GenerationResponse:
    claims = [
        ClaimVerification(
            claim_text=f"c{i}",
            verdict=VerificationVerdict.SUPPORTED,
            explanation="ok",
        )
        for i in range(supported)
    ] + [
        ClaimVerification(
            claim_text=f"u{i}",
            verdict=VerificationVerdict.UNSUPPORTED,
            explanation="no",
        )
        for i in range(unsupported)
    ]
    total = len(claims)
    cov = supported / total if total > 0 else 0.0

    return GenerationResponse(
        query="test?",
        answer=GroundedAnswer(
            answer="answer [C1]",
            citations=[Citation(citation_id="[C1]", chunk_id="c1")],
            has_sufficient_context=True,
        ),
        verification=VerificationResult(
            claims=claims,
            citation_coverage=cov,
            all_supported=(unsupported == 0 and total > 0),
        ),
        confidence=CompositeConfidence(
            overall_score=confidence,
            retrieval_score=0.9,
            citation_coverage=cov,
            completeness_score=1.0,
        ),
        model="gemini-3.5-flash-lite",
    )


# ── Configs ──


def test_experiment_configs_exist():
    assert "dense_only" in EXPERIMENT_CONFIGS
    assert "hybrid" in EXPERIMENT_CONFIGS
    assert "hybrid_rerank" in EXPERIMENT_CONFIGS


def test_dense_only_config():
    c = EXPERIMENT_CONFIGS["dense_only"]
    assert c.alpha == 1.0
    assert c.rerank_enabled is False


def test_hybrid_rerank_config():
    c = EXPERIMENT_CONFIGS["hybrid_rerank"]
    assert c.rerank_enabled is True
    assert c.lost_in_middle is True


# ── extract_generation_metrics ──


def test_extract_metrics_none():
    m = extract_generation_metrics(None)
    assert m["citation_accuracy"] == 0.0
    assert m["composite_confidence"] == 0.0


def test_extract_metrics_with_response():
    gen = _make_gen_response(confidence=0.85, supported=3, unsupported=1)
    from hybrid_rag.eval.citation_metrics import compute_citation_metrics

    cs = compute_citation_metrics(gen.verification)
    m = extract_generation_metrics(gen, cs)
    assert abs(m["citation_accuracy"] - 0.75) < 1e-9
    assert abs(m["composite_confidence"] - 0.85) < 1e-9


# ── run_experiment_4_single ──


def test_single_question():
    gen = _make_gen_response()
    result = run_experiment_4_single(
        question_id="q1",
        category="factual",
        retrieved_ids=["a", "b", "c"],
        relevant_ids={"a", "c"},
        gen_response=gen,
        stage_durations={"generation": 100.0, "verification": 50.0},
        k=3,
    )
    assert result.question_id == "q1"
    assert result.ir_metrics["precision_at_k"] > 0
    assert result.ir_metrics["recall_at_k"] > 0
    assert "citation_accuracy" in result.generation_metrics
    assert result.latency_ms == 150.0


def test_single_question_no_generation():
    result = run_experiment_4_single(
        question_id="q2",
        category="unanswerable",
        retrieved_ids=["a"],
        relevant_ids=set(),
        gen_response=None,
        k=5,
    )
    assert result.generation_metrics["composite_confidence"] == 0.0
    assert result.latency_ms == 0.0


# ── aggregate_exp4 ──


def _make_question_results(n: int = 5) -> list[Exp4QuestionResult]:
    results = []
    for i in range(n):
        results.append(
            Exp4QuestionResult(
                question_id=f"q{i}",
                category="factual",
                retrieved_ids=["a", "b"],
                relevant_ids={"a"},
                ir_metrics={
                    "precision_at_k": 0.5,
                    "recall_at_k": 1.0,
                    "mrr": 1.0,
                    "ndcg_at_k": 1.0,
                },
                generation_metrics={
                    "citation_accuracy": 0.8 + i * 0.02,
                    "composite_confidence": 0.7,
                },
                latency_breakdown={"generation": 100.0},
            )
        )
    return results


def test_aggregate_exp4():
    results = _make_question_results(5)
    ir, gen = aggregate_exp4(results)
    assert "precision_at_k" in ir
    assert "recall_at_k" in ir
    assert "citation_accuracy" in gen
    assert "composite_confidence" in gen
    assert abs(ir["precision_at_k"].mean - 0.5) < 1e-9


def test_aggregate_exp4_empty():
    ir, gen = aggregate_exp4([])
    assert ir == {}
    assert gen == {}


# ── run_experiment_4 ──


def test_run_experiment_4():
    qrs = _make_question_results(3)
    result = run_experiment_4(
        name="test_exp",
        config=EXPERIMENT_CONFIGS["dense_only"],
        question_results=qrs,
        dataset_hash_value="abc123",
        environment="test",
    )
    assert result.name == "test_exp"
    assert len(result.per_question) == 3
    assert "precision_at_k" in result.ir_overall
    assert result.latency_profiler.environment == "test"


def test_run_experiment_4_to_dict():
    qrs = _make_question_results(3)
    result = run_experiment_4(
        name="test",
        config=EXPERIMENT_CONFIGS["hybrid"],
        question_results=qrs,
        dataset_hash_value="abc",
    )
    d = result.to_dict()
    assert d["name"] == "test"
    assert d["n_questions"] == 3
    assert "ir_overall" in d
    assert "gen_overall" in d
    assert "latency" in d


# ── compare_exp4 ──


def test_compare_exp4():
    qrs_a = _make_question_results(10)
    qrs_b = _make_question_results(10)
    for qr in qrs_b:
        qr.ir_metrics["ndcg_at_k"] = 0.5

    result_a = run_experiment_4(
        "a",
        EXPERIMENT_CONFIGS["hybrid_rerank"],
        qrs_a,
        "h1",
    )
    result_b = run_experiment_4(
        "b",
        EXPERIMENT_CONFIGS["dense_only"],
        qrs_b,
        "h1",
    )
    comp = compare_exp4(result_a, result_b)
    assert comp["experiment_a"] == "a"
    assert comp["experiment_b"] == "b"
    assert "ir" in comp
    assert "gen" in comp
    assert "wilcoxon_p" in comp["ir"]
    assert "significant" in comp["ir"]
