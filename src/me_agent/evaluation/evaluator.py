"""Honest evaluation utilities for golden ECU questions."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from me_agent.retrieval.embeddings import EmbeddingModel, tokenize
from me_agent.workflow.graph import EngineeringAssistant
from me_agent.core.schemas import EvaluationCase, EvaluationResult

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in",
    "is", "it", "of", "on", "or", "the", "to", "with", "while", "than", "this", "that",
}


def load_evaluation_cases(csv_path: str | Path) -> list[EvaluationCase]:
    """Load golden evaluation cases from a CSV file."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            EvaluationCase(
                question_id=row["Question_ID"],
                category=row["Category"],
                question=row["Question"],
                expected_answer=row["Expected_Answer"],
                evaluation_criteria=row["Evaluation_Criteria"],
            )
            for row in reader
        ]


def evaluate_cases(
    assistant: EngineeringAssistant,
    cases: list[EvaluationCase],
    *,
    pass_threshold: float | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> list[EvaluationResult]:
    """Run the assistant and score answers against Expected_Answer generically."""

    threshold = pass_threshold or assistant.config.eval_pass_threshold
    scorer = embedding_model or EmbeddingModel(
        model_name=assistant.config.embedding_model_name,
        backend=assistant.config.embedding_backend,
    )
    results: list[EvaluationResult] = []
    for case in cases:
        start = time.perf_counter()
        response = assistant.ask(case.question)
        latency = time.perf_counter() - start
        similarity = semantic_similarity(case.expected_answer, response.answer, scorer)
        coverage = token_coverage(case.expected_answer, response.answer)
        combined = 0.65 * similarity + 0.35 * coverage
        results.append(
            EvaluationResult(
                case=case,
                response=response,
                latency_seconds=latency,
                semantic_similarity=similarity,
                token_coverage=coverage,
                passed=combined >= threshold,
                source_diagnostic=round(len(response.sources), 4),
                route_diagnostic=response.route_category,
            )
        )
    return results


def semantic_similarity(expected: str, actual: str, embedding_model: EmbeddingModel) -> float:
    """Compute cosine similarity between expected and actual answer embeddings."""

    vectors = embedding_model.encode([expected, actual])
    if vectors.shape[0] != 2:
        return 0.0
    return round(float(np.dot(vectors[0], vectors[1])), 4)


def token_coverage(expected: str, actual: str) -> float:
    """Measure expected-answer content token coverage in the actual answer."""

    expected_tokens = _content_tokens(expected)
    if not expected_tokens:
        return 0.0
    actual_tokens = set(_content_tokens(actual))
    return round(len(set(expected_tokens) & actual_tokens) / len(set(expected_tokens)), 4)


def summarize_evaluation(results: list[EvaluationResult]) -> dict[str, Any]:
    """Summarize scored evaluation results."""

    total = len(results)
    passed = sum(1 for result in results if result.passed)
    avg_latency = sum(result.latency_seconds for result in results) / total if total else 0.0
    used_llm = sum(1 for result in results if result.response.used_llm)
    fallback_reasons: dict[str, int] = {}
    for result in results:
        if not result.response.used_llm:
            reason = result.response.fallback_reason or "unknown"
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
    case_results = [_case_result_detail(result) for result in results]
    failed_cases = [detail for detail in case_results if not detail["passed"]]
    mean_similarity = _mean(result.semantic_similarity for result in results)
    mean_coverage = _mean(result.token_coverage for result in results)
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases_count": len(failed_cases),
        "accuracy": round(passed / total, 4) if total else 0.0,
        "used_llm_cases": used_llm,
        "used_llm_rate": round(used_llm / total, 4) if total else 0.0,
        "fallback_cases": total - used_llm,
        "fallback_reasons": fallback_reasons,
        "avg_latency_seconds": round(avg_latency, 4),
        "max_latency_seconds": round(max((r.latency_seconds for r in results), default=0.0), 4),
        "mean_semantic_similarity": mean_similarity,
        "mean_token_coverage": mean_coverage,
        "case_results": case_results,
        "failed_cases": failed_cases,
    }


def write_evaluation_results(
    results: list[EvaluationResult],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write detailed evaluation JSON and return the summary."""

    path = Path(output_path)
    summary = summarize_evaluation(results)
    payload = {"summary": summary, "results": [_case_result_detail(result) for result in results]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary


def log_evaluation_to_mlflow(summary: dict[str, Any], artifact_path: str | Path) -> None:
    """Log evaluation metrics and result JSON to an active MLflow run."""

    import mlflow  # pylint: disable=import-outside-toplevel

    metric_names = (
        "accuracy",
        "avg_latency_seconds",
        "max_latency_seconds",
        "used_llm_rate",
        "fallback_cases",
        "mean_semantic_similarity",
        "mean_token_coverage",
    )
    for name in metric_names:
        mlflow.log_metric(name, float(summary.get(name, 0.0)))
    mlflow.log_artifact(str(artifact_path))


def _case_result_detail(result: EvaluationResult) -> dict[str, Any]:
    return {
        "question_id": result.case.question_id,
        "category": result.case.category,
        "question": result.case.question,
        "expected_answer": result.case.expected_answer,
        "actual_answer": result.response.answer,
        "agent_answer": result.response.answer,
        "semantic_similarity": result.semantic_similarity,
        "token_coverage": result.token_coverage,
        "passed": result.passed,
        "retriever_mode": result.response.retriever_mode,
        "used_llm": result.response.used_llm,
        "fallback_reason": result.response.fallback_reason,
        "latency_seconds": round(result.latency_seconds, 4),
        "sources": list(result.response.sources),
        "route_category": result.response.route_category,
        "source_diagnostic": result.source_diagnostic,
        "route_diagnostic": result.route_diagnostic,
        "confidence": result.response.confidence,
        "verifier_status": result.response.verifier_status,
        "needs_human_review": result.response.needs_human_review,
    }


def _content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def _mean(values) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return round(sum(collected) / len(collected), 4)
