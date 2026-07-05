"""Honest evaluation utilities for golden ECU questions."""

from __future__ import annotations

import csv
import json
import re
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


# Case loading and evaluation runner ----------------------------------------


def load_evaluation_cases(csv_path: str | Path) -> list[EvaluationCase]:
    """Load evaluation cases from a CSV file.

    The base challenge columns are required. Stress-set columns are optional and
    default to empty tuples, so the same loader supports both datasets.
    """

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
                required_facts=_split_criteria(row.get("Required_Facts", "")),
                forbidden_facts=_split_criteria(row.get("Forbidden_Facts", "")),
                expected_sources=_split_criteria(row.get("Expected_Sources", "")),
                expected_route=row.get("Expected_Route", "").strip(),
            )
            for row in reader
        ]


def evaluate_cases(  # pylint: disable=too-many-locals
    assistant: EngineeringAssistant,
    cases: list[EvaluationCase],
    *,
    pass_threshold: float | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> list[EvaluationResult]:
    """Run the assistant and score answers against Expected_Answer and stress criteria."""

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
        # Base scoring applies to every row. Enhanced stress rows add explicit
        # factual, source, and route checks below without changing the CSV loader.
        similarity = semantic_similarity(case.expected_answer, response.answer, scorer)
        coverage = token_coverage(case.expected_answer, response.answer)
        required_facts = case.required_facts or infer_required_facts(case.expected_answer)
        fact_score = fact_recall(required_facts, response.answer)
        forbidden_violations = forbidden_fact_violations(case.forbidden_facts, response.answer)
        source_score = source_match(case.expected_sources, response.sources)
        route_score = route_match(case.expected_route, response.route_category)
        combined = combined_evaluation_score(
            semantic_similarity_score=similarity,
            token_coverage_score=coverage,
            required_fact_recall_score=fact_score,
            source_match_score=source_score,
            route_match_score=route_score,
            forbidden_violations=forbidden_violations,
            enhanced_criteria=_has_enhanced_criteria(case),
        )
        results.append(
            EvaluationResult(
                case=case,
                response=response,
                latency_seconds=latency,
                semantic_similarity=similarity,
                token_coverage=coverage,
                required_fact_recall=fact_score,
                forbidden_fact_violations=forbidden_violations,
                source_match=source_score,
                route_match=route_score,
                combined_score=combined,
                passed=combined >= threshold,
                source_diagnostic=round(len(response.sources), 4),
                route_diagnostic=response.route_category,
            )
        )
    return results


# Metric functions ----------------------------------------------------------


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


def fact_recall(required_facts: tuple[str, ...], actual: str) -> float:
    """Return the share of required fact phrases supported by the answer text."""

    if not required_facts:
        return 1.0
    matched = sum(1 for fact in required_facts if _fact_present(fact, actual))
    return round(matched / len(required_facts), 4)


def infer_required_facts(expected: str) -> tuple[str, ...]:
    """Extract technical facts from a golden answer for fact-first scoring.

    The base challenge CSV does not include explicit Required_Facts columns, so
    this parser extracts stable engineering facts from the expected answer:
    model identifiers, numeric specifications, ECU feature acronyms, processor
    families, and exact CLI commands. It stays generic and avoids question-id
    rules while reducing false negatives when an answer is factually correct but
    more complete than the reference wording.
    """

    patterns = (
        r"\bECU-\d+[a-z]?\b",
        r"\bme-driver-ctl\s+--[a-z0-9=-]+(?:\s+--[a-z0-9=-]+)*\b",
        r"[+-]?\d+(?:\.\d+)?\s*(?:°C|GHz|MHz|Mbps|TOPS|GB|MB|KB|mA|A)\b",
        r"\b(?:NPU|OTA|CAN FD|CAN|LPDDR\d|eMMC|SRAM)\b",
        r"\bCortex-A\d+\b",
    )
    facts: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, expected, flags=re.IGNORECASE):
            fact = " ".join(match.group(0).split())
            key = fact.lower()
            if key not in seen:
                seen.add(key)
                facts.append(fact)
    return tuple(facts)


def forbidden_fact_violations(forbidden_facts: tuple[str, ...], actual: str) -> int:
    """Count explicit forbidden fact phrases that appear in the answer text."""

    return sum(1 for fact in forbidden_facts if _forbidden_fact_present(fact, actual))


def source_match(expected_sources: tuple[str, ...], actual_sources: tuple[str, ...]) -> float:
    """Return recall of expected sources in the retrieved source list."""

    if not expected_sources:
        return 1.0
    actual = set(actual_sources)
    return round(len(set(expected_sources) & actual) / len(set(expected_sources)), 4)


def route_match(expected_route: str, actual_route: str) -> float:
    """Return 1.0 when route diagnostics match or no expected route is configured."""

    if not expected_route:
        return 1.0
    return 1.0 if expected_route == actual_route else 0.0


def combined_evaluation_score(  # pylint: disable=too-many-arguments
    *,
    semantic_similarity_score: float,
    token_coverage_score: float,
    required_fact_recall_score: float,
    source_match_score: float,
    route_match_score: float,
    forbidden_violations: int,
    enhanced_criteria: bool,
) -> float:
    """Combine metrics for base and enhanced evaluation rows.

    Basic rows combine semantic similarity, token coverage, and fact recall.
    Enhanced rows also include explicit checks for forbidden claims, expected
    sources, and expected route. Forbidden claims subtract a capped penalty
    because one explicit contradiction can be more serious than a missing
    optional fact.
    """

    if not enhanced_criteria:
        return round(
            0.45 * semantic_similarity_score
            + 0.20 * token_coverage_score
            + 0.35 * required_fact_recall_score,
            4,
        )
    penalty = min(0.35, 0.20 * forbidden_violations)
    score = (
        0.35 * semantic_similarity_score
        + 0.20 * token_coverage_score
        + 0.25 * required_fact_recall_score
        + 0.10 * source_match_score
        + 0.10 * route_match_score
        - penalty
    )
    return round(max(0.0, score), 4)


# Summary, JSON artifact, and MLflow logging --------------------------------


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
        "mean_required_fact_recall": _mean(result.required_fact_recall for result in results),
        "mean_source_match": _mean(result.source_match for result in results),
        "mean_route_match": _mean(result.route_match for result in results),
        "forbidden_fact_violations": sum(
            result.forbidden_fact_violations for result in results
        ),
        "case_results": case_results,
        "failed_cases": failed_cases,
    }


def write_evaluation_results(
    results: list[EvaluationResult],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write detailed evaluation JSON and return the summary."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "mean_required_fact_recall",
        "mean_source_match",
        "mean_route_match",
        "forbidden_fact_violations",
    )
    for name in metric_names:
        mlflow.log_metric(name, float(summary.get(name, 0.0)))
    mlflow.log_artifact(str(artifact_path))


# Internal matching and formatting helpers ----------------------------------


def _case_result_detail(result: EvaluationResult) -> dict[str, Any]:
    """Flatten one evaluation result into JSON-serializable diagnostics."""

    return {
        "question_id": result.case.question_id,
        "category": result.case.category,
        "question": result.case.question,
        "expected_answer": result.case.expected_answer,
        "actual_answer": result.response.answer,
        "agent_answer": result.response.answer,
        "semantic_similarity": result.semantic_similarity,
        "token_coverage": result.token_coverage,
        "required_fact_recall": result.required_fact_recall,
        "forbidden_fact_violations": result.forbidden_fact_violations,
        "source_match": result.source_match,
        "route_match": result.route_match,
        "combined_score": result.combined_score,
        "passed": result.passed,
        "retriever_mode": result.response.retriever_mode,
        "used_llm": result.response.used_llm,
        "fallback_reason": result.response.fallback_reason,
        "latency_seconds": round(result.latency_seconds, 4),
        "sources": list(result.response.sources),
        "expected_sources": list(result.case.expected_sources),
        "route_category": result.response.route_category,
        "expected_route": result.case.expected_route,
        "source_diagnostic": result.source_diagnostic,
        "route_diagnostic": result.route_diagnostic,
        "confidence": result.response.confidence,
        "verifier_status": result.response.verifier_status,
        "needs_human_review": result.response.needs_human_review,
    }


def _split_criteria(value: str | None) -> tuple[str, ...]:
    """Split pipe-separated optional CSV criteria into normalized strings."""

    if not value:
        return ()
    return tuple(part.strip() for part in value.split("|") if part.strip())


def _has_enhanced_criteria(case: EvaluationCase) -> bool:
    """Return whether a row uses stress-set fields beyond Expected_Answer."""

    return bool(
        case.required_facts
        or case.forbidden_facts
        or case.expected_sources
        or case.expected_route
    )


def _fact_present(fact: str, text: str) -> bool:
    """Check whether a required fact phrase is supported by answer text."""

    fact_normalized = _normalize_for_fact_match(fact)
    text_normalized = _normalize_for_fact_match(text)
    if not fact_normalized:
        return False
    if fact_normalized in text_normalized:
        return True
    fact_tokens = set(_content_tokens(fact))
    text_tokens = set(_content_tokens(text))
    return bool(fact_tokens) and fact_tokens <= text_tokens


def _forbidden_fact_present(fact: str, text: str) -> bool:
    """Check whether a forbidden phrase appears as an asserted answer fact."""

    fact_normalized = _normalize_for_fact_match(fact)
    text_normalized = _normalize_for_fact_match(text)
    if not fact_normalized:
        return False
    start = text_normalized.find(fact_normalized)
    if start < 0:
        return False
    context_start = max(0, start - 80)
    context_end = min(len(text_normalized), start + len(fact_normalized) + 80)
    local_context = text_normalized[context_start:context_end]
    # A forbidden phrase is not counted as a violation when the nearby text is
    # rejecting or correcting it, for example "ECU-750 does not support OTA".
    rejection_markers = (
        "cannot",
        "can not",
        "do not",
        "does not",
        "not ",
        "contradict",
        "false",
        "instead",
        "rather than",
    )
    return not any(marker in local_context for marker in rejection_markers)


def _normalize_for_fact_match(text: str) -> str:
    """Normalize punctuation and casing for phrase-level fact matching."""

    return " ".join(re.findall(r"[a-zA-Z0-9+.-]+", text.lower()))


def _content_tokens(text: str) -> list[str]:
    """Return non-stopword tokens used by token coverage and fact matching."""

    return [token for token in tokenize(text) if token not in STOPWORDS]


def _mean(values) -> float:
    """Return a rounded arithmetic mean for an iterable of numeric values."""

    collected = list(values)
    if not collected:
        return 0.0
    return round(sum(collected) / len(collected), 4)
