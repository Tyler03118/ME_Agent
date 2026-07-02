"""Evaluation utilities for golden ECU questions."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from me_agent.graph import EngineeringAssistant
from me_agent.router import ECU_700_SOURCE, ECU_800_BASE_SOURCE, ECU_800_PLUS_SOURCE
from me_agent.schemas import EvaluationCase, EvaluationResult


EXPECTED_SOURCES = {
    "1": {ECU_700_SOURCE},
    "2": {ECU_800_BASE_SOURCE},
    "3": {ECU_800_PLUS_SOURCE},
    "4": {ECU_800_BASE_SOURCE, ECU_800_PLUS_SOURCE},
    "5": {ECU_700_SOURCE, ECU_800_BASE_SOURCE},
    "6": {ECU_800_PLUS_SOURCE},
    "7": {ECU_700_SOURCE, ECU_800_BASE_SOURCE, ECU_800_PLUS_SOURCE},
    "8": {ECU_700_SOURCE, ECU_800_BASE_SOURCE, ECU_800_PLUS_SOURCE},
    "9": {ECU_700_SOURCE, ECU_800_BASE_SOURCE, ECU_800_PLUS_SOURCE},
    "10": {ECU_800_PLUS_SOURCE},
}

EXPECTED_ROUTES = {
    "1": "ecu_700_lookup",
    "2": "ecu_800_lookup",
    "3": "ecu_850b_lookup",
    "4": "comparison",
    "5": "comparison",
    "6": "ecu_850b_lookup",
    "7": "feature_availability",
    "8": "comparison",
    "9": "comparison",
    "10": "configuration",
}

KEY_FACTS = {
    "1": ("+85", "-40"),
    "2": ("2 GB", "LPDDR4"),
    "3": ("NPU", "5 TOPS"),
    "4": ("5 TOPS", "4 GB", "2 GB", "1.5 GHz", "1.2 GHz"),
    "5": ("Single Channel", "1 Mbps", "Dual Channel", "2 Mbps"),
    "6": ("1.7A", "550mA"),
    "7": ("ECU-850", "ECU-850b", "ECU-750", "not"),
    "8": ("2 MB", "16 GB", "32 GB"),
    "9": ("ECU-850", "ECU-850b", "+105", "+85"),
    "10": ("me-driver-ctl --enable-npu --mode=performance",),
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
) -> list[EvaluationResult]:
    """Run the assistant over evaluation cases and capture scored results."""

    results: list[EvaluationResult] = []
    for case in cases:
        start = time.perf_counter()
        response = assistant.ask(case.question)
        latency = time.perf_counter() - start
        key_facts_found = _key_facts_found(case.question_id, response.answer)
        source_correct = EXPECTED_SOURCES.get(case.question_id, set()).issubset(response.sources)
        route_correct = response.route_category == EXPECTED_ROUTES.get(case.question_id)
        passed = bool(key_facts_found) and len(key_facts_found) == len(
            KEY_FACTS.get(case.question_id, ())
        ) and source_correct and route_correct
        results.append(
            EvaluationResult(
                case=case,
                response=response,
                latency_seconds=latency,
                passed=passed,
                source_correct=source_correct,
                route_correct=route_correct,
                key_facts_found=key_facts_found,
            )
        )
    return results


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
    payload = {
        "summary": summary,
        "results": [_case_result_detail(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary


def _case_result_detail(result: EvaluationResult) -> dict[str, Any]:
    key_facts = KEY_FACTS.get(result.case.question_id, ())
    found = set(result.key_facts_found)
    return {
        "question_id": result.case.question_id,
        "category": result.case.category,
        "question": result.case.question,
        "expected_answer": result.case.expected_answer,
        "agent_answer": result.response.answer,
        "passed": result.passed,
        "retriever_mode": result.response.retriever_mode,
        "used_llm": result.response.used_llm,
        "fallback_reason": result.response.fallback_reason,
        "latency_seconds": round(result.latency_seconds, 4),
        "sources": list(result.response.sources),
        "route_category": result.response.route_category,
        "key_facts_found": list(result.key_facts_found),
        "missing_key_facts": [fact for fact in key_facts if fact not in found],
        "source_correct": result.source_correct,
        "route_correct": result.route_correct,
        "confidence": result.response.confidence,
        "needs_human_review": result.response.needs_human_review,
    }


def _key_facts_found(question_id: str, answer: str) -> tuple[str, ...]:
    answer_lower = answer.lower()
    found = []
    for fact in KEY_FACTS.get(question_id, ()):
        if fact.lower() in answer_lower:
            found.append(fact)
    return tuple(found)
