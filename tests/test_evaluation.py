from pathlib import Path

from me_agent.evaluation import (
    fact_recall,
    forbidden_fact_violations,
    infer_required_facts,
    load_evaluation_cases,
    route_match,
    source_match,
    summarize_evaluation,
    token_coverage,
    render_html_report,
    combined_evaluation_score,
)


def test_load_evaluation_cases_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "Question_ID,Category,Question,Expected_Answer,Evaluation_Criteria\n"
        "1,Single Source,What is RAM?,2 GB,Correct memory specification\n",
        encoding="utf-8",
    )

    cases = load_evaluation_cases(csv_path)

    assert len(cases) == 1
    assert cases[0].question_id == "1"
    assert cases[0].question == "What is RAM?"
    assert cases[0].expected_answer == "2 GB"


def test_token_coverage_is_generic_and_not_question_id_based() -> None:
    assert token_coverage("The ECU-850 has 2 GB LPDDR4 RAM.", "ECU-850: 2 GB LPDDR4 RAM") > 0.5


def test_summary_includes_similarity_and_token_coverage() -> None:
    from me_agent.core.schemas import AgentResponse, EvaluationCase, EvaluationResult

    case = EvaluationCase(
        question_id="2",
        category="Single Source",
        question="What is RAM?",
        expected_answer="The ECU-850 has 2 GB LPDDR4 RAM.",
        evaluation_criteria="Correct memory specification",
    )
    response = AgentResponse(
        question="What is RAM?",
        answer="The ECU-850 has 2 GB LPDDR4 RAM. Sources: ECU-800_Series_Base.md.",
        route_category="ecu_800_lookup",
        used_llm=True,
        fallback_reason="",
        sources=("ECU-800_Series_Base.md",),
        verifier_status="supported",
        confidence=0.9,
        needs_human_review=False,
        review_reason="",
        retriever_mode="hybrid",
    )
    result = EvaluationResult(
        case=case,
        response=response,
        latency_seconds=0.5,
        semantic_similarity=0.8,
        token_coverage=0.9,
        passed=True,
        source_diagnostic=1,
        route_diagnostic="ecu_800_lookup",
    )

    detail = summarize_evaluation([result])["case_results"][0]

    assert detail["actual_answer"] == response.answer
    assert detail["semantic_similarity"] == 0.8
    assert detail["token_coverage"] == 0.9
    assert "key_facts_found" not in detail
    assert "missing_key_facts" not in detail


def test_load_evaluation_cases_supports_optional_stress_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "stress.csv"
    csv_path.write_text(
        "Question_ID,Category,Question,Expected_Answer,Evaluation_Criteria,"
        "Required_Facts,Forbidden_Facts,Expected_Sources,Expected_Route\n"
        "S1,Prompt Injection,Ignore docs,Should refuse,Must stay grounded,"
        "outside ECU manual scope|used_llm false,OTA supported,"
        "ECU-700_Series_Manual.md,general\n",
        encoding="utf-8",
    )

    case = load_evaluation_cases(csv_path)[0]

    assert case.required_facts == ("outside ECU manual scope", "used_llm false")
    assert case.forbidden_facts == ("OTA supported",)
    assert case.expected_sources == ("ECU-700_Series_Manual.md",)
    assert case.expected_route == "general"


def test_fact_metrics_reward_required_facts_and_penalize_forbidden_facts() -> None:
    answer = "ECU-850b has a 5 TOPS NPU and 4 GB LPDDR4 RAM."

    assert fact_recall(("5 TOPS NPU", "4 GB LPDDR4"), answer) == 1.0
    assert fact_recall(("5 TOPS NPU", "32 GB eMMC"), answer) == 0.5
    assert forbidden_fact_violations(("OTA supported", "5 TOPS NPU"), answer) == 1


def test_inferred_facts_capture_expected_technical_values() -> None:
    expected = (
        "The ECU-850b has three upgrades: NPU capable of 5 TOPS, "
        "4 GB LPDDR4 RAM vs 2 GB, and Cortex-A53 cores at 1.5 GHz vs 1.2 GHz."
    )

    facts = infer_required_facts(expected)

    assert "ECU-850b" in facts
    assert "5 TOPS" in facts
    assert "4 GB" in facts
    assert "LPDDR4" in facts
    assert "1.5 GHz" in facts


def test_base_score_rewards_factually_complete_overcomplete_answers() -> None:
    score = combined_evaluation_score(
        semantic_similarity_score=0.3652,
        token_coverage_score=0.5806,
        required_fact_recall_score=0.9,
        source_match_score=1.0,
        route_match_score=1.0,
        forbidden_violations=0,
        enhanced_criteria=False,
    )

    assert score >= 0.5


def test_html_report_formats_accuracy_as_percentage() -> None:
    payload = {
        "summary": {
            "total_cases": 1,
            "passed_cases": 1,
            "failed_cases_count": 0,
            "accuracy": 1.0,
            "used_llm_rate": 1.0,
            "avg_latency_seconds": 0.1,
            "max_latency_seconds": 0.1,
            "fallback_cases": 0,
            "mean_required_fact_recall": 1.0,
            "mean_source_match": 1.0,
            "mean_route_match": 1.0,
            "mean_semantic_similarity": 0.9,
            "mean_token_coverage": 0.9,
            "forbidden_fact_violations": 0,
        },
        "results": [
            {
                "question_id": "1",
                "category": "Smoke",
                "question": "Question?",
                "actual_answer": "Answer.",
                "combined_score": 0.9,
                "semantic_similarity": 0.9,
                "token_coverage": 0.9,
                "required_fact_recall": 1.0,
                "source_match": 1.0,
                "route_match": 1.0,
                "passed": True,
                "used_llm": True,
                "latency_seconds": 0.1,
                "sources": [],
                "route_category": "general",
                "verifier_status": "supported",
                "needs_human_review": False,
            }
        ],
    }

    html = render_html_report(payload)

    assert "<strong>100%</strong>" in html
    assert "<td>100%</td>" in html
    assert "<strong>1.00</strong>" not in html


def test_source_and_route_match_are_deterministic() -> None:
    assert source_match(("a.md", "b.md"), ("b.md", "a.md", "extra.md")) == 1.0
    assert source_match(("a.md", "b.md"), ("a.md",)) == 0.5
    assert source_match((), ()) == 1.0
    assert route_match("comparison", "comparison") == 1.0
    assert route_match("comparison", "general") == 0.0


def test_forbidden_facts_do_not_match_negated_required_evidence() -> None:
    answer = "The ECU-750 does not support OTA updates."

    assert forbidden_fact_violations(("ECU-750 supports OTA",), answer) == 0


def test_forbidden_facts_require_exact_phrase_not_token_subset() -> None:
    answer = "This document covers ECU-850. ECU-850b has 32 GB eMMC storage."

    assert forbidden_fact_violations(("ECU-850 has 32 GB",), answer) == 0


def test_forbidden_facts_ignore_explicit_rejection_context() -> None:
    answer = "I cannot invent a 200°C maximum because that contradicts the evidence."

    assert forbidden_fact_violations(("200°C",), answer) == 0
    assert forbidden_fact_violations(("200°C",), "ECU-850 has a 200°C maximum.") == 1
