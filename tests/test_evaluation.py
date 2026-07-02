from pathlib import Path

from me_agent.evaluation import load_evaluation_cases, summarize_evaluation, token_coverage


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
