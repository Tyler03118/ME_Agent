from pathlib import Path

from me_agent.evaluation import load_evaluation_cases


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


def test_summary_includes_detailed_expected_and_agent_answers() -> None:
    from me_agent.evaluation import summarize_evaluation
    from me_agent.schemas import AgentResponse, EvaluationCase, EvaluationResult

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
        passed=True,
        source_correct=True,
        route_correct=True,
        key_facts_found=("2 GB", "LPDDR4"),
    )

    summary = summarize_evaluation([result])

    assert summary["case_results"] == [
        {
            "question_id": "2",
            "category": "Single Source",
            "question": "What is RAM?",
            "expected_answer": "The ECU-850 has 2 GB LPDDR4 RAM.",
            "agent_answer": "The ECU-850 has 2 GB LPDDR4 RAM. Sources: ECU-800_Series_Base.md.",
            "passed": True,
            "used_llm": True,
            "fallback_reason": "",
            "latency_seconds": 0.5,
            "sources": ["ECU-800_Series_Base.md"],
            "route_category": "ecu_800_lookup",
            "key_facts_found": ["2 GB", "LPDDR4"],
            "missing_key_facts": [],
            "source_correct": True,
            "route_correct": True,
            "confidence": 0.9,
            "needs_human_review": False,
            "retriever_mode": "hybrid",
        }
    ]
