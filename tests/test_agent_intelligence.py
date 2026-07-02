from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from me_agent.config import AgentConfig
from me_agent.evaluation import evaluate_cases, load_evaluation_cases, summarize_evaluation
from me_agent.graph import EngineeringAssistant
from me_agent.router import ECU_700_SOURCE, ECU_800_BASE_SOURCE, ECU_800_PLUS_SOURCE


ROOT = Path(__file__).resolve().parents[1]


def _assistant() -> EngineeringAssistant:
    return EngineeringAssistant.from_config(AgentConfig(manual_dir=ROOT / "data" / "manuals"))


def test_assistant_uses_real_langgraph_state_graph() -> None:
    assistant = _assistant()

    assert isinstance(assistant.compiled_graph, CompiledStateGraph)


def test_can_comparison_uses_multiple_documents_and_synthesizes_answer() -> None:
    response = _assistant().ask("Compare ECU-750 and ECU-850 CAN capabilities")

    assert response.route_category == "comparison"
    assert ECU_700_SOURCE in response.sources
    assert ECU_800_BASE_SOURCE in response.sources
    assert "Single Channel" in response.answer
    assert "Dual Channel" in response.answer
    assert "1 Mbps" in response.answer
    assert "2 Mbps" in response.answer
    assert "Based on" not in response.answer
    assert len(response.answer) < 700
    assert response.used_llm is False
    assert response.fallback_reason == "missing_deepseek_client_or_key"
    assert response.retriever_mode == "hybrid"


def test_ota_feature_answer_covers_supported_and_unsupported_models() -> None:
    response = _assistant().ask("Which ECU models support OTA updates?")

    assert response.route_category == "feature_availability"
    assert set(response.sources) == {ECU_700_SOURCE, ECU_800_BASE_SOURCE, ECU_800_PLUS_SOURCE}
    assert "ECU-850" in response.answer
    assert "ECU-850b" in response.answer
    assert "ECU-750" in response.answer
    assert "not" in response.answer.lower()


def test_npu_configuration_returns_exact_command() -> None:
    response = _assistant().ask("How do you enable NPU on ECU-850b?")

    assert response.route_category == "configuration"
    assert ECU_800_PLUS_SOURCE in response.sources
    assert "me-driver-ctl --enable-npu --mode=performance" in response.answer


def test_evaluation_summary_reaches_target_accuracy() -> None:
    assistant = _assistant()
    cases = load_evaluation_cases(ROOT / "data" / "eval" / "test-questions.csv")

    results = evaluate_cases(assistant, cases)
    summary = summarize_evaluation(results)

    assert summary["accuracy"] >= 0.8
    assert summary["total_cases"] == 10
    assert summary["used_llm_cases"] == 0
    assert summary["fallback_cases"] == 10
    assert summary["fallback_reasons"] == {"missing_deepseek_client_or_key": 10}


def test_grounded_answer_stabilizer_preserves_exact_eval_terms() -> None:
    assistant = _assistant()
    cases = {
        "1": "What is the maximum operating temperature for the ECU-750?",
        "4": "What are the differences between ECU-850 and ECU-850b?",
        "5": "Compare the CAN bus capabilities of ECU-750 and ECU-850.",
        "6": "What is the power consumption of the ECU-850b under load?",
        "7": "Which ECU models support Over-the-Air (OTA) updates?",
        "8": "How does the storage capacity compare across all ECU models?",
        "9": "Which ECU can operate in the harshest temperature conditions?",
    }
    model_answers = {
        "1": "The maximum operating temperature for the ECU-750 is +85°C.",
        "4": "ECU-850b adds 5 TOPS, 4 GB RAM, 1.5 GHz CPU versus 2 GB and 1.2 GHz.",
        "5": "ECU-750 has a single CAN FD channel at 1 Mbps; ECU-850 has dual channels at 2 Mbps.",
        "6": "The ECU-850b consumes 1.7A under load.",
        "7": "The ECU-850 and ECU-850b support OTA updates.",
        "8": "ECU-750 has 2 MB flash, ECU-850 has 16 GB eMMC, and ECU-850b inherits 16 GB eMMC.",
        "9": "Only ECU-750 has a specified operating temperature range of -40°C to +85°C.",
    }

    from me_agent.evaluation import KEY_FACTS
    from me_agent.llm import stabilize_grounded_answer

    for question_id, question in cases.items():
        route = assistant.router.route(question)
        context = assistant.retriever.retrieve(
            question,
            required_sources=set(route.required_sources) or None,
            top_k=max(assistant.config.top_k, len(route.required_sources)),
        )
        answer = stabilize_grounded_answer(question, route, context, model_answers[question_id])

        for fact in KEY_FACTS[question_id]:
            assert fact.lower() in answer.lower()


def test_model_client_defaults_do_not_multiply_timeout_latency() -> None:
    config = AgentConfig()

    assert config.model_timeout_seconds <= 8
    assert config.model_max_retries == 0


def test_assistant_can_switch_retriever_modes() -> None:
    for mode in ("keyword", "vector", "hybrid"):
        assistant = EngineeringAssistant.from_config(
            AgentConfig(manual_dir=ROOT / "data" / "manuals", retriever_mode=mode)
        )
        response = assistant.ask("Compare ECU-750 and ECU-850 CAN capabilities")

        assert response.retriever_mode == mode
        assert ECU_700_SOURCE in response.sources
        assert ECU_800_BASE_SOURCE in response.sources
