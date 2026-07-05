from me_agent.workflow.router import DeterministicRouter


def test_router_classifies_ecu_750_lookup() -> None:
    decision = DeterministicRouter().route("What is the temperature for ECU-750?")

    assert decision.category == "ecu_700_lookup"
    assert "ECU-700_Series_Manual.md" in decision.required_sources


def test_router_classifies_cross_document_comparison() -> None:
    decision = DeterministicRouter().route("Compare the CAN bus capabilities of ECU-750 and ECU-850.")

    assert decision.category == "comparison"
    assert "ECU-700_Series_Manual.md" in decision.required_sources
    assert "ECU-800_Series_Base.md" in decision.required_sources


def test_router_classifies_npu_configuration() -> None:
    decision = DeterministicRouter().route("How do you enable the NPU on ECU-850b?")

    assert decision.category == "configuration"
    assert "ECU-800_Series_Plus.md" in decision.required_sources


def test_router_sends_non_ecu_questions_to_general() -> None:
    decision = DeterministicRouter().route("今天天气如何")

    assert decision.category == "general"
    assert decision.required_sources == ()


def test_router_classifies_terse_difference_phrasing_as_comparison() -> None:
    decision = DeterministicRouter().route("List only the changed specs between ECU-850 and ECU-850b.")

    assert decision.category == "comparison"
    assert "ECU-800_Series_Base.md" in decision.required_sources
    assert "ECU-800_Series_Plus.md" in decision.required_sources


def test_router_classifies_vs_phrasing_as_comparison() -> None:
    decision = DeterministicRouter().route("ECU-750 vs ECU-850 CAN speed")

    assert decision.category == "comparison"
    assert "ECU-700_Series_Manual.md" in decision.required_sources
    assert "ECU-800_Series_Base.md" in decision.required_sources
