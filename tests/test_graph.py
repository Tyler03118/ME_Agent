from me_agent.workflow.graph import EngineeringAssistant


def test_graph_returns_structured_response_for_empty_corpus() -> None:
    response = EngineeringAssistant.from_documents([]).ask("How much RAM does ECU-850 have?")

    assert response.question == "How much RAM does ECU-850 have?"
    assert response.answer
    assert response.verifier_status in {
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
    }
    assert 0.0 <= response.confidence <= 1.0
    assert response.needs_human_review is True
