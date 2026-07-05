from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from me_agent.core.config import AgentConfig
from me_agent.evaluation import evaluate_cases, load_evaluation_cases, summarize_evaluation
from me_agent.workflow.graph import EngineeringAssistant
from me_agent.generation.llm import DeepSeekAnswerGenerator, _build_prompt
from me_agent.workflow.router import ECU_700_SOURCE, ECU_800_BASE_SOURCE
from me_agent.core.schemas import ManualChunk, RetrievalResult

ROOT = Path(__file__).resolve().parents[1]


def _assistant() -> EngineeringAssistant:
    return EngineeringAssistant.from_config(
        AgentConfig(manual_dir=ROOT / "data" / "manuals", embedding_backend="hashing")
    )


def test_assistant_uses_real_langgraph_state_graph_with_conditional_edges() -> None:
    assistant = _assistant()
    mermaid = assistant.compiled_graph.get_graph().draw_mermaid()

    assert isinstance(assistant.compiled_graph, CompiledStateGraph)
    assert "out_of_scope" in mermaid
    assert "broaden_retrieve" in mermaid
    assert "human_review" in mermaid


def test_out_of_domain_question_returns_scope_response() -> None:
    response = _assistant().ask("今天天气如何")

    assert response.route_category == "general"
    assert response.sources == ()
    assert "outside the ECU manual scope" in response.answer
    assert response.fallback_reason == "out_of_scope"


def test_generic_fallback_is_grounded_in_retrieved_context() -> None:
    response = _assistant().ask("How much RAM does the ECU-850 have?")

    assert response.used_llm is False
    assert response.sources
    assert "ECU-800_Series_Base.md" in response.sources
    assert "2 GB" in response.answer or "LPDDR4" in response.answer
    assert "me-driver-ctl --enable-npu" not in response.answer


def test_evaluation_summary_uses_generic_scores() -> None:
    assistant = _assistant()
    cases = load_evaluation_cases(ROOT / "data" / "eval" / "test-questions.csv")

    results = evaluate_cases(assistant, cases)
    summary = summarize_evaluation(results)

    assert summary["total_cases"] == 10
    assert "mean_semantic_similarity" in summary
    assert "mean_token_coverage" in summary
    assert summary["used_llm_cases"] == 0
    assert summary["fallback_cases"] == 10


def test_deepseek_output_is_not_overwritten(monkeypatch) -> None:
    class FakeClient:
        def invoke(self, prompt):
            del prompt
            return type("Response", (), {"content": "  Live model answer with custom wording.  "})()

    generator = DeepSeekAnswerGenerator(AgentConfig())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(generator, "_build_client", lambda: FakeClient())
    result = generator.generate(
        question="Compare ECU-750 and ECU-850 CAN capabilities",
        route=_assistant().router.route("Compare ECU-750 and ECU-850 CAN capabilities"),
        retrieved_context=[
            RetrievalResult(
                ManualChunk(content="Context says something else.", metadata={"source": "manual.md"}),
                0.5,
            )
        ],
    )

    assert result.used_llm is True
    assert result.answer == "Live model answer with custom wording."


def test_comparison_prompt_prioritizes_key_differences() -> None:
    assistant = _assistant()
    route = assistant.router.route("What are the differences between ECU-850 and ECU-850b?")
    prompt = _build_prompt(
        "What are the differences between ECU-850 and ECU-850b?",
        route,
        [
            RetrievalResult(
                ManualChunk(
                    content="ECU-850 has 2 GB RAM. ECU-850b has 4 GB RAM and a 5 TOPS NPU.",
                    metadata={"source": "manual.md"},
                ),
                0.9,
            )
        ],
    )

    assert "changed specifications" in prompt
    assert "avoid exhaustive tables" in prompt


def test_low_confidence_retrieval_triggers_one_broaden_retry() -> None:
    class LowScoreRetriever:
        def __init__(self):
            self.chunks = [ManualChunk(content="ECU-850 has RAM information.", metadata={"source": "x.md"})]
            self.calls = []

        def retrieve(self, query, *, required_sources=None, top_k=None):
            self.calls.append((required_sources, top_k))
            return [RetrievalResult(self.chunks[0], 0.0)]

    retriever = LowScoreRetriever()
    assistant = EngineeringAssistant(
        retriever,
        config=AgentConfig(retrieval_retry_threshold=0.5, embedding_backend="hashing"),
    )

    response = assistant.ask("How much RAM does ECU-850 have?")

    assert response.answer
    assert len(retriever.calls) == 2
    assert retriever.calls[1][0] is None



def test_comparison_route_retrieves_multiple_chunks_per_required_source() -> None:
    class RecordingRetriever:
        def __init__(self):
            self.chunks = [ManualChunk(content="context", metadata={"source": "x.md"})]
            self.calls = []

        def retrieve(self, query, *, required_sources=None, top_k=None):
            self.calls.append((query, required_sources, top_k))
            return [RetrievalResult(self.chunks[0], 0.4)]

    retriever = RecordingRetriever()
    assistant = EngineeringAssistant(
        retriever,
        config=AgentConfig(top_k=4, embedding_backend="hashing"),
    )

    assistant.ask("How does the storage capacity compare across all ECU models?")

    _query, required_sources, top_k = retriever.calls[0]
    assert required_sources is not None
    assert len(required_sources) == 3
    assert top_k == 6

def test_empty_in_domain_context_branches_to_human_review() -> None:
    response = EngineeringAssistant.from_documents([]).ask("How much RAM does ECU-850 have?")

    assert response.needs_human_review is True
    assert response.verifier_status == "unsupported"


def test_assistant_can_switch_retriever_modes() -> None:
    for mode in ("keyword", "vector", "hybrid"):
        assistant = EngineeringAssistant.from_config(
            AgentConfig(
                manual_dir=ROOT / "data" / "manuals",
                retriever_mode=mode,
                embedding_backend="hashing",
            )
        )
        response = assistant.ask("Compare ECU-750 and ECU-850 CAN capabilities")

        assert response.retriever_mode == mode
        assert ECU_700_SOURCE in response.sources
        assert ECU_800_BASE_SOURCE in response.sources
