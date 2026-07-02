"""LangGraph workflow for a simple conditional ECU RAG agent."""

from __future__ import annotations

import argparse
import json
from typing import NotRequired, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from me_agent.workflow.confidence import compute_confidence
from me_agent.ingestion.chunking import MarkdownChunker
from me_agent.core.config import AgentConfig
from me_agent.ingestion.data_loader import MarkdownManualLoader
from me_agent.workflow.hitl import human_review_decision
from me_agent.generation.llm import DeepSeekAnswerGenerator, GenerationResult
from me_agent.retrieval.retriever import Retriever, build_retriever
from me_agent.workflow.router import DeterministicRouter
from me_agent.core.schemas import (
    AgentResponse,
    ManualDocument,
    RetrievalResult,
    RouteDecision,
    VerificationResult,
)
from me_agent.generation.verifier import verify_answer


class AgentState(TypedDict):
    """State passed between LangGraph nodes."""

    question: str
    route: NotRequired[RouteDecision]
    retrieved_context: NotRequired[list[RetrievalResult]]
    generation: NotRequired[GenerationResult]
    verification: NotRequired[VerificationResult]
    confidence: NotRequired[float]
    retry_count: NotRequired[int]
    response: NotRequired[AgentResponse]


class EngineeringAssistant:
    """Grounded ECU question-answering workflow with minimal agent-like branching."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        router: DeterministicRouter | None = None,
        config: AgentConfig | None = None,
        generator: DeepSeekAnswerGenerator | None = None,
    ) -> None:
        self.retriever = retriever
        self.router = router or DeterministicRouter()
        self.config = config or AgentConfig()
        self.generator = generator or DeepSeekAnswerGenerator(self.config)
        self.compiled_graph = self._build_graph()

    @classmethod
    def from_documents(
        cls,
        documents: list[ManualDocument],
        *,
        config: AgentConfig | None = None,
    ) -> "EngineeringAssistant":
        """Create an assistant from already-loaded documents."""

        resolved_config = config or AgentConfig()
        chunks = MarkdownChunker(
            chunk_size=resolved_config.chunk_size,
            chunk_overlap=resolved_config.chunk_overlap,
        ).split(documents)
        retriever = build_retriever(
            resolved_config.retriever_mode,
            chunks,
            top_k=resolved_config.top_k,
            keyword_weight=resolved_config.keyword_weight,
            vector_weight=resolved_config.vector_weight,
            embedding_backend=resolved_config.embedding_backend,
            embedding_model_name=resolved_config.embedding_model_name,
        )
        return cls(retriever, config=resolved_config)

    @classmethod
    def from_config(cls, config: AgentConfig | None = None) -> "EngineeringAssistant":
        """Create an assistant from the configured manual directory."""

        resolved_config = config or AgentConfig.from_env()
        documents = MarkdownManualLoader(resolved_config.manual_dir).load()
        return cls.from_documents(documents, config=resolved_config)

    def ask(self, question: str) -> AgentResponse:
        """Execute the compiled LangGraph workflow for one question."""

        result = self.compiled_graph.invoke({"question": question, "retry_count": 0})
        return result["response"]

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("validate_input", self._validate_input)
        graph.add_node("route_query", self._route_query)
        graph.add_node("out_of_scope", self._out_of_scope)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("broaden_retrieve", self._broaden_retrieve)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("verify_answer", self._verify_answer)
        graph.add_node("compute_confidence", self._compute_confidence)
        graph.add_node("finalize_response", self._finalize_response)
        graph.add_node("human_review", self._human_review)

        graph.set_entry_point("validate_input")
        graph.add_edge("validate_input", "route_query")
        graph.add_conditional_edges(
            "route_query",
            self._route_branch,
            {"out_of_scope": "out_of_scope", "retrieve": "retrieve_context"},
        )
        graph.add_edge("out_of_scope", END)
        graph.add_conditional_edges(
            "retrieve_context",
            self._retrieval_branch,
            {"broaden": "broaden_retrieve", "generate": "generate_answer"},
        )
        graph.add_edge("broaden_retrieve", "generate_answer")
        graph.add_edge("generate_answer", "verify_answer")
        graph.add_edge("verify_answer", "compute_confidence")
        graph.add_conditional_edges(
            "compute_confidence",
            self._confidence_branch,
            {"human_review": "human_review", "finalize": "finalize_response"},
        )
        graph.add_edge("human_review", END)
        graph.add_edge("finalize_response", END)
        return graph.compile()

    @staticmethod
    def _validate_input(state: AgentState) -> AgentState:
        question = state["question"].strip()
        if not question:
            raise ValueError("question must not be empty")
        return {"question": question, "retry_count": state.get("retry_count", 0)}

    def _route_query(self, state: AgentState) -> AgentState:
        return {"route": self.router.route(state["question"])}

    @staticmethod
    def _route_branch(state: AgentState) -> str:
        return "out_of_scope" if state["route"].category == "general" else "retrieve"

    def _out_of_scope(self, state: AgentState) -> AgentState:
        response = AgentResponse(
            question=state["question"],
            answer="This question is outside the ECU manual scope.",
            route_category="general",
            retriever_mode=self.config.retriever_mode,
            used_llm=False,
            fallback_reason="out_of_scope",
            sources=(),
            verifier_status="unsupported",
            confidence=0.0,
            needs_human_review=False,
            review_reason="",
        )
        return {"retrieved_context": [], "confidence": 0.0, "response": response}

    def _retrieve_context(self, state: AgentState) -> AgentState:
        route = state["route"]
        if not self.retriever.chunks:
            return {"retrieved_context": []}
        min_route_depth = (
            len(route.required_sources) * 2
            if route.category in {"comparison", "feature_availability"}
            else len(route.required_sources)
        )
        results = self.retriever.retrieve(
            state["question"],
            required_sources=set(route.required_sources) or None,
            top_k=max(self.config.top_k, min_route_depth),
        )
        return {"retrieved_context": results}

    def _broaden_retrieve(self, state: AgentState) -> AgentState:
        if not self.retriever.chunks:
            return {"retrieved_context": [], "retry_count": state.get("retry_count", 0) + 1}
        results = self.retriever.retrieve(
            state["question"],
            required_sources=None,
            top_k=self.config.top_k * 2,
        )
        return {"retrieved_context": results, "retry_count": state.get("retry_count", 0) + 1}

    def _retrieval_branch(self, state: AgentState) -> str:
        if state.get("retry_count", 0) > 0:
            return "generate"
        results = state.get("retrieved_context", [])
        if not results or _average_score(results) < self.config.retrieval_retry_threshold:
            return "broaden"
        return "generate"

    def _generate_answer(self, state: AgentState) -> AgentState:
        generation = self.generator.generate(
            question=state["question"],
            route=state["route"],
            retrieved_context=state.get("retrieved_context", []),
        )
        return {"generation": generation}

    @staticmethod
    def _verify_answer(state: AgentState) -> AgentState:
        verification = verify_answer(
            state["generation"].answer,
            state.get("retrieved_context", []),
        )
        return {"verification": verification}

    @staticmethod
    def _compute_confidence(state: AgentState) -> AgentState:
        retrieved = state.get("retrieved_context", [])
        route = state["route"]
        confidence = compute_confidence(
            retrieval_confidence=_average_score(retrieved),
            source_coverage=_source_coverage(route.required_sources, retrieved),
            verifier_score=state["verification"].score,
        )
        return {"confidence": confidence}

    def _confidence_branch(self, state: AgentState) -> str:
        review = human_review_decision(
            confidence=state["confidence"],
            verifier_status=state["verification"].status,
            threshold=self.config.confidence_threshold,
        )
        return "human_review" if review.needs_human_review else "finalize"

    def _human_review(self, state: AgentState) -> AgentState:
        return {"response": self._build_response(state, force_human_review=True)}

    def _finalize_response(self, state: AgentState) -> AgentState:
        return {"response": self._build_response(state, force_human_review=False)}

    def _build_response(self, state: AgentState, *, force_human_review: bool) -> AgentResponse:
        review = human_review_decision(
            confidence=state["confidence"],
            verifier_status=state["verification"].status,
            threshold=self.config.confidence_threshold,
        )
        needs_review = force_human_review or review.needs_human_review
        return AgentResponse(
            question=state["question"],
            answer=state["generation"].answer,
            route_category=state["route"].category,
            retriever_mode=self.config.retriever_mode,
            used_llm=state["generation"].used_llm,
            fallback_reason=state["generation"].fallback_reason,
            sources=_sources(state.get("retrieved_context", [])),
            verifier_status=state["verification"].status,
            confidence=state["confidence"],
            needs_human_review=needs_review,
            review_reason=review.review_reason if needs_review else "",
        )


def _average_score(results: list[RetrievalResult]) -> float:
    if not results:
        return 0.0
    return round(sum(result.score for result in results) / len(results), 4)


def _source_coverage(required_sources: tuple[str, ...], results: list[RetrievalResult]) -> float:
    if not required_sources:
        return 1.0 if results else 0.0
    retrieved_sources = set(_sources(results))
    return round(len(retrieved_sources & set(required_sources)) / len(required_sources), 4)


def _sources(results: list[RetrievalResult]) -> tuple[str, ...]:
    seen: list[str] = []
    for result in results:
        source = result.chunk.metadata.get("source")
        if isinstance(source, str) and source not in seen:
            seen.append(source)
    return tuple(seen)


def main() -> None:
    """Console entry point for a single agent query."""

    parser = argparse.ArgumentParser(description="Ask the ME Engineering Assistant a question.")
    parser.add_argument("question")
    args = parser.parse_args()
    response = EngineeringAssistant.from_config().ask(args.question)
    print(json.dumps(response.to_dict(), indent=2))


if __name__ == "__main__":
    main()
