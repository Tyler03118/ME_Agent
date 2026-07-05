"""Shared dataclass schemas for the ME Engineering Assistant."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RouteCategory = Literal[
    "ecu_700_lookup",
    "ecu_800_lookup",
    "ecu_850b_lookup",
    "comparison",
    "feature_availability",
    "configuration",
    "general",
]
VerifierStatus = Literal["supported", "partially_supported", "unsupported", "contradicted"]


@dataclass(frozen=True)
class ManualDocument:
    """A loaded source manual with metadata extracted from path and content."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManualChunk:
    """A retrievable text chunk derived from a manual."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """Retriever output containing the matched chunk and normalized score."""

    chunk: ManualChunk
    score: float


@dataclass(frozen=True)
class RouteDecision:
    """Routing decision for a user query."""

    category: RouteCategory
    required_sources: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class VerificationResult:
    """Verifier judgment for answer support against retrieved context."""

    status: VerifierStatus
    score: float
    rationale: str


@dataclass(frozen=True)
class HumanReviewDecision:
    """Human-in-the-loop review flag."""

    needs_human_review: bool
    review_reason: str


@dataclass(frozen=True)
class AgentResponse:  # pylint: disable=too-many-instance-attributes
    """Structured assistant response returned by graph and MLflow entry points."""

    question: str
    answer: str
    route_category: RouteCategory
    retriever_mode: str
    used_llm: bool
    fallback_reason: str
    sources: tuple[str, ...]
    verifier_status: VerifierStatus
    confidence: float
    needs_human_review: bool
    review_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the response into plain Python containers."""

        return asdict(self)


@dataclass(frozen=True)
class EvaluationCase:  # pylint: disable=too-many-instance-attributes
    """Single golden evaluation question."""

    question_id: str
    category: str
    question: str
    expected_answer: str
    evaluation_criteria: str
    required_facts: tuple[str, ...] = ()
    forbidden_facts: tuple[str, ...] = ()
    expected_sources: tuple[str, ...] = ()
    expected_route: str = ""


@dataclass(frozen=True)
class EvaluationResult:  # pylint: disable=too-many-instance-attributes
    """Evaluation output for one question."""

    case: EvaluationCase
    response: AgentResponse
    latency_seconds: float
    semantic_similarity: float
    token_coverage: float
    required_fact_recall: float = 1.0
    forbidden_fact_violations: int = 0
    source_match: float = 1.0
    route_match: float = 1.0
    combined_score: float = 0.0
    passed: bool = False
    source_diagnostic: float = 0.0
    route_diagnostic: str = ""
