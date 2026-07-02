"""Answer verification against retrieved context."""

from __future__ import annotations

from me_agent.retriever import _tokens
from me_agent.schemas import RetrievalResult, VerificationResult


def verify_answer(answer: str, retrieved_context: list[RetrievalResult]) -> VerificationResult:
    """Heuristically judge whether an answer is supported by retrieved chunks."""

    if not retrieved_context:
        return VerificationResult(
            status="unsupported",
            score=0.0,
            rationale="No retrieved context was available.",
        )
    if "do not contain enough information" in answer.lower():
        return VerificationResult(
            status="unsupported",
            score=0.0,
            rationale="The answer explicitly reports insufficient context.",
        )

    context_tokens = set()
    for result in retrieved_context:
        context_tokens.update(_tokens(result.chunk.content))
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return VerificationResult("unsupported", 0.0, "The answer is empty.")

    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    if overlap >= 0.6:
        return VerificationResult("supported", 1.0, "Most answer tokens appear in context.")
    if overlap >= 0.3:
        return VerificationResult(
            "partially_supported",
            0.5,
            "Some answer tokens appear in context.",
        )
    return VerificationResult("unsupported", 0.0, "The answer has low overlap with context.")
