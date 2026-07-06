"""Answer verification against retrieved context."""

from __future__ import annotations

import re

from me_agent.retrieval.retriever import _tokens
from me_agent.core.schemas import RetrievalResult, VerificationResult


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "provided", "the", "to",
    "with", "while", "than", "this", "that", "following", "follows", "models",
    "model", "series", "base", "plus", "md", "markdown", "evidence",
}
CITATION_PATTERN = re.compile(r"\[[^\]]+\]")
MEASUREMENT_PATTERN = re.compile(
    r"(?<![\w.])[+-]?\d+(?:\.\d+)?\s*(?:°\s*)?(?:c|gb|mb|kb|ghz|mhz|mbps|tops|ma|a)\b",
    re.IGNORECASE,
)


def verify_answer(  # pylint: disable=too-many-return-statements
    answer: str,
    retrieved_context: list[RetrievalResult],
) -> VerificationResult:
    """Check whether the generated answer is backed by retrieved text.

    Decision flow:
    - no retrieved chunks -> unsupported;
    - answer says context is insufficient -> unsupported;
    - answer introduces unseen measurements -> contradicted;
    - otherwise compare answer tokens with context tokens;
    - high overlap -> supported, medium overlap -> partially supported.
    """

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
    context_measurements = set()
    for result in retrieved_context:
        context_tokens.update(_content_tokens(result.chunk.content))
        context_measurements.update(_measurements(result.chunk.content))
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return VerificationResult("unsupported", 0.0, "The answer is empty.")

    missing_measurements = _measurements(answer) - context_measurements
    if missing_measurements:
        missing = ", ".join(sorted(missing_measurements))
        return VerificationResult(
            "contradicted",
            0.0,
            f"Answer includes numeric fact(s) not present in retrieved context: {missing}.",
        )

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


def _content_tokens(text: str) -> set[str]:
    """Remove citations, tokenize text, and drop verifier stopwords."""

    cleaned = CITATION_PATTERN.sub(" ", text)
    return {token for token in _tokens(cleaned) if token not in STOPWORDS}


def _measurements(text: str) -> set[str]:
    """Extract numeric specs such as temperature, memory, speed, and current.

    Normalization examples:
    - ``+105 °C`` and ``+105C`` both become ``+105c``;
    - citation brackets are removed before matching;
    - spaces inside a measurement are removed before comparison.
    """

    cleaned = CITATION_PATTERN.sub(" ", text).lower().replace("−", "-").replace("º", "°")
    measurements: set[str] = set()
    for match in MEASUREMENT_PATTERN.findall(cleaned):
        measurements.add(re.sub(r"\s+", "", match).replace("°", ""))
    return measurements
