"""Human-in-the-loop review decisions."""

from __future__ import annotations

from me_agent.core.schemas import HumanReviewDecision, VerifierStatus


def human_review_decision(
    *,
    confidence: float,
    verifier_status: VerifierStatus,
    threshold: float = 0.65,
) -> HumanReviewDecision:
    """Flag low-confidence or unsupported answers for human review."""

    if verifier_status in {"unsupported", "contradicted"}:
        return HumanReviewDecision(
            needs_human_review=True,
            review_reason=f"Verifier marked answer as {verifier_status}.",
        )
    if confidence < threshold:
        return HumanReviewDecision(
            needs_human_review=True,
            review_reason=f"Answer confidence {confidence:.2f} is below threshold {threshold:.2f}.",
        )
    return HumanReviewDecision(False, "")
