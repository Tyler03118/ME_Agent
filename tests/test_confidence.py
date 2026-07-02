from me_agent.workflow.confidence import compute_confidence
from me_agent.workflow.hitl import human_review_decision


def test_confidence_uses_weighted_formula() -> None:
    score = compute_confidence(
        retrieval_confidence=0.5,
        source_coverage=1.0,
        verifier_score=0.5,
    )

    assert score == 0.7


def test_human_review_flags_low_confidence() -> None:
    review = human_review_decision(confidence=0.4, verifier_status="supported")

    assert review.needs_human_review is True
    assert "confidence" in review.review_reason
