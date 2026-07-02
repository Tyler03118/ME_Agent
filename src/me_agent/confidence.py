"""Confidence scoring for grounded ECU answers."""

from __future__ import annotations


def compute_confidence(
    *,
    retrieval_confidence: float,
    source_coverage: float,
    verifier_score: float,
) -> float:
    """Compute the documented heuristic confidence score."""

    retrieval = _clamp(retrieval_confidence)
    coverage = _clamp(source_coverage)
    verifier = _clamp(verifier_score)
    return round(0.40 * retrieval + 0.40 * coverage + 0.20 * verifier, 4)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
