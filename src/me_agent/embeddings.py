"""Embedding layer for local vector retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


class EmbeddingModel:
    """Small deterministic embedding wrapper used by the local vector store."""

    def encode(self, texts: list[str]) -> list[dict[str, float]]:
        """Encode texts into normalized sparse vectors."""

        return [_normalize(_term_counts(_expanded_tokens(text))) for text in texts]


def tokenize(text: str) -> list[str]:
    """Tokenize text for keyword and vector representations."""

    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _expanded_tokens(text: str) -> list[str]:
    terms = tokenize(text)
    expanded = list(terms)
    token_set = set(terms)
    if "ai" in token_set or "accelerator" in token_set:
        expanded.extend(["neural", "processing", "unit", "npu", "tops", "edge"])
    if "npu" in token_set:
        expanded.extend(["neural", "processing", "unit", "ai", "accelerator"])
    if "memory" in token_set:
        expanded.append("ram")
    if "ram" in token_set:
        expanded.append("memory")
    if "over" in token_set and "air" in token_set:
        expanded.append("ota")
    if "ota" in token_set:
        expanded.extend(["over", "air", "updates"])
    return expanded


def _term_counts(terms: list[str]) -> dict[str, float]:
    counts = Counter(terms)
    return {term: float(count) for term, count in counts.items()}


def _normalize(vector: Mapping[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}
    return {term: value / norm for term, value in vector.items()}
