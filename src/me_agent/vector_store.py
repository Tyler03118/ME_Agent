"""Vector index layer for local in-memory retrieval."""

from __future__ import annotations

from collections.abc import Mapping

from me_agent.embeddings import EmbeddingModel
from me_agent.schemas import ManualChunk, RetrievalResult


class VectorStore:
    """Simple in-memory vector index built once at startup."""

    def __init__(
        self,
        chunks: list[ManualChunk],
        vectors: list[dict[str, float]],
        embedding_model: EmbeddingModel,
    ) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.embedding_model = embedding_model

    @classmethod
    def from_chunks(
        cls,
        chunks: list[ManualChunk],
        embedding_model: EmbeddingModel | None = None,
    ) -> "VectorStore":
        """Build an index from chunks once during retriever initialization."""

        resolved_model = embedding_model or EmbeddingModel()
        vectors = resolved_model.encode([chunk.content for chunk in chunks])
        return cls(chunks=list(chunks), vectors=vectors, embedding_model=resolved_model)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        required_sources: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """Search the prebuilt index with cosine similarity."""

        query_vector = self.embedding_model.encode([query])[0]
        scored: list[RetrievalResult] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if required_sources and chunk.metadata.get("source") not in required_sources:
                continue
            scored.append(
                RetrievalResult(
                    chunk=chunk,
                    score=round(_cosine(query_vector, vector), 4),
                )
            )
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())
