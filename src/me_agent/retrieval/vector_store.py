"""In-memory vector index backed by FAISS with a numpy fallback."""

from __future__ import annotations

import numpy as np

from me_agent.retrieval.embeddings import EmbeddingModel
from me_agent.core.schemas import ManualChunk, RetrievalResult


class VectorStore:
    """Build a vector index once and search it for each query.

    FAISS is used when available for fast inner-product search. The numpy path is
    kept as a first-class fallback and is also used when source filtering is
    requested, because FAISS does not know about per-chunk metadata.
    """

    def __init__(
        self,
        chunks: list[ManualChunk],
        vectors: np.ndarray,
        embedding_model: EmbeddingModel,
        index=None,
    ) -> None:
        """Store vectors, chunks, embedding model, and optional FAISS index."""

        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.embedding_model = embedding_model
        self.index = index
        self.backend = "faiss" if index is not None else "numpy"

    @classmethod
    def from_chunks(
        cls,
        chunks: list[ManualChunk],
        embedding_model: EmbeddingModel | None = None,
    ) -> "VectorStore":
        """Build an in-memory index from chunks during retriever initialization.

        Corpus embeddings are computed once here. Query-time work is then
        limited to encoding the user query and scoring against this fixed matrix
        or FAISS index.
        """

        resolved_model = embedding_model or EmbeddingModel()
        chunk_list = list(chunks)
        vectors = resolved_model.encode([chunk.content for chunk in chunk_list])
        index = _build_faiss_index(vectors)
        return cls(chunk_list, vectors, resolved_model, index=index)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        required_sources: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """Search the existing vector index and return scored chunks.

        Embeddings are L2-normalized, so inner product is equivalent to cosine
        similarity. FAISS handles the unfiltered fast path; metadata-filtered
        searches use numpy over the allowed candidate rows.
        """

        if not self.chunks or top_k <= 0:
            return []
        query_vector = self.embedding_model.encode([query])
        candidate_indices = self._candidate_indices(required_sources)
        if not candidate_indices:
            return []
        if self.index is not None and required_sources is None:
            # Fast path: FAISS can search the full corpus directly when no
            # metadata filter is needed.
            scores, indices = self.index.search(
                query_vector.astype(np.float32),
                min(top_k, len(self.chunks)),
            )
            pairs = [
                (int(index), float(score))
                for index, score in zip(indices[0], scores[0])
                if index >= 0
            ]
        else:
            # Filtered path: apply source constraints in Python, then score the
            # smaller candidate matrix with the same normalized embeddings.
            candidate_vectors = self.vectors[candidate_indices]
            scores = candidate_vectors @ query_vector[0]
            order = np.argsort(scores)[::-1][:top_k]
            pairs = [
                (candidate_indices[int(position)], float(scores[int(position)]))
                for position in order
            ]
        return [
            RetrievalResult(chunk=self.chunks[index], score=round(max(score, 0.0), 4))
            for index, score in pairs
        ]

    def _candidate_indices(self, required_sources: set[str] | None) -> list[int]:
        """Return vector row indices allowed by source constraints."""

        if not required_sources:
            return list(range(len(self.chunks)))
        return [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.metadata.get("source") in required_sources
        ]


def _build_faiss_index(vectors: np.ndarray):
    """Build a FAISS inner-product index when FAISS and vectors are available."""

    try:
        import faiss  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    if vectors.size == 0:
        return None
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))  # pylint: disable=no-value-for-parameter
    return index
