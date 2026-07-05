"""Keyword, vector, and hybrid retrievers over preprocessed chunks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

from me_agent.core.domain_terms import RETRIEVAL_QUERY_EXPANSION_GROUPS
from me_agent.retrieval.embeddings import EmbeddingModel
from me_agent.core.schemas import ManualChunk, RetrievalResult
from me_agent.retrieval.vector_store import VectorStore

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
SUPPORTED_RETRIEVER_MODES = {"keyword", "vector", "hybrid"}


# Retriever implementations -------------------------------------------------


class Retriever(Protocol):
    """Common retriever interface used by the LangGraph workflow."""

    chunks: list[ManualChunk]

    def retrieve(
        self,
        query: str,
        *,
        required_sources: set[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return relevant chunks for the query."""


class InMemoryKeywordRetriever:
    """Keyword-overlap retriever for exact identifiers and numeric specs.

    ECU questions often depend on tokens such as model IDs, command flags, units,
    and storage sizes. A simple overlap score keeps those exact matches visible
    even when vector similarity would blur them.
    """

    def __init__(self, chunks: Iterable[ManualChunk], top_k: int = 4) -> None:
        """Store chunks and the default retrieval depth."""

        self.chunks = list(chunks)
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        *,
        required_sources: set[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return top chunks by normalized token overlap."""

        query_tokens = _query_tokens(query)
        if not query_tokens:
            raise ValueError("query must contain at least one searchable token")
        limit = top_k or self.top_k
        candidates = _candidate_chunks(self.chunks, required_sources)
        scored = [
            RetrievalResult(chunk=chunk, score=_keyword_score(query_tokens, _tokens(chunk.content)))
            for chunk in candidates
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return _with_required_source_coverage(scored, required_sources, limit)


class FullDocumentRetriever(InMemoryKeywordRetriever):
    """Backward-compatible alias for the original keyword retriever."""


class InMemoryVectorRetriever:
    """Vector retriever backed by a prebuilt VectorStore."""

    def __init__(
        self,
        chunks: Iterable[ManualChunk],
        top_k: int = 4,
        *,
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Store chunks and build or reuse the vector store."""

        self.chunks = list(chunks)
        self.top_k = top_k
        self.vector_store = vector_store or VectorStore.from_chunks(
            self.chunks,
            embedding_model or EmbeddingModel(),
        )

    def retrieve(
        self,
        query: str,
        *,
        required_sources: set[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return top chunks from the prebuilt vector index."""

        if not _tokens(query):
            raise ValueError("query must contain at least one searchable token")
        limit = top_k or self.top_k
        scored = self.vector_store.search(
            _expanded_query_text(query),
            top_k=max(limit, len(required_sources or ())),
            required_sources=required_sources,
        )
        return _with_required_source_coverage(scored, required_sources, limit)


class HybridRetriever:
    """Hybrid retriever that merges keyword-first results with vector recall.

    Keyword results are kept first because exact specifications are the highest
    precision signal in this domain. Vector results add recall for paraphrases;
    the merge deduplicates by chunk id without adding an opaque reranker.
    """

    def __init__(
        self,
        chunks: Iterable[ManualChunk],
        *,
        top_k: int = 4,
        keyword_retriever: Retriever | None = None,
        vector_retriever: Retriever | None = None,
    ) -> None:
        """Create keyword and vector retrievers over the same chunk set."""

        self.chunks = list(chunks)
        self.top_k = top_k
        self._keyword = keyword_retriever or InMemoryKeywordRetriever(self.chunks, top_k=top_k)
        self._vector = vector_retriever or InMemoryVectorRetriever(self.chunks, top_k=top_k)
        if not self.chunks:
            self.chunks = _dedupe_chunks(self._keyword.chunks + self._vector.chunks)

    def retrieve(
        self,
        query: str,
        *,
        required_sources: set[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return merged keyword and vector results without reranking."""

        limit = top_k or self.top_k
        candidate_limit = max(limit, len(required_sources or ()))
        keyword_results = self._keyword.retrieve(
            query,
            required_sources=required_sources,
            top_k=candidate_limit,
        )
        vector_results = self._vector.retrieve(
            query,
            required_sources=required_sources,
            top_k=candidate_limit,
        )
        merged = _merge_keyword_then_vector(keyword_results, vector_results)
        return _with_required_source_coverage(merged, required_sources, limit)


# Factory -------------------------------------------------------------------


def build_retriever(
    mode: str,
    chunks: Iterable[ManualChunk],
    *,
    top_k: int,
    embedding_backend: str = "auto",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Retriever:
    """Build a retriever for the configured mode.

    The vector retriever builds its index during construction, so runtime queries
    reuse the same embeddings and avoid rebuilding the corpus for each request.
    """

    normalized_mode = mode.lower().strip()
    chunk_list = list(chunks)
    if normalized_mode == "keyword":
        return InMemoryKeywordRetriever(chunk_list, top_k=top_k)
    if normalized_mode == "vector":
        return InMemoryVectorRetriever(
            chunk_list,
            top_k=top_k,
            embedding_model=EmbeddingModel(
                model_name=embedding_model_name,
                backend=embedding_backend,
            ),
        )
    if normalized_mode == "hybrid":
        vector = InMemoryVectorRetriever(
            chunk_list,
            top_k=top_k,
            embedding_model=EmbeddingModel(
                model_name=embedding_model_name,
                backend=embedding_backend,
            ),
        )
        return HybridRetriever(
            chunk_list,
            top_k=top_k,
            keyword_retriever=InMemoryKeywordRetriever(chunk_list, top_k=top_k),
            vector_retriever=vector,
        )
    raise ValueError(
        (
            f"Unsupported retriever mode: {mode}. "
            f"Expected one of {sorted(SUPPORTED_RETRIEVER_MODES)}."
        )
    )


# Ranking and source-coverage helpers ---------------------------------------


def _candidate_chunks(
    chunks: list[ManualChunk],
    required_sources: set[str] | None,
) -> list[ManualChunk]:
    """Filter chunks by required source filenames when a route requests them."""

    if not required_sources:
        return chunks
    return [chunk for chunk in chunks if chunk.metadata.get("source") in required_sources]


def _with_required_source_coverage(
    scored: list[RetrievalResult],
    required_sources: set[str] | None,
    limit: int,
) -> list[RetrievalResult]:
    """Keep at least one top result from each required source when possible."""

    if not required_sources:
        return scored[:limit]
    selected: list[RetrievalResult] = []
    for source in sorted(required_sources):
        source_results = [
            result for result in scored if result.chunk.metadata.get("source") == source
        ]
        if source_results:
            selected.append(source_results[0])
    selected_ids = {_chunk_id(result.chunk) for result in selected}
    for result in scored:
        if len(selected) >= max(limit, len(required_sources)):
            break
        if _chunk_id(result.chunk) not in selected_ids:
            selected.append(result)
            selected_ids.add(_chunk_id(result.chunk))
    return selected


def _merge_keyword_then_vector(
    keyword_results: list[RetrievalResult],
    vector_results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Merge retrieval lists while preserving keyword-first ordering."""

    merged: list[RetrievalResult] = []
    seen: set[str] = set()
    for result in [*keyword_results, *vector_results]:
        chunk_id = _chunk_id(result.chunk)
        if chunk_id in seen:
            continue
        merged.append(result)
        seen.add(chunk_id)
    return merged


def _dedupe_chunks(chunks: list[ManualChunk]) -> list[ManualChunk]:
    """Return chunks with duplicate chunk IDs removed in first-seen order."""

    deduped: list[ManualChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_id = _chunk_id(chunk)
        if chunk_id in seen:
            continue
        deduped.append(chunk)
        seen.add(chunk_id)
    return deduped


def _chunk_id(chunk: ManualChunk) -> str:
    """Return the stable identifier used for deduplication."""

    value = chunk.metadata.get("chunk_id") or chunk.metadata.get("source") or id(chunk)
    return str(value)


def _tokens(text: str) -> set[str]:
    """Tokenize text for keyword overlap scoring."""

    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}


def _query_tokens(text: str) -> set[str]:
    """Tokenize a query and add deterministic domain synonym expansions."""

    return _tokens(_expanded_query_text(text))


def _expanded_query_text(text: str) -> str:
    """Append domain synonyms that bridge user phrasing to manual wording."""

    normalized = text.lower()
    tokens = _tokens(normalized)
    expansions: list[str] = []
    for triggers, expansion in RETRIEVAL_QUERY_EXPANSION_GROUPS:
        if tokens & set(triggers):
            expansions.append(expansion)
    if not expansions:
        return text
    return " ".join([text, *expansions])


def _keyword_score(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    """Score a chunk by the share of query tokens it contains."""

    if not query_tokens or not chunk_tokens:
        return 0.0
    return round(len(query_tokens & chunk_tokens) / len(query_tokens), 4)
