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
    """Interface shared by keyword, vector, and hybrid retrievers."""

    chunks: list[ManualChunk]

    def retrieve(
        self,
        query: str,
        *,
        required_sources: set[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return ranked chunks for ``query`` as ``RetrievalResult`` objects."""


class InMemoryKeywordRetriever:
    """Find chunks by exact token overlap.

    This is useful for ECU manuals because many answers depend on exact strings:
    - model IDs such as ``ECU-850b``;
    - units such as ``+105C``, ``512 KB``, or ``2 MB``;
    - feature terms such as ``OTA``, ``NPU``, or ``CAN``.
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
        """Score chunks by how many query tokens they contain.

        Steps:
        - expand the query with domain synonyms;
        - optionally keep only chunks from ``required_sources``;
        - score each chunk with ``_keyword_score``;
        - sort highest score first;
        - make sure required sources are represented in the final list.
        """

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


class InMemoryVectorRetriever:
    """Find chunks by embedding similarity using a prebuilt ``VectorStore``."""

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
        """Search the vector index for semantically similar chunks.

        The query is expanded before vector search so paraphrases like
        ``thermal tolerance`` also include manual wording like
        ``operating temperature``.
        """

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
    """Run keyword and vector retrieval, then merge their results.

    Merge behavior:
    - keyword results come first because exact specs should win;
    - vector results are appended to catch paraphrases;
    - duplicate chunks are removed by ``chunk_id``;
    - source coverage is applied after merging.
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
        """Return a keyword-first union of keyword and vector matches.

        ``candidate_limit`` may be larger than ``top_k`` so comparison queries
        have room to include at least one chunk from every required manual.
        """

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
    """Create the retriever selected by config.

    Supported modes:
    - ``keyword``: exact token overlap only;
    - ``vector``: embedding search only;
    - ``hybrid``: keyword-first merge with vector recall.

    Vector indexes are built here once, then reused for every query.
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
    """Return all chunks, or only chunks whose ``source`` is required."""

    if not required_sources:
        return chunks
    return [chunk for chunk in chunks if chunk.metadata.get("source") in required_sources]


def _with_required_source_coverage(
    scored: list[RetrievalResult],
    required_sources: set[str] | None,
    limit: int,
) -> list[RetrievalResult]:
    """Add source coverage to an already-scored result list.

    Example for a comparison query requiring three manuals:
    - first pick the best result from manual A, B, and C;
    - then fill remaining slots with the next highest-scoring chunks;
    - skip any chunk already selected in the first pass.
    """

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
    """Append vector results after keyword results and remove duplicates."""

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
    """Keep the first copy of each chunk ID and drop later duplicates."""

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
    """Return the best available stable key for one chunk."""

    value = chunk.metadata.get("chunk_id") or chunk.metadata.get("source") or id(chunk)
    return str(value)


def _tokens(text: str) -> set[str]:
    """Convert text into lowercase alphanumeric tokens for overlap scoring."""

    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}


def _query_tokens(text: str) -> set[str]:
    """Tokenize a query after adding domain synonym text."""

    return _tokens(_expanded_query_text(text))


def _expanded_query_text(text: str) -> str:
    """Append manual-style wording when a query uses common paraphrases.

    Example:
    - user asks: ``best thermal tolerance``;
    - expansion adds wording such as ``operating temperature``;
    - keyword and vector search can now match the table row in the manuals.
    """

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
    """Return ``matched query tokens / total query tokens`` rounded to 4 dp."""

    if not query_tokens or not chunk_tokens:
        return 0.0
    return round(len(query_tokens & chunk_tokens) / len(query_tokens), 4)
