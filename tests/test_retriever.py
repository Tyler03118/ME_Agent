from me_agent.retriever import InMemoryKeywordRetriever
from me_agent.schemas import ManualChunk


def test_retriever_returns_relevant_chunks_with_scores() -> None:
    chunks = [
        ManualChunk(
            content="The ECU-850 has 2 GB LPDDR4 RAM.",
            metadata={"source": "ECU-800_Series_Base.md"},
        ),
        ManualChunk(
            content="The ECU-750 supports CAN FD up to 1 Mbps.",
            metadata={"source": "ECU-700_Series_Manual.md"},
        ),
    ]

    results = InMemoryKeywordRetriever(chunks).retrieve("How much RAM does ECU-850 have?")

    assert results[0].chunk.metadata["source"] == "ECU-800_Series_Base.md"
    assert 0.0 <= results[0].score <= 1.0


def test_retriever_applies_source_filter() -> None:
    chunks = [
        ManualChunk(content="ECU-850 RAM is 2 GB.", metadata={"source": "base.md"}),
        ManualChunk(content="ECU-850b RAM is 4 GB.", metadata={"source": "plus.md"}),
    ]

    results = InMemoryKeywordRetriever(chunks).retrieve(
        "RAM",
        required_sources={"plus.md"},
    )

    assert len(results) == 1
    assert results[0].chunk.metadata["source"] == "plus.md"


def test_build_retriever_supports_keyword_vector_and_hybrid_modes() -> None:
    from me_agent.retriever import (
        HybridRetriever,
        InMemoryVectorRetriever,
        build_retriever,
    )

    chunks = [
        ManualChunk(content="ECU-750 has Single Channel CAN FD at 1 Mbps.", metadata={"source": "700.md"}),
        ManualChunk(content="ECU-850 has Dual Channel CAN FD at 2 Mbps.", metadata={"source": "800.md"}),
    ]

    keyword = build_retriever("keyword", chunks, top_k=2)
    vector = build_retriever("vector", chunks, top_k=2)
    hybrid = build_retriever("hybrid", chunks, top_k=2)

    assert isinstance(keyword, InMemoryKeywordRetriever)
    assert isinstance(vector, InMemoryVectorRetriever)
    assert isinstance(hybrid, HybridRetriever)


def test_vector_retriever_uses_in_memory_vectors_and_source_filters() -> None:
    from me_agent.retriever import InMemoryVectorRetriever

    chunks = [
        ManualChunk(
            content="ECU-850b includes a Neural Processing Unit for edge AI workloads.",
            metadata={"source": "plus.md"},
        ),
        ManualChunk(
            content="ECU-750 has a basic CAN FD interface.",
            metadata={"source": "legacy.md"},
        ),
    ]

    retriever = InMemoryVectorRetriever(chunks, top_k=2)
    results = retriever.retrieve("AI accelerator", required_sources={"plus.md"})

    assert results
    assert results[0].chunk.metadata["source"] == "plus.md"
    assert results[0].score > 0


def test_hybrid_retriever_preserves_exact_terms_and_required_source_coverage() -> None:
    from me_agent.retriever import HybridRetriever

    chunks = [
        ManualChunk(
            content="ECU-750 provides Single Channel CAN FD up to 1 Mbps.",
            metadata={"source": "700.md"},
        ),
        ManualChunk(
            content="ECU-850 provides Dual Channel CAN FD up to 2 Mbps per channel.",
            metadata={"source": "800.md"},
        ),
        ManualChunk(
            content="ECU-850b includes a Neural Processing Unit capable of 5 TOPS.",
            metadata={"source": "plus.md"},
        ),
    ]

    retriever = HybridRetriever(chunks, top_k=2)
    results = retriever.retrieve(
        "Compare ECU-750 and ECU-850 CAN capabilities",
        required_sources={"700.md", "800.md"},
        top_k=2,
    )

    sources = {result.chunk.metadata["source"] for result in results}
    joined = " ".join(result.chunk.content for result in results)
    assert sources == {"700.md", "800.md"}
    assert "Single Channel" in joined
    assert "Dual Channel" in joined


def test_unknown_retriever_mode_is_rejected() -> None:
    from me_agent.retriever import build_retriever

    try:
        build_retriever("unknown", [], top_k=1)
    except ValueError as exc:
        assert "retriever mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown retriever mode")


def test_keyword_retriever_does_not_initialize_embedding_layer(monkeypatch) -> None:
    import me_agent.retriever as retriever_module

    def fail_embedding_init():
        raise AssertionError("keyword mode must not initialize embeddings")

    monkeypatch.setattr(retriever_module, "EmbeddingModel", fail_embedding_init)
    chunks = [ManualChunk(content="ECU-850 has 2 GB LPDDR4 RAM.", metadata={"source": "base.md"})]

    retriever = retriever_module.build_retriever("keyword", chunks, top_k=1)
    results = retriever.retrieve("RAM")

    assert results[0].chunk.metadata["source"] == "base.md"


def test_vector_retriever_uses_prebuilt_vector_store() -> None:
    from me_agent.retriever import InMemoryVectorRetriever
    from me_agent.vector_store import VectorStore

    chunks = [ManualChunk(content="ECU-850b has an NPU.", metadata={"source": "plus.md"})]

    retriever = InMemoryVectorRetriever(chunks, top_k=1)

    assert isinstance(retriever.vector_store, VectorStore)


def test_hybrid_retriever_only_merges_keyword_and_vector_results() -> None:
    from me_agent.retriever import HybridRetriever
    from me_agent.schemas import RetrievalResult

    class StubRetriever:
        def __init__(self, results):
            self.chunks = [result.chunk for result in results]
            self.results = results

        def retrieve(self, query, *, required_sources=None, top_k=None):
            del query, required_sources, top_k
            return self.results

    keyword_chunk = ManualChunk(content="exact command", metadata={"source": "a.md", "chunk_id": "a:0"})
    vector_chunk = ManualChunk(content="semantic match", metadata={"source": "b.md", "chunk_id": "b:0"})
    duplicate_chunk = ManualChunk(content="exact command", metadata={"source": "a.md", "chunk_id": "a:0"})
    hybrid = HybridRetriever(
        [],
        keyword_retriever=StubRetriever([RetrievalResult(keyword_chunk, 1.0)]),
        vector_retriever=StubRetriever([
            RetrievalResult(duplicate_chunk, 0.9),
            RetrievalResult(vector_chunk, 0.8),
        ]),
        top_k=2,
    )

    results = hybrid.retrieve("query", top_k=2)

    assert [result.chunk.metadata["chunk_id"] for result in results] == ["a:0", "b:0"]
