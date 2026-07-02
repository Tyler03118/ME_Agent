import numpy as np

from me_agent.core.schemas import ManualChunk
from me_agent.retrieval.vector_store import VectorStore


class CountingEmbeddingModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.info = None

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            lower = text.lower()
            rows.append([
                float("neural" in lower or "ai" in lower),
                float("can" in lower),
                float("ram" in lower),
            ])
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


def test_vector_store_builds_index_once_and_searches_without_reindexing() -> None:
    chunks = [
        ManualChunk(
            content="ECU-850b has a Neural Processing Unit for edge AI workloads.",
            metadata={"source": "plus.md", "chunk_id": "plus.md:0"},
        ),
        ManualChunk(
            content="ECU-750 has CAN FD support.",
            metadata={"source": "legacy.md", "chunk_id": "legacy.md:0"},
        ),
    ]
    embeddings = CountingEmbeddingModel()

    store = VectorStore.from_chunks(chunks, embeddings)
    results = store.search("AI accelerator", top_k=1)

    assert store.backend in {"faiss", "numpy"}
    assert len(embeddings.calls) == 2
    assert embeddings.calls[0] == [chunk.content for chunk in chunks]
    assert embeddings.calls[1] == ["AI accelerator"]
    assert results[0].chunk.metadata["source"] == "plus.md"
    assert results[0].score > 0


def test_vector_store_can_filter_sources() -> None:
    chunks = [
        ManualChunk(content="AI accelerator", metadata={"source": "plus.md"}),
        ManualChunk(content="CAN bus", metadata={"source": "base.md"}),
    ]
    store = VectorStore.from_chunks(chunks, CountingEmbeddingModel())

    results = store.search("AI", top_k=2, required_sources={"base.md"})

    assert [result.chunk.metadata["source"] for result in results] == ["base.md"]
