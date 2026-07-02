from me_agent.embeddings import EmbeddingModel
from me_agent.schemas import ManualChunk
from me_agent.vector_store import VectorStore


class CountingEmbeddingModel(EmbeddingModel):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]):
        self.calls.append(list(texts))
        return super().encode(texts)


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

    assert len(embeddings.calls) == 2
    assert embeddings.calls[0] == [chunk.content for chunk in chunks]
    assert embeddings.calls[1] == ["AI accelerator"]
    assert results[0].chunk.metadata["source"] == "plus.md"
    assert results[0].score > 0
