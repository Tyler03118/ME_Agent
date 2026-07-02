from me_agent.chunking import MarkdownChunker
from me_agent.schemas import ManualDocument


def test_chunker_creates_chunks_and_preserves_metadata() -> None:
    document = ManualDocument(
        content="A" * 80 + "\n\n" + "B" * 80,
        metadata={"source": "manual.md", "model": "ECU-850"},
    )

    chunks = MarkdownChunker(chunk_size=60, chunk_overlap=10).split([document])

    assert len(chunks) > 1
    assert chunks[0].content
    assert chunks[0].metadata["source"] == "manual.md"
    assert chunks[0].metadata["model"] == "ECU-850"
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunker_rejects_invalid_overlap() -> None:
    try:
        MarkdownChunker(chunk_size=10, chunk_overlap=10)
    except ValueError as exc:
        assert "chunk_overlap must be smaller" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid chunk overlap")


def test_chunk_documents_returns_standardized_chunk_dicts() -> None:
    from me_agent.chunking import chunk_documents

    documents = [
        {
            "text": "A" * 40 + "B" * 40,
            "source": "manual.md",
        }
    ]

    chunks = chunk_documents(documents, chunk_size=50, overlap=10)

    assert chunks
    assert set(chunks[0]) == {"text", "source", "chunk_id"}
    assert chunks[0]["source"] == "manual.md"
    assert chunks[0]["chunk_id"] == "manual.md:0"
    assert len(chunks[0]["text"]) <= 50
