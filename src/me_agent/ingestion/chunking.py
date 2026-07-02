"""Markdown chunking utilities."""

from __future__ import annotations

from typing import Any

from me_agent.core.schemas import ManualChunk, ManualDocument


def chunk_documents(
    docs: list[ManualDocument] | list[dict[str, Any]],
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict[str, str]]:
    """Split raw documents into standardized chunk dictionaries.

    This is a preprocessing-only helper. It does not embed, score, or retrieve.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[dict[str, str]] = []
    for document in docs:
        text, source = _document_text_and_source(document)
        text = text.strip()
        if not text:
            continue
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source,
                        "chunk_id": f"{source}:{chunk_index}",
                    }
                )
                chunk_index += 1
            if end == len(text):
                break
            start = end - overlap
    return chunks


class MarkdownChunker:
    """Split documents into deterministic overlapping character chunks."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[ManualDocument]) -> list[ManualChunk]:
        """Split all documents while preserving source metadata."""

        chunks: list[ManualChunk] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks

    def split_document(self, document: ManualDocument) -> list[ManualChunk]:
        """Split one document into chunks."""

        standardized_chunks = chunk_documents(
            [document],
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        manual_chunks: list[ManualChunk] = []
        for index, chunk in enumerate(standardized_chunks):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            metadata["chunk_id"] = chunk["chunk_id"]
            manual_chunks.append(ManualChunk(content=chunk["text"], metadata=metadata))
        return manual_chunks


def _document_text_and_source(document: ManualDocument | dict[str, Any]) -> tuple[str, str]:
    if isinstance(document, ManualDocument):
        return document.content, str(document.metadata.get("source", "document"))
    text = str(document.get("text", document.get("content", "")))
    source = str(document.get("source", document.get("metadata", {}).get("source", "document")))
    return text, source
