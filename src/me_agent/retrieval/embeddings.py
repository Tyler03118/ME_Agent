"""Embedding backends for vector retrieval and semantic evaluation."""

from __future__ import annotations

import hashlib
import os
import re
from importlib import metadata
from dataclasses import dataclass

import numpy as np

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


@dataclass(frozen=True)
class EmbeddingInfo:
    """Runtime information about the selected embedding backend."""

    backend: str
    model_name: str
    dimension: int
    fallback_reason: str = ""


class EmbeddingModel:
    """Small wrapper that always returns usable embedding vectors.

    Backend behavior:
    - use sentence-transformers when it is installed and compatible;
    - use deterministic token hashing when offline or dependencies are missing;
    - expose ``self.info`` so reports can show which backend actually ran.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        backend: str = "auto",
        fallback_dimensions: int = 384,
    ) -> None:
        """Select and initialize the requested embedding backend."""

        self.model_name = model_name
        self.requested_backend = backend.lower().strip()
        self.fallback_dimensions = fallback_dimensions
        self._model = None
        self.info = self._load_backend()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Turn a list of strings into normalized ``float32`` vectors.

        Flow:
        - sentence-transformers path: call the loaded model, then normalize;
        - fallback path: hash tokens into a fixed-size vector, then normalize;
        - both paths return rows that can be compared with inner product.
        """

        if self._model is not None:
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return _normalize(np.asarray(vectors, dtype=np.float32))
        return _hashing_encode(texts, self.fallback_dimensions)

    def _load_backend(self) -> EmbeddingInfo:
        """Choose the embedding backend at startup.

        Decision order:
        - if config explicitly asks for hashing, return hashing immediately;
        - if optional packages look incompatible, skip the import;
        - otherwise try sentence-transformers with local files by default;
        - on any failure, return hashing plus the failure type in metadata.
        """

        if self.requested_backend in {"hashing", "sparse", "fallback"}:
            return EmbeddingInfo("hashing", "generic-token-hashing", self.fallback_dimensions)
        try:
            incompatibility = _sentence_transformer_incompatibility()
            if incompatibility:
                raise RuntimeError(incompatibility)
            from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel,import-error

            allow_download = os.getenv("ME_AGENT_ALLOW_EMBEDDING_DOWNLOAD", "0") == "1"
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=not allow_download,
                )
            except TypeError:
                if not allow_download:
                    raise
                self._model = SentenceTransformer(self.model_name)
            dimension = int(self._model.get_sentence_embedding_dimension())
            return EmbeddingInfo("sentence-transformers", self.model_name, dimension)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._model = None
            return EmbeddingInfo(
                "hashing",
                "generic-token-hashing",
                self.fallback_dimensions,
                fallback_reason=type(exc).__name__,
            )


def tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens used by hashing and scoring."""

    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _hashing_encode(texts: list[str], dimensions: int) -> np.ndarray:
    """Encode text without downloading a model.

    For each token:
    - hash the token with BLAKE2b;
    - use part of the digest to choose a vector index;
    - use another digest byte to add either ``+1`` or ``-1``;
    - normalize the final row so dot product behaves like cosine similarity.
    """

    vectors = np.zeros((len(texts), dimensions), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vectors[row, index] += sign
    return _normalize(vectors)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector row to length 1 while keeping zero rows as zero."""

    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32)


def _sentence_transformer_incompatibility() -> str:
    """Detect optional package states that would make model import noisy."""

    try:
        transformers_version = metadata.version("transformers")
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        return "missing-optional-embedding-package"
    if _version_at_least(transformers_version, 5, 0) and not _version_at_least(torch_version, 2, 4):
        return "transformers-requires-newer-torch"
    return ""


def _version_at_least(version: str, major: int, minor: int) -> bool:
    """Return whether ``version`` has at least the requested major/minor."""

    parts = re.findall(r"\d+", version)[:2]
    if len(parts) < 2:
        return False
    return (int(parts[0]), int(parts[1])) >= (major, minor)
