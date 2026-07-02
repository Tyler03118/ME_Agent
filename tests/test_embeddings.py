import numpy as np

from me_agent.retrieval.embeddings import EmbeddingModel, tokenize


def test_embedding_model_encodes_and_normalizes_vectors() -> None:
    model = EmbeddingModel(backend="hashing")

    vectors = model.encode(["AI accelerator", "Neural Processing Unit"])

    assert vectors.shape[0] == 2
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0)


def test_tokenizer_does_not_add_corpus_specific_synonyms() -> None:
    assert tokenize("AI accelerator memory over air") == ["ai", "accelerator", "memory", "over", "air"]
