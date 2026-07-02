from me_agent.embeddings import EmbeddingModel


def test_embedding_model_encodes_and_normalizes_vectors() -> None:
    model = EmbeddingModel()

    vectors = model.encode(["AI accelerator", "Neural Processing Unit"])

    assert len(vectors) == 2
    assert all(vector for vector in vectors)
    for vector in vectors:
        norm = sum(value * value for value in vector.values()) ** 0.5
        assert round(norm, 6) == 1.0
