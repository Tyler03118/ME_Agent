# Scalability Strategy

The current corpus is small, so the default implementation keeps retrieval local
and in memory. The architecture still separates preprocessing, embedding, vector
indexing, retrieval, generation, and evaluation so each layer can scale later.

Planned scaling steps:

- Persist chunk metadata and embeddings outside process memory.
- Move FAISS or equivalent vector indexing to a managed service when corpus size grows.
- Add reranking after initial retrieval if precision becomes a bottleneck.
- Track retrieval misses, latency, fallback rate, and human-review rate in production metrics.
- Add a larger evaluation set with regression buckets for comparisons, missing context, and conflicting documents.
