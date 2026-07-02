# Scalability Strategy

The first corpus is intentionally small, so an in-memory fallback retriever is
sufficient. Scaling to thousands of manuals should add:

- Durable document indexing and versioned metadata.
- Incremental re-indexing for changed manuals.
- Embedding-backed vector retrieval with deterministic keyword fallback.
- Source-level filters by product family, model, document version, and section.
- Offline evaluation gates before publishing a new index.
- Production monitoring for latency, retrieval misses, source coverage, and
  human-review rates.
