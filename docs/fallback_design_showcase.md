# Fallback Design Showcase

Concise summary of how the system behaves when context, optional dependencies,
provider access, or confidence is missing.

| Fallback | Trigger | Behavior | Tradeoff |
| --- | --- | --- | --- |
| Out of scope | Router returns `general`. | Skip retrieval and LLM, return scoped answer. | May reject unknown ECU-like phrasing if router is too strict. |
| Embedding hashing | Sentence-transformers unavailable or disabled. | Convert tokens to deterministic vectors. | Runnable offline, but not true semantic embeddings. |
| Numpy vector search | FAISS missing or source filtering needed. | Use normalized matrix multiplication. | Portable for small corpus, not large-scale indexing. |
| Retrieval broadening | First pass confidence is low. | Relax source constraints once. | Recovers narrow routing, but may add noise. |
| Extractive generation | Missing key, import failure, provider error, or empty response. | Rank retrieved facts and cite sources. | Less fluent than LLM output. |
| Verification fallback | Answer introduces unsupported numeric specs. | Do not silently trust the answer. | Numeric-focused, not full semantic proof. |
| Human review | Confidence/support is low. | Mark response for review. | Flag exists; production queue is future work. |
