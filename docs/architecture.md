# Architecture

ME Engineering Assistant is organized as a package-first RAG system over ECU
Markdown manuals.

## Data Flow

1. `MarkdownManualLoader` loads every manual into `ManualDocument` records with
   source, product family, and model metadata.
2. `chunk_documents()` performs preprocessing into standardized chunk dictionaries
   with text, source, and stable chunk ids; `MarkdownChunker` adapts those chunks
   into `ManualChunk` records used by the graph.
3. For vector and hybrid modes only, `EmbeddingModel` encodes chunks once and
   `VectorStore` builds an in-memory index during retriever initialization.
4. `DeterministicRouter` classifies the question and chooses required source
   coverage for ECU-700, ECU-800 base, ECU-800 plus, or cross-document questions.
5. The configured retriever mode consumes preprocessed chunks and returns grounded context.
6. The LangGraph workflow generates, verifies, scores confidence, and flags
   human review when confidence is low.
7. `MEEngineeringAssistantModel` exposes the same pipeline through MLflow
   `pyfunc.predict()`.

## Retriever Modes

The retriever is selected with `ME_AGENT_RETRIEVER_MODE`:

- `keyword`: deterministic token-overlap retrieval. Best for exact engineering
  identifiers, commands, part numbers, and numeric specs.
- `vector`: local in-memory sparse embedding retrieval with cosine similarity
  and small ECU-domain query expansion. Best for paraphrases such as "AI
  accelerator" matching "Neural Processing Unit".
- `hybrid`: production default. Calls keyword retrieval, calls vector retrieval,
  then merges keyword-first results with vector recall and deduplicates chunks.
  It does not rerank or call an LLM.

All modes implement the same `retrieve(query, required_sources, top_k)` contract,
so LangGraph generation, verification, confidence, HITL, MLflow, and evaluation
remain mode-agnostic.


## Layer Boundaries

- Preprocessing: `src/me_agent/chunking.py` owns chunk creation only.
- Representation: `src/me_agent/embeddings.py` owns local embedding creation only.
- Indexing: `src/me_agent/vector_store.py` owns vector indexing and search only.
- Retrieval: `src/me_agent/retriever.py` owns mode selection and result merging only.

Keyword mode does not initialize the embedding layer. Vector indexes are built
once when the retriever is constructed, not recomputed per query. LangGraph
continues to depend only on `MarkdownChunker` and `build_retriever()`.

## Source Coverage

Routing still controls document coverage. Comparison and feature-availability
questions pass required sources into the retriever so results include at least
one chunk from every required manual before filling remaining slots by score.
This prevents cross-document questions from collapsing to a single high-scoring
manual.

## Model Configuration

The default LLM provider is DeepSeek V4 Flash through the OpenAI-compatible
endpoint `https://api.deepseek.com`. API keys are loaded from `DEEPSEEK_API_KEY`
via the local `.env` file or runtime environment. The model client defaults to
an 8 second timeout and zero client retries so provider timeouts do not exceed
the challenge latency target.

## Evaluation

Evaluation stores aggregate metrics and per-question details including expected
answer, agent answer, route, sources, key facts, missing facts, retriever mode,
LLM usage, fallback reason, confidence, and latency.


## Databricks Bundle

`databricks.yml` defines a package-first bundle job that builds the wheel, logs
the MLflow pyfunc model, and runs the golden evaluation from wheel entry points.
The logged MLflow model includes the manuals and evaluation CSV as artifacts so
model loading does not depend on the original repository checkout.
