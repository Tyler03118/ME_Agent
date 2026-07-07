# Scalability Plan Showcase

Concise visual summary of current limitations and the production roadmap.

## Core Principle

The current system is a strong challenge prototype, not a finished production
RAG platform. The important strength is that the boundaries are already clean:
ingestion, routing, retrieval, generation, verification, evaluation, tracking,
and packaging are separate.

The scalable direction is not to add a bigger LLM first. The scalable direction
is to make the evidence path stronger:

```mermaid
flowchart LR
    A["Better chunks + metadata"] --> B["Better source selection"]
    B --> C["Better retrieval"]
    C --> D["Better grounded generation"]
    D --> E["Better verification + evaluation"]
```

## Target Evidence Path

```mermaid
flowchart LR
    Q["User query"] --> R["Deterministic router"]
    R --> S["Source candidate selection"]
    S --> T["Optional query rewrite / multi-query"]
    T --> H["Hybrid retrieval"]
    H --> F["Fusion + reranking"]
    F --> C["Context compression + ordering"]
    C --> G["Grounded generation"]
    G --> V["Claim-level verification"]
    V --> O["Answer + citations + diagnostics"]
```

## Focus Areas

### 1. Ingestion And Metadata

| Current limitation | Scalable upgrade | Why it matters |
| --- | --- | --- |
| Current chunking is deterministic and simple. It works for a small Markdown corpus, but it is not fully document-aware or semantic-aware. | Move toward structure-aware and semantic chunking. For example, preserve Markdown headings, tables, table rows, section paths, model names, and stable chunk IDs. | Better chunks solve three problems: LLM context-window limits, retrieval signal-to-noise ratio, and generation cost/latency. |
| Current metadata is basic. | Add richer metadata such as `model`, `series`, `section`, `spec_type`, `unit`, `version`, and `chunk_id`. | Metadata enables source filtering, comparison coverage checks, precise citations, and better evaluation. |

### 2. Router And Retrieval

| Current limitation | Scalable upgrade | Why it matters |
| --- | --- | --- |
| The router is deterministic and explainable, but simple rules can over-trigger or miss paraphrases. | Keep the deterministic router as the control gate, but harden it with word-boundary matching, co-occurrence rules, and a declarative rule table. | Routing should stay stable and inspectable because it controls scope and source constraints. |
| `ALL_SOURCES` works for three manuals, but it will not scale to hundreds of manuals. | Add source candidate selection by model, series, feature, document version, and metadata before chunk retrieval. | The system should narrow the search space before retrieving chunks. |
| Current hybrid retrieval is a prototype-level merge of keyword and vector results. | Use BM25 for exact terms, dense retrieval for semantic recall, RRF for rank fusion, and a reranker for final ordering. | This balances exact engineering terms like `ECU-850b` and `5 TOPS` with paraphrased user questions. |
| Some user questions are vague or use different wording from the manuals. | Add optional query rewriting or multi-query retrieval. HyDE can be considered later if normal query rewriting is not enough. | Query transformation improves recall, but it should support retrieval rather than replace deterministic routing. |
| Current vector search is local and prototype-oriented. | Use a production vector database only when data size, metadata filtering, persistence, or concurrency require it. Local `FAISS` is enough for prototypes; `Milvus`, `Weaviate`, or `Pinecone` fits larger production workloads. | Vector DB is a scaling tool, not the first thing to optimize in a tiny corpus. |

### 3. Generation And Verification

| Current limitation | Scalable upgrade | Why it matters |
| --- | --- | --- |
| Current generation uses retrieved chunks directly in the prompt. | Add context compression and context ordering before generation. Put the most relevant evidence near the beginning or end of the prompt. | This reduces noise, lowers cost, and helps with the "lost in the middle" problem. |
| Current output is grounded text with source filenames. | Move toward structured output with answer text, cited evidence IDs, unsupported-claim slots, and diagnostics. | Structured answers are easier to verify, debug, and evaluate. |
| Current verifier checks numeric grounding and token support. | Split the answer into claims and verify each claim against cited evidence. Add citation correctness checks over time. | A mostly correct answer can still contain one unsupported engineering spec, so verification should happen at the claim level. |

### 4. Evaluation And Operations

| Current limitation | Scalable upgrade | Why it matters |
| --- | --- | --- |
| Weighted scores can hide specific failures. | Promote required facts, forbidden facts, expected source, expected route, and numeric contradiction checks into hard gates by case type. | A high average score should not hide a wrong source or unsupported numeric spec. |
| Current eval set is useful but small. | Expand by category: single-source lookup, cross-source comparison, feature availability, negative evidence, configuration, paraphrase, out-of-scope, and adversarial cases. | Category coverage is more useful than adding random questions. |
| Current metrics cover the full workflow, but retrieval and generation can be evaluated more directly. | Add RAG metrics such as context precision, context recall, faithfulness, answer relevancy, claim support rate, and citation correctness. Tools like RAGAS or TruLens can help. | These metrics separate retrieval quality from generation quality, which makes failures easier to debug. |
| Current reports include JSON, HTML, and MLflow artifacts. | Track route mix, fallback rate, retrieval confidence, verifier status, latency, eval trend, corpus hash, index version, prompt version, and package versions. | Production debugging needs versioned evidence and trend signals, not just one report. |

## Priority Roadmap

| Priority | Focus | Concrete upgrades |
| --- | --- | --- |
| P0 | Evidence correctness | Structure-aware chunks, richer metadata, router hardening, and hard-gate evaluation. |
| P1 | Retrieval quality | Source candidate selection, BM25, dense retrieval, RRF, reranking, and optional query rewriting or multi-query retrieval. |
| P2 | Production readiness | Context compression, claim-level verification, RAG metrics, vector DB if scale requires it, versioning, and observability. |