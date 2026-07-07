# System Design Showcase

The core design principle is evidence-path control. In an industrial setting,
wrong specs can lead to wrong engineering decisions, so the system prioritizes
grounded evidence, traceable sources, and conservative behavior when support is
weak.


## Runtime Graph

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	validate_input(validate_input)
	route_query(route_query)
	out_of_scope(out_of_scope)
	retrieve_context(retrieve_context)
	broaden_retrieve(broaden_retrieve)
	generate_answer(generate_answer)
	verify_answer(verify_answer)
	compute_confidence(compute_confidence)
	finalize_response(finalize_response)
	human_review(human_review)
	__end__([<p>__end__</p>]):::last
	__start__ --> validate_input;
	broaden_retrieve --> generate_answer;
	compute_confidence -. &nbsp;finalize&nbsp; .-> finalize_response;
	compute_confidence -.-> human_review;
	generate_answer --> verify_answer;
	retrieve_context -. &nbsp;broaden&nbsp; .-> broaden_retrieve;
	retrieve_context -. &nbsp;generate&nbsp; .-> generate_answer;
	route_query -.-> out_of_scope;
	route_query -. &nbsp;retrieve&nbsp; .-> retrieve_context;
	validate_input --> route_query;
	verify_answer --> compute_confidence;
	finalize_response --> __end__;
	human_review --> __end__;
	out_of_scope --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Layer Map

| Layer | Code area | What it does | Why it matters |
| --- | --- | --- | --- |
| Config/schema | `core/` | Runtime settings and shared response types. | Keeps graph, eval, and MLflow using the same contracts. |
| Ingestion | `ingestion/` | Load Markdown manuals and create stable chunks. | Makes evidence deterministic and citable. |
| Routing | `workflow/router.py` | Classify query scope, route category, and source policy. | Prevents the LLM from controlling retrieval scope. |
| Retrieval | `retrieval/` | Keyword, vector, and hybrid evidence search. | Finds manual chunks before generation. |
| Generation | `generation/llm.py` | DeepSeek answer or deterministic fallback. | Keeps no-key and provider-failure paths usable. |
| Verification | `generation/verifier.py` | Numeric grounding and support checks. | Catches high-risk ECU spec hallucinations. |
| Workflow | `workflow/graph.py` | LangGraph orchestration and branches. | Makes the runtime path explicit and testable. |
| Evaluation | `evaluation/` | Score answers and write JSON/HTML reports. | Turns quality into reviewable artifacts. |
| Tracking | `tracking/` | MLflow pyfunc packaging and metadata. | Reproduces code + config + corpus + eval context. |


## Ingestion

Ingestion creates stable, source-aware evidence units before retrieval. The
loader extracts source, document ID, product family, and model metadata; the
chunker creates overlapping chunks that preserve that metadata.

```mermaid
flowchart TD
    A["Markdown manuals"] --> B["MarkdownManualLoader"]
    B --> C["ManualDocument<br/>content + metadata"]
    C --> D["MarkdownChunker"]
    D --> E["ManualChunk<br/>content + source metadata"]
    E --> F["Retrievers build indexes"]
```


## Routing

The router is the first control-flow decision point. It does not answer the
question; it decides whether the query is inside the ECU manual scope, which
route category applies, and which manuals retrieval should prioritize.

| Signal | Routing effect |
| --- | --- |
| `ECU-750` / `ECU-700` | Prioritize `ECU-700_Series_Manual.md`. |
| `ECU-850` / `ECU-800` | Prioritize `ECU-800_Series_Base.md`. |
| `ECU-850b` | Include `ECU-800_Series_Plus.md`; base ECU-800 context may also apply. |
| OTA / firmware update | Route to `feature_availability` and check all manuals. |
| Compare / best / strongest / harshest / which model | Route to `comparison` and expand to cross-model evidence. |
| NPU / AI / edge inference / accelerator | Route to `ecu_850b_lookup`, because those details live in the plus addendum. |
| No ECU manual signal | Return an out-of-scope route instead of using general model knowledge. |

```mermaid
flowchart TD
    A["User query"] --> B["Normalize text"]
    B --> C{"Inside ECU manual scope?"}
    C -- "No" --> D["general route<br/>no sources<br/>out-of-scope response"]
    C -- "Yes" --> E["Infer required sources"]

    E --> F{"Specific intent?"}
    F -- "Driver command / enable NPU" --> G["configuration<br/>ECU-800 Plus"]
    F -- "OTA / firmware update" --> H["feature_availability<br/>all manuals"]
    F -- "Compare / best / strongest" --> I["comparison<br/>relevant manuals or all manuals"]
    F -- "AI / NPU / accelerator" --> J["ecu_850b_lookup<br/>ECU-800 Plus"]
    F -- "Model-only lookup" --> K["ecu_700_lookup / ecu_800_lookup / ecu_850b_lookup"]

    G --> L["RouteDecision(category, sources, rationale)"]
    H --> L
    I --> L
    J --> L
    K --> L
    L --> M["Retrieval prioritizes routed manuals"]
```

Routing is deterministic rather than LLM-based because the domain is small and
the routing signals are explicit: model names, feature names, technical specs,
and comparison language. This keeps routing low-latency, low-cost,
reproducible, and easy to unit test.

It also protects the retrieval policy. The LLM is used for grounded synthesis,
but it does not decide which manuals are valid sources. Prompt text such as
"ignore the manuals" cannot change the source policy.

## Retrieval Design
Retrieval starts from the route decision. The router decides the required source
policy, while the retriever decides which chunks inside that policy are the best
evidence for generation.

```mermaid
flowchart TD
    A["Question + RouteDecision"] --> B["Required source policy"]
    B --> C["Keyword retriever"]
    B --> D["Vector retriever"]

    C --> C1["Exact-match strength<br/>model IDs, units, commands"]

    D --> E["EmbeddingModel"]
    E --> E1{"sentence-transformers available?"}
    E1 -- "Yes" --> E2["SentenceTransformer embeddings"]
    E1 -- "No" --> E3["Deterministic hashing embeddings"]

    E2 --> F["VectorStore"]
    E3 --> F
    F --> F1{"FAISS available?"}
    F1 -- "Yes" --> F2["FAISS similarity search"]
    F1 -- "No" --> F3["Numpy similarity search"]

    C1 --> G["Hybrid merge"]
    F2 --> G
    F3 --> G
    G --> H["Deduplicate chunks"]
    H --> I["Preserve required source coverage"]
    I --> J["Top evidence chunks"]
    J --> K["Generation"]
```

Keyword retrieval protects exact engineering facts such as model IDs, numeric
specs, units, and driver commands. Vector retrieval improves recall for
paraphrased questions by embedding chunks once and searching them through FAISS
when available, with numpy similarity as the local fallback. If the semantic
embedding stack is unavailable, deterministic hashing keeps offline runs
reproducible, but it is treated as a fallback rather than production semantic
retrieval.

## Generation

`generation/llm.py` turns retrieved evidence into the final answer. It does not
decide routing or retrieval scope; it only synthesizes from the chunks already
selected by the workflow.

```mermaid
flowchart TD
    A["Question + route + retrieved chunks"] --> B{"Any retrieved context?"}
    B -- "No" --> C["Insufficient-context response<br/>fallback_reason=no_context"]
    B -- "Yes" --> D["Build grounded prompt"]

    D --> E["Evidence lines<br/>highest-signal facts first"]
    D --> F["Full context<br/>source-labeled chunks"]
    E --> G{"Live LLM available?"}
    F --> G

    G -- "Yes" --> H["DeepSeek synthesis<br/>answer only from context"]
    G -- "No / provider error" --> I["Deterministic fallback<br/>rank facts from retrieved chunks"]

    H --> J["GenerationResult<br/>answer, used_llm, fallback_reason"]
    I --> J
```

The prompt puts curated `EVIDENCE_LINES` before full `CONTEXT` so the model sees
exact specification rows and high-signal facts first. Comparison and
feature-availability routes get comparison-oriented instructions; lookup routes
prefer exact values from evidence lines and tables.

If the live model is unavailable, deterministic fallback ranks facts by question
overlap, domain-specific matches such as thermal tolerance to operating
temperature, and retrieval score. This keeps the assistant usable without the
provider while preserving source-grounded answers.

## Verification

`generation/verifier.py` checks whether generated answers are grounded in the
retrieved evidence before confidence and final response handling. It focuses on
high-risk ECU specifications such as temperature, memory, frequency, current,
TOPS, and Mbps.

```mermaid
flowchart TD
    A["Generated answer"] --> B["Extract answer measurements"]
    C["Retrieved context"] --> D["Extract evidence measurements"]

    B --> E{"Answer measurements supported?"}
    D --> E

    E -- "No" --> F["contradicted"]
    E -- "Yes" --> G["Token overlap support check"]

    G --> H{"Support level?"}
    H -- "High" --> I["supported"]
    H -- "Partial" --> J["partially_supported"]
    H -- "Low" --> K["unsupported"]

    F --> L["Confidence calculation"]
    I --> L
    J --> L
    K --> L
```

The verifier is intentionally narrow. It is not a universal fact checker; it is
a practical grounding layer for the engineering facts most likely to cause harm
if hallucinated.

## Tier Coverage

| Tier | Evidence in project |
| --- | --- |
| Tier 1 | Multi-source ECU RAG, LangGraph workflow, intelligent routing, MLflow pyfunc, 10/10 live eval. |
| Tier 2 | Installable package, tests, fallback/error handling, monitoring fields, model metadata. |
| Tier 3 | Custom evaluator, stress set, JSON/HTML reports, MLflow logging, human-review flag, scalability plan. |
