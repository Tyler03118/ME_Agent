# System Design Showcase

ME Agent is a source-aware ECU manual RAG assistant with explicit control over
query scope, source routing, evidence retrieval, grounded generation,
verification, confidence, and evaluation.


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

## Retrieval Design

| Mode | How it works | Best for |
| --- | --- | --- |
| Keyword | Exact token overlap. | Model IDs, commands, units, numeric specs. |
| Vector | Embedding similarity with FAISS/numpy. | Paraphrased questions. |
| Hybrid | Keyword first, vector recall second. | Current default for balanced precision and recall. |

## Why Deterministic Routing

- Small domain: ECU-700/800 manual signals are explicit.
- Lower cost and latency than LLM routing.
- Easier to unit test.
- Safer control flow: prompt text cannot change the source policy.

## Tier Coverage

| Tier | Evidence in project |
| --- | --- |
| Tier 1 | Multi-source ECU RAG, LangGraph workflow, intelligent routing, MLflow pyfunc, 10/10 live eval. |
| Tier 2 | Installable package, tests, fallback/error handling, monitoring fields, model metadata. |
| Tier 3 | Custom evaluator, stress set, JSON/HTML reports, MLflow logging, human-review flag, scalability plan. |
