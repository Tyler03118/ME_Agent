# ME Engineering Assistant

ME Engineering Assistant is an installable Python 3.11 package for answering
questions over ECU product manuals with a grounded RAG workflow. It uses
LangChain-compatible DeepSeek generation, LangGraph workflow orchestration,
keyword/vector/hybrid retrieval, honest evaluation, and MLflow pyfunc packaging.

The implementation intentionally stays as a single-agent RAG system. It has
agent-like control flow through conditional routing, one retrieval retry, and a
human-review branch, without introducing unnecessary multi-agent complexity.

Routing is deterministic by design. An LLM router would be more flexible, but for
this small ECU corpus the regex/keyword router is cheaper, lower-latency, easier
to test, and less likely to misroute exact model/specification questions.

## Architecture

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

Package layout:

- `me_agent.core`: runtime configuration and shared dataclass schemas.
- `me_agent.ingestion`: Markdown loading and deterministic chunk preprocessing.
- `me_agent.retrieval`: embeddings, FAISS/numpy vector store, and keyword/vector/hybrid retrievers.
- `me_agent.generation`: DeepSeek generation, grounded fallback, prompt construction, and answer verification.
- `me_agent.workflow`: LangGraph workflow, routing, confidence, and human-review branching.
- `me_agent.evaluation`: Expected_Answer similarity, token coverage, latency, and MLflow eval logging.
- `me_agent.tracking`: MLflow pyfunc model and reusable model logging helpers.
- `me_agent.cli`: console entry points declared in `pyproject.toml`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import me_agent"
```

## Model Configuration

Default live generation uses DeepSeek V4 Flash through the OpenAI-compatible API.
Without `DEEPSEEK_API_KEY`, the system falls back to generic extractive synthesis
from retrieved context. The fallback is intentionally not tuned to the evaluation
questions.

```text
ME_AGENT_MODEL_NAME=deepseek-v4-flash
ME_AGENT_OPENAI_BASE_URL=https://api.deepseek.com
ME_AGENT_API_KEY_ENV_VAR=DEEPSEEK_API_KEY
ME_AGENT_RETRIEVER_MODE=hybrid  # keyword | vector | hybrid
ME_AGENT_EMBEDDING_BACKEND=auto # auto | sentence-transformers | hashing
ME_AGENT_ALLOW_EMBEDDING_DOWNLOAD=0
DEEPSEEK_API_KEY=
```

`auto` attempts sentence-transformers first and gracefully falls back to generic
hashing if the package/model is unavailable. To allow the first run to download
`sentence-transformers/all-MiniLM-L6-v2` (roughly 80 MB), set
`ME_AGENT_ALLOW_EMBEDDING_DOWNLOAD=1`. Offline CI can force
`ME_AGENT_EMBEDDING_BACKEND=hashing`.

## Usage

After `pip install -e ".[dev]"`, the package exposes console entry points:

```bash
me-agent "How much RAM does the ECU-850 have?"
me-agent "How's the weather today?"
ME_AGENT_RETRIEVER_MODE=keyword me-agent-run-eval
ME_AGENT_RETRIEVER_MODE=vector me-agent-run-eval
ME_AGENT_RETRIEVER_MODE=hybrid me-agent-run-eval
me-agent-render-eval-report reports/eval_results.json reports/eval_report.html
me-agent-log-model
me-agent-load-model
```

The `scripts/*.py` files expose the same workflows for local development. If the
package has not been installed yet, run them with `PYTHONPATH=src`.

## Evaluation

Evaluation compares each golden answer to `Expected_Answer` using semantic
similarity, expected-token coverage, and inferred technical fact recall. Stress
cases can additionally provide `Required_Facts`, `Forbidden_Facts`,
`Expected_Sources`, and `Expected_Route`, which enable deterministic fact
recall, forbidden-fact violation, source-match, and route-match metrics.

`python scripts/run_eval.py` starts an MLflow run, logs metrics, logs the JSON
result artifact, and can optionally produce a self-contained visual HTML report.
Key metrics include accuracy, latency, `used_llm_rate`, fallback count, mean
semantic similarity, mean token coverage, mean required-fact recall, mean source
match, mean route match, and forbidden fact violations.

The provided challenge set lives at `data/eval/test-questions.csv`. The custom
stress set lives at `data/eval/stress-questions.csv` and can be run with:

```bash
ME_AGENT_EVAL_PATH=data/eval/stress-questions.csv python scripts/run_eval.py
python scripts/run_eval.py \
  --eval-path data/eval/stress-questions.csv \
  --output reports/stress_eval_results.json \
  --html-report reports/stress_eval_report.html \
  --title "ME Agent Stress Evaluation"
```

Existing JSON artifacts can be re-rendered without rerunning the model:

```bash
python scripts/render_eval_report.py \
  reports/eval_results.json \
  reports/eval_report.html
```

Representative live results from the latest approved DeepSeek run on 2026-07-05:

| Mode | Accuracy | used_llm_rate | Avg latency | Max latency | Mean similarity | Mean token coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Live DeepSeek (`deepseek-v4-flash`) | 100% | 100% | 2.5614s | 4.1594s | 0.7049 | 0.7176 |
| Live DeepSeek stress set | 100% | 79% | 2.1731s | 4.0309s | 0.6901 | 0.8279 |

Offline no-key evaluation is a smoke/degradation check for the generic
extractive fallback path, not the main challenge scoring path. Read its current
result from `reports/eval_results_offline.json` after running the validation script. The
stress run includes two intentional out-of-scope cases and one verifier-triggered
grounded fallback for prompt-injection defense while still passing
fact/source/route checks.

## Tier Coverage

- **Tier 1:** Multi-source ECU-700/ECU-800 RAG, deterministic intelligent routing,
  cross-document comparison retrieval, LangGraph control flow, MLflow pyfunc
  packaging, and 10/10 live evaluation under the 10-second target in the latest
  approved DeepSeek run.
- **Tier 2:** Installable Python package, modular source layout, unit tests,
  validation commands, MLflow model artifacts, logged configuration metadata,
  and documented performance/error-handling strategy.
- **Tier 3:** Custom evaluation framework with MLflow metric/artifact logging,
  deterministic stress metrics, a custom adversarial stress set, visual HTML eval
  reports, low-confidence human-review flagging, and a concrete scalability
  strategy in `docs/scalability_strategy.md`.

## Validation

Recommended checks from a fresh clone:

```bash
./scripts/run_all_checks.sh
```

This script creates a local `.venv`, installs the package with development
dependencies, runs import checks, unit tests, pylint, compile checks, and an
offline evaluation smoke test. To include live DeepSeek evaluation and regenerate
the JSON/HTML reports, provide a temporary API key and opt in explicitly:

```bash
export DEEPSEEK_API_KEY="<temporary-key>"
RUN_LIVE_EVAL=1 ./scripts/run_all_checks.sh
```

The equivalent manual commands are:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m compileall src tests scripts
pytest -q
pylint src/me_agent
me-agent-run-eval
me-agent-log-model
me-agent-load-model
```

Equivalent script commands also work after installation, for example
`python scripts/run_eval.py`. Without installing the package, prefix script
commands with `PYTHONPATH=src` so Python can import the `src/me_agent` package.

Current tests cover importability without hard dependency on `.env`, generic
fallback behavior, out-of-domain routing, conditional graph branches, retrieval
modes, vector index build-once behavior, generic evaluation fields, and MLflow
model input handling.

## Error Handling Evidence

- Missing API key: `DeepSeekAnswerGenerator` falls back to grounded extractive
  synthesis with `fallback_reason="missing_deepseek_client_or_key"`; offline
  evaluation records this path separately from live model generation.
- Empty query: `validate_input` rejects blank questions with `ValueError` and is
  covered by unit tests.
- Out-of-scope query: non-ECU questions route to `general`, skip retrieval, and
  return a direct scope response.
- Embedding fallback: `EmbeddingModel(backend="auto")` tries sentence-transformers
  and falls back to normalized token hashing when local model dependencies are
  missing or incompatible.
- Vector fallback: `VectorStore` uses FAISS when available and numpy inner-product
  search when FAISS is unavailable or source filtering is required.
- Low confidence: the LangGraph workflow computes confidence from retrieval,
  source coverage, and verifier support, then flags borderline answers for
  human review instead of silently overclaiming.

## Limitations

- No-key fallback is extractive and may be less polished than live DeepSeek output.
- Sentence-transformers may need a first-run model download unless hashing fallback is used.
- FAISS is preferred, but numpy similarity fallback keeps offline runs from hard-crashing.
- Reranking is future work; hybrid retrieval currently merges keyword-first and vector results.
- Chunking is deterministic character-based splitting today; Markdown-aware table/section splitting is a planned improvement for larger manuals.
- Live evaluation quality and latency depend on provider/network availability.
