# ME Engineering Assistant

ME Engineering Assistant is a Python package for answering engineering questions
over ECU product manuals with a grounded RAG workflow. The target production
shape is LangChain plus LangGraph orchestration, deterministic routing across
ECU product families, an in-memory retriever, MLflow model packaging, and
Databricks Asset Bundle deployment.

This repository now contains the local challenge implementation: Markdown
manual loading, metadata-preserving chunking, deterministic routing, switchable
keyword/vector/hybrid in-memory retrieval, a real LangGraph workflow,
DeepSeek/OpenAI-compatible generation with deterministic grounded fallback,
evaluation artifacts, and MLflow pyfunc packaging.

## Architecture

The package is organized around small modules with narrow responsibilities:

- `config`: runtime paths, retrieval parameters, and DeepSeek client settings.
- `schemas`: dataclasses used across the agent.
- `data_loader`: Markdown manual discovery and metadata extraction.
- `chunking`: deterministic preprocessing into standardized chunks with metadata preservation.
- `embeddings`: local normalized embedding wrapper used only by vector and hybrid retrieval.
- `vector_store`: in-memory vector index built once from chunks at startup.
- `retriever`: switchable keyword, vector, and hybrid retrievers that consume preprocessed chunks.
- `router`: deterministic query routing across ECU-700 and ECU-800 sources.
- `graph`: LangGraph workflow entry point and structured response assembly.
- `verifier`, `confidence`, `hitl`: answer support, heuristic confidence, and
  human-review decisions.
- `evaluation`: CSV evaluation case loading and simple evaluation helpers.
- `mlflow_model`: custom `mlflow.pyfunc.PythonModel` wrapper.

See `docs/architecture.md` for the current design and expansion plan.

## Setup

```bash
pip install -e ".[dev]"
```

## Model Configuration

The default model configuration targets DeepSeek V4 Flash:

```text
ME_AGENT_MODEL_NAME=deepseek-v4-flash
ME_AGENT_OPENAI_BASE_URL=https://api.deepseek.com
ME_AGENT_ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ME_AGENT_API_KEY_ENV_VAR=DEEPSEEK_API_KEY
ME_AGENT_RETRIEVER_MODE=hybrid  # keyword | vector | hybrid
ME_AGENT_KEYWORD_WEIGHT=0.5
ME_AGENT_VECTOR_WEIGHT=0.5
ME_AGENT_MODEL_TIMEOUT_SECONDS=8
ME_AGENT_MODEL_MAX_RETRIES=0
DEEPSEEK_API_KEY=
```

Do not commit API keys. Add `DEEPSEEK_API_KEY` locally through your shell,
`.env`, Databricks secrets, or the deployment environment. Local MLflow scripts
default to `sqlite:///mlflow.db` unless `MLFLOW_TRACKING_URI` is already set.

## Usage

```bash
python scripts/run_agent.py "How much RAM does the ECU-850 have?"
python scripts/run_eval.py
ME_AGENT_RETRIEVER_MODE=vector python scripts/run_eval.py
ME_AGENT_RETRIEVER_MODE=keyword python scripts/run_eval.py
```


## Databricks Asset Bundle

This repo includes `databricks.yml` with a `me-agent-log-and-validate` job. The
job builds the wheel, logs the MLflow pyfunc model, and runs the golden
evaluation task from package entry points.

```bash
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle run me_agent_log_and_validate -t dev
```

Set `DEEPSEEK_API_KEY` or the equivalent Databricks secret-backed environment
variable in the target workspace before running live model evaluation.

## Validation

Current validation commands:

```bash
python -m compileall src tests scripts
pytest -q
pylint src/me_agent
python scripts/run_eval.py
python scripts/log_mlflow_model.py
python scripts/load_mlflow_model.py
```

The latest live DeepSeek Flash evaluation using the default `hybrid` retriever
passed 10/10 with `accuracy: 1.0`, `used_llm_rate: 1.0`, average latency
2.8850 seconds, and max latency 6.9124 seconds. `eval_results.json` and the CLI output include each question,
expected answer, agent-produced answer, sources, route, key facts, and latency.

## Limitations

The vector retriever is a local in-memory sparse embedding index, not a hosted
neural embedding model backed by FAISS or Chroma. The deterministic fallback and
post-generation stabilizer are intentionally tuned for the provided ECU manuals
and fixed challenge evaluation set. Databricks Asset Bundle packaging is included, but it still requires a configured
Databricks workspace, cluster policy compatibility, and workspace secrets before
running in your target environment. MLflow currently uses local SQLite-backed
tracking for local smoke tests and emits the standard CloudPickle warning for
pyfunc object serialization.
