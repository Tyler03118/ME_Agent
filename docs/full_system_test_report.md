# ME Engineering Assistant Test Report

Date: 2026-07-02
Repository: /Users/ziji/Dev/ME_Agent
Branch: master

## Current Status

The project now implements an honest, installable single-agent RAG system for ECU
manual QA. The previous hardcoded answer paths and question-id-specific
evaluation criteria were removed. The system now has conditional LangGraph
branches, DeepSeek live generation, generic extractive fallback, FAISS/vector
retrieval with offline fallback, MLflow metrics/artifact logging, and pyfunc
packaging with signature and input example.

## Validation Results

- `python3 -m pip install -e ".[dev]"`: passed in the active environment.
- Clean venv verification: `pip install -e ".[dev]"` and `python -c "import me_agent"` passed in `/private/tmp/me_agent_clean_venv`.
- `python3 -m compileall src tests scripts`: passed.
- `python3 -m pytest -q`: 44 passed, 1 MLflow type-hint warning.
- `python3 -m pylint src/me_agent`: 10.00/10.
- `python3 -m build --wheel`: built `dist/me_agent-0.1.0-py3-none-any.whl`.
- Out-of-domain query (`python scripts/run_agent.py "今天天气如何"`): routed to `general` and returned the out-of-scope response.
- Retrieval smoke tests: `keyword`, `vector`, and `hybrid` modes all executed successfully.
- MLflow packaging: `python scripts/log_mlflow_model.py` logged run `2c02e990203c4ae792b1dd1a740b56d3`; `python scripts/load_mlflow_model.py` loaded and predicted successfully.

## Evaluation Results

Live DeepSeek evaluation (`python scripts/run_eval.py`):

- Accuracy: 1.00 (10/10)
- used_llm_rate: 1.00
- fallback_cases: 0
- avg_latency_seconds: 2.9374
- max_latency_seconds: 5.2371
- mean_semantic_similarity: 0.6935
- mean_token_coverage: 0.7248
- failed_cases_count: 0

Offline no-key fallback evaluation (`eval_results_offline.json`):

- Accuracy: 0.70 (7/10)
- used_llm_rate: 0.00
- fallback_cases: 10
- avg_latency_seconds: 0.0034
- max_latency_seconds: 0.0119
- mean_semantic_similarity: 0.5487
- mean_token_coverage: 0.5704
- failed_cases_count: 3

## Major Fixes

- Removed deterministic per-question answer branches and exact-term answer stabilization.
- Preserved live model output when `used_llm=True`; only whitespace normalization is applied.
- Replaced fallback generation with a generic grounded extractor over retrieved context.
- Replaced question-id-specific evaluation facts with Expected_Answer similarity and token coverage.
- Added MLflow metric logging and evaluation artifact logging.
- Added conditional LangGraph branches for out-of-scope questions, broadened retrieval, and human review.
- Added FAISS `IndexFlatIP` vector search with generic hashing/numpy fallback.
- Added multi-source retrieval depth for comparison and feature-availability questions so table rows are not dropped.
- Added MLflow pyfunc signature, input example, artifact paths, and resolvable pip requirements.
- Removed non-required platform-specific deployment scope from project docs/config.

## Known Tradeoffs

- Offline no-key accuracy is lower because the fallback is extractive and corpus-agnostic.
- First sentence-transformers use may require a model download unless hashing fallback is forced.
- Current hybrid retrieval only merges keyword and vector results; reranking is future work.
- Live accuracy and latency depend on the DeepSeek API and network behavior.
- MLflow emits non-fatal warnings about pyfunc type hints and CloudPickle serialization.
