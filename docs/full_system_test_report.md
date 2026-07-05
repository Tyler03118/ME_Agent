# ME Engineering Assistant Test Report

Date: 2026-07-03
Repository: /Users/ziji/Dev/ME_Agent
Branch: master

## Current Status

The project implements an honest, installable single-agent RAG system for ECU
manual QA. The system includes an enhanced stress-evaluation schema with required
facts, forbidden facts, expected sources, and expected routes. It also has
conditional LangGraph branches, DeepSeek live generation, generic extractive
fallback, FAISS/vector retrieval with offline fallback, MLflow metrics/artifact
logging, and pyfunc packaging with signature and input example.

## Validation Results

- `python3 -m pip install -e ".[dev]"`: passed in the active environment.
- Clean venv verification: `pip install -e ".[dev]"` and `python -c "import me_agent"` passed in `/private/tmp/me_agent_clean_venv`.
- `python3 -m compileall src tests scripts`: passed.
- `python3 -m pytest -q`: 57 passed, 1 non-fatal MLflow flexible-input type-hint warning.
- `python3 -m pylint src/me_agent`: 10.00/10.
- HTML report smoke render: `reports/eval_report.html` generated from
  `eval_results.json` with `docs/full_system_test_report.md` embedded.
- `python -m pip wheel . --no-deps --no-build-isolation --wheel-dir /private/tmp/me-agent-dist-check`: built `me_agent-0.1.0-py3-none-any.whl`.
- Out-of-domain query (`python scripts/run_agent.py "今天天气如何"`): routed to `general` and returned the out-of-scope response.
- Retrieval smoke tests: `keyword`, `vector`, and `hybrid` modes all executed successfully.
- MLflow packaging: a fresh temp run `d16d06b70318408eae5aed5c1fa14679` logged successfully, and `mlflow.pyfunc.load_model` loaded and predicted from outside the source checkout without `PYTHONPATH`.
- MLflow metadata includes package version, model name, retriever mode, embedding backend/model, chunking, top-k, thresholds, and artifact paths.
- Custom stress evaluation set: `data/eval/stress-questions.csv` covers paraphrases, negative evidence, prompt injection, out-of-scope refusal, multi-hop filtering, exact commands, concise differences, and terse comparison phrasing.

## Evaluation Results

Live DeepSeek evaluation (`python scripts/run_eval.py`):

- Accuracy: 1.00 (10/10)
- used_llm_rate: 1.00
- fallback_cases: 0
- avg_latency_seconds: 3.4372
- max_latency_seconds: 6.0252
- mean_semantic_similarity: 0.7038
- mean_token_coverage: 0.7284
- failed_cases_count: 0
- human_review_flags: 1 low-confidence comparison answer was conservatively flagged for review while still passing evaluation.

Live DeepSeek stress evaluation (`ME_AGENT_EVAL_PATH=data/eval/stress-questions.csv python scripts/run_eval.py` equivalent):

- Accuracy: 1.00 (14/14)
- used_llm_rate: 0.7857
- fallback_cases: 3 (2 out-of-scope, 1 provider timeout fallback)
- avg_latency_seconds: 3.1127
- max_latency_seconds: 8.5587
- mean_required_fact_recall: 0.9464
- mean_source_match: 0.9643
- mean_route_match: 1.0000
- forbidden_fact_violations: 0

Offline no-key fallback evaluation (`eval_results_offline.json`):

- Accuracy: 0.70 (7/10)
- used_llm_rate: 0.00
- fallback_cases: 10
- avg_latency_seconds: 0.0034
- max_latency_seconds: 0.0119
- mean_semantic_similarity: 0.5487
- mean_token_coverage: 0.5704
- failed_cases_count: 3

## Implemented Capabilities

- Generic grounded extraction fallback over retrieved context.
- Live model output path with source-grounded prompting and whitespace normalization.
- Expected_Answer semantic similarity and token coverage for golden evaluation.
- Optional stress-eval fields for required facts, forbidden facts, expected sources, and expected routes.
- MLflow metric logging and evaluation artifact logging.
- Self-contained HTML evaluation report renderer for JSON artifacts,
  including optional Markdown report embedding.
- Conditional LangGraph branches for out-of-scope questions, broadened retrieval, and human review.
- FAISS `IndexFlatIP` vector search with generic hashing/numpy fallback.
- Multi-source retrieval depth for comparison and feature-availability questions.
- MLflow pyfunc signature, input example, artifact paths, package code paths, and resolvable pip requirements.
- Project-root-relative data paths for scripts and MLflow logging from non-repo working directories.

## Known Tradeoffs

- Offline no-key accuracy is lower because the fallback is extractive and corpus-agnostic.
- First sentence-transformers use may require a model download unless hashing fallback is forced.
- Current hybrid retrieval only merges keyword and vector results; reranking is future work.
- Stress metrics are deterministic heuristics; production validation should add SME review and optionally LLM-as-judge evaluation for factual consistency.
- Live accuracy and latency depend on the DeepSeek API and network behavior.
- MLflow may emit non-fatal warnings about flexible pyfunc input typing, output schema inference, and CloudPickle serialization.
