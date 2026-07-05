"""Reusable MLflow logging helpers for the ME Engineering Assistant."""

from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from me_agent.core.config import AgentConfig, PROJECT_ROOT
from me_agent.tracking.mlflow_model import MEEngineeringAssistantModel

PACKAGE_NAME = "me-agent"
MODEL_CODE_PATHS = [(PROJECT_ROOT / "src" / "me_agent").as_posix()]

MODEL_REQUIREMENTS = [
    "faiss-cpu>=1.8",
    "httpx[socks]>=0.28",
    "langchain>=0.2",
    "langchain-openai>=0.3",
    "langgraph>=0.2",
    "mlflow>=2.14",
    "numpy>=1.24",
    "pandas>=1.5",
    "python-dotenv>=1.0",
    "sentence-transformers>=3.0",
]


def configure_mlflow_tracking() -> None:
    """Use SQLite locally while respecting externally configured tracking URIs."""

    if not os.getenv("MLFLOW_TRACKING_URI"):
        import mlflow  # pylint: disable=import-outside-toplevel

        mlflow.set_tracking_uri("sqlite:///mlflow.db")


def model_tracking_params(config: AgentConfig) -> dict[str, str | int | float]:
    """Return non-secret config metadata logged with the MLflow model.

    The logged params make a model run reproducible without exposing API keys.
    They capture retrieval, chunking, evaluation, and model timeout settings that
    materially affect answer quality and latency.
    """

    return {
        "package_name": PACKAGE_NAME,
        "package_version": _package_version(),
        "model_name": config.model_name,
        "retriever_mode": config.retriever_mode,
        "embedding_backend": config.embedding_backend,
        "embedding_model_name": config.embedding_model_name,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "confidence_threshold": config.confidence_threshold,
        "retrieval_retry_threshold": config.retrieval_retry_threshold,
        "eval_pass_threshold": config.eval_pass_threshold,
        "model_timeout_seconds": config.model_timeout_seconds,
        "model_max_retries": config.model_max_retries,
        "manual_dir": config.manual_dir.as_posix(),
        "eval_path": config.eval_path.as_posix(),
        "api_key_env_var": config.api_key_env_var,
    }


def _package_version() -> str:
    """Return the installed package version or the project fallback version."""

    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return "0.1.0"


def log_assistant_model(config: AgentConfig | None = None) -> str:
    """Log the pyfunc model with signature, input example, and corpus artifacts."""

    import mlflow  # pylint: disable=import-outside-toplevel
    from mlflow.models import infer_signature  # pylint: disable=import-outside-toplevel

    configure_mlflow_tracking()
    resolved_config = config or AgentConfig.from_env()
    artifacts = {
        "manuals": resolved_config.manual_dir.as_posix(),
        "eval_questions": resolved_config.eval_path.as_posix(),
    }
    # A realistic signature matters for deployment: consumers can inspect that
    # the pyfunc expects a DataFrame-like input with a ``question`` column and
    # returns structured response dictionaries.
    input_example = pd.DataFrame({"question": ["How much RAM does the ECU-850 have?"]})
    output_example: list[dict[str, Any]] = [
        {
            "question": "How much RAM does the ECU-850 have?",
            "answer": "The ECU-850 has 2 GB LPDDR4 RAM.",
            "route_category": "ecu_800_lookup",
            "retriever_mode": resolved_config.retriever_mode,
            "used_llm": False,
            "fallback_reason": "example",
            "sources": ("ECU-800_Series_Base.md",),
            "verifier_status": "supported",
            "confidence": 0.75,
            "needs_human_review": False,
            "review_reason": "",
        }
    ]
    signature = infer_signature(input_example, output_example)
    tracking_params = model_tracking_params(resolved_config)
    with mlflow.start_run(run_name="me-engineering-assistant") as run:
        mlflow.log_params(tracking_params)
        mlflow.set_tags(
            {
                "package_name": PACKAGE_NAME,
                "package_version": str(tracking_params["package_version"]),
                "model_flavor": "mlflow.pyfunc",
                "llm_provider": "deepseek",
                "retriever_mode": resolved_config.retriever_mode,
                "embedding_backend": resolved_config.embedding_backend,
            }
        )
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MEEngineeringAssistantModel(),
            artifacts=artifacts,
            code_paths=MODEL_CODE_PATHS,
            signature=signature,
            input_example=input_example,
            pip_requirements=MODEL_REQUIREMENTS,
        )
        Path("outputs").mkdir(exist_ok=True)
        (Path("outputs") / "latest_run_id.txt").write_text(run.info.run_id, encoding="utf-8")
        print(f"Logged model in run {run.info.run_id}")
        return run.info.run_id
