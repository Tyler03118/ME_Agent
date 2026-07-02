"""Reusable MLflow logging helpers for the ME Engineering Assistant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from me_agent.core.config import AgentConfig
from me_agent.tracking.mlflow_model import MEEngineeringAssistantModel

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
    with mlflow.start_run(run_name="me-engineering-assistant") as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MEEngineeringAssistantModel(),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example,
            pip_requirements=MODEL_REQUIREMENTS,
        )
        Path("outputs").mkdir(exist_ok=True)
        (Path("outputs") / "latest_run_id.txt").write_text(run.info.run_id, encoding="utf-8")
        print(f"Logged model in run {run.info.run_id}")
        return run.info.run_id
