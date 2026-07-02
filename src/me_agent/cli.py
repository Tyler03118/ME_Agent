"""Console entry points for local ME Engineering Assistant workflows."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd

from me_agent.core.config import AgentConfig
from me_agent.evaluation import (
    evaluate_cases,
    load_evaluation_cases,
    log_evaluation_to_mlflow,
    write_evaluation_results,
)
from me_agent.workflow.graph import EngineeringAssistant
from me_agent.tracking.mlflow_logging import configure_mlflow_tracking, log_assistant_model


def ask_cli() -> None:
    """Answer a single question from command-line arguments."""

    if len(sys.argv) < 2:
        raise SystemExit('Usage: me-agent "question"')
    assistant = EngineeringAssistant.from_config()
    response = assistant.ask(" ".join(sys.argv[1:]))
    print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))


def run_eval_cli() -> None:
    """Run the golden evaluation set and log metrics to MLflow."""

    configure_mlflow_tracking()
    config = AgentConfig.from_env()
    cases = load_evaluation_cases(config.eval_path)
    assistant = EngineeringAssistant.from_config(config)
    with mlflow.start_run(run_name="me-agent-evaluation"):
        results = evaluate_cases(assistant, cases)
        summary = write_evaluation_results(results, "eval_results.json")
        log_evaluation_to_mlflow(summary, "eval_results.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def log_model_cli() -> None:
    """Log the assistant as a self-contained MLflow pyfunc model."""

    log_assistant_model()


def load_model_cli() -> None:
    """Load the latest logged MLflow model and run a smoke prediction."""

    configure_mlflow_tracking()
    model_uri = os.getenv("ME_AGENT_MODEL_URI")
    if not model_uri:
        run_id_path = Path("outputs") / "latest_run_id.txt"
        if not run_id_path.exists():
            raise FileNotFoundError("Set ME_AGENT_MODEL_URI or run me-agent-log-model first.")
        run_id = run_id_path.read_text(encoding="utf-8").strip()
        model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    print(model.predict(pd.DataFrame({"question": ["How much RAM does the ECU-850 have?"]})))
