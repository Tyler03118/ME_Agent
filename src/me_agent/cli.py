"""Console entry points for local and Databricks bundle jobs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow

from me_agent.config import AgentConfig
from me_agent.evaluation import evaluate_cases, load_evaluation_cases, write_evaluation_results
from me_agent.graph import EngineeringAssistant
from me_agent.mlflow_model import MEEngineeringAssistantModel


def ask_cli() -> None:
    """Answer a single question from command-line arguments."""

    if len(sys.argv) < 2:
        raise SystemExit('Usage: me-agent "question"')
    assistant = EngineeringAssistant.from_config()
    response = assistant.ask(" ".join(sys.argv[1:]))
    print(json.dumps(response.to_dict(), indent=2))


def run_eval_cli() -> None:
    """Run the golden evaluation set and write eval_results.json."""

    config = AgentConfig.from_env()
    cases = load_evaluation_cases(config.eval_path)
    assistant = EngineeringAssistant.from_config(config)
    results = evaluate_cases(assistant, cases)
    summary = write_evaluation_results(results, "eval_results.json")
    print(json.dumps(summary, indent=2))


def log_model_cli() -> None:
    """Log the assistant as a self-contained MLflow pyfunc model."""

    _configure_mlflow_tracking()
    config = AgentConfig.from_env()
    artifacts = {
        "manuals": config.manual_dir.as_posix(),
        "eval_questions": config.eval_path.as_posix(),
    }
    with mlflow.start_run(run_name="me-engineering-assistant") as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MEEngineeringAssistantModel(),
            artifacts=artifacts,
            pip_requirements=["me-agent"],
        )
        Path("outputs").mkdir(exist_ok=True)
        (Path("outputs") / "latest_run_id.txt").write_text(run.info.run_id, encoding="utf-8")
        print(f"Logged model in run {run.info.run_id}")


def load_model_cli() -> None:
    """Load the latest logged MLflow model and run a smoke prediction."""

    _configure_mlflow_tracking()
    model_uri = os.getenv("ME_AGENT_MODEL_URI")
    if not model_uri:
        run_id_path = Path("outputs") / "latest_run_id.txt"
        if not run_id_path.exists():
            raise FileNotFoundError("Set ME_AGENT_MODEL_URI or run me-agent-log-model first.")
        run_id = run_id_path.read_text(encoding="utf-8").strip()
        model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    print(model.predict(["How much RAM does the ECU-850 have?"]))


def _configure_mlflow_tracking() -> None:
    """Use an explicit local MLflow backend unless the runtime provides one."""

    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
