"""Run the local ECU evaluation set and log metrics to MLflow."""

from __future__ import annotations

import json

import mlflow

from me_agent.core.config import AgentConfig
from me_agent.evaluation import (
    evaluate_cases,
    load_evaluation_cases,
    log_evaluation_to_mlflow,
    write_evaluation_results,
)
from me_agent.workflow.graph import EngineeringAssistant
from me_agent.tracking.mlflow_logging import configure_mlflow_tracking


def main() -> None:
    """Evaluate the assistant against configured golden questions."""

    configure_mlflow_tracking()
    config = AgentConfig.from_env()
    cases = load_evaluation_cases(config.eval_path)
    assistant = EngineeringAssistant.from_config(config)
    with mlflow.start_run(run_name="me-agent-evaluation"):
        results = evaluate_cases(assistant, cases)
        summary = write_evaluation_results(results, "eval_results.json")
        log_evaluation_to_mlflow(summary, "eval_results.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
