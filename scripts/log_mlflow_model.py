"""Log the ME Engineering Assistant as an MLflow pyfunc model."""

from __future__ import annotations

import os
from pathlib import Path

from me_agent.config import AgentConfig
from me_agent.mlflow_model import MEEngineeringAssistantModel


def main() -> None:
    """Log the model to the active MLflow tracking URI."""

    import mlflow

    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

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


if __name__ == "__main__":
    main()
