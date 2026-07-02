"""Load a logged MLflow model and run a smoke prediction."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    """Load the model URI from environment or latest local run output."""

    import mlflow

    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model_uri = os.getenv("ME_AGENT_MODEL_URI")
    if not model_uri:
        run_id_path = Path("outputs") / "latest_run_id.txt"
        if not run_id_path.exists():
            raise FileNotFoundError(
                "Set ME_AGENT_MODEL_URI or run scripts/log_mlflow_model.py first."
            )
        run_id = run_id_path.read_text(encoding="utf-8").strip()
        model_uri = f"runs:/{run_id}/model"

    model = mlflow.pyfunc.load_model(model_uri)
    prediction = model.predict(["How much RAM does the ECU-850 have?"])
    print(prediction)


if __name__ == "__main__":
    main()
