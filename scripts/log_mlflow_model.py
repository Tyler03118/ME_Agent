"""Log the ME Engineering Assistant as an MLflow pyfunc model."""

from __future__ import annotations

from me_agent.tracking.mlflow_logging import log_assistant_model


def main() -> None:
    """Log the model to the active MLflow tracking URI."""

    log_assistant_model()


if __name__ == "__main__":
    main()
