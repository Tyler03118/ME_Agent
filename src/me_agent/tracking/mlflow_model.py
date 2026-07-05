"""MLflow pyfunc wrapper for the ME Engineering Assistant."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from me_agent.core.config import AgentConfig
from me_agent.workflow.graph import EngineeringAssistant

try:
    import mlflow.pyfunc
except ImportError:  # pragma: no cover - exercised only before optional dependency install.
    _PythonModelBase = object
else:
    _PythonModelBase = mlflow.pyfunc.PythonModel


class MEEngineeringAssistantModel(_PythonModelBase):
    """Custom MLflow model exposing the assistant through ``predict``.

    MLflow loads this wrapper in a fresh process, so ``load_context`` rebuilds the
    assistant from packaged artifacts instead of relying on the source checkout's
    working directory.
    """

    def __init__(self) -> None:
        """Defer assistant construction until MLflow provides load context."""

        self._assistant: EngineeringAssistant | None = None

    def load_context(self, context: Any) -> None:
        """Load the assistant with packaged MLflow artifacts when available."""

        config = AgentConfig.from_env()
        artifacts = getattr(context, "artifacts", None) if context is not None else None
        if artifacts:
            # Logged models carry their manual directory and eval CSV as MLflow
            # artifacts. Replacing these paths makes the pyfunc portable outside
            # the repository where it was originally logged.
            manual_dir = artifacts.get("manuals")
            eval_path = artifacts.get("eval_questions")
            config = replace(
                config,
                manual_dir=Path(manual_dir) if manual_dir else config.manual_dir,
                eval_path=Path(eval_path) if eval_path else config.eval_path,
            )
        self._assistant = EngineeringAssistant.from_config(config)

    # MLflow requires the context parameter even though prediction does not use it directly.
    # pylint: disable=unused-argument
    def predict(
        self,
        context: Any,
        model_input,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return one structured assistant response per input question."""

        del context, params
        assistant = self._assistant
        if assistant is None:
            assistant = EngineeringAssistant.from_config()
            self._assistant = assistant
        return [assistant.ask(question).to_dict() for question in _extract_questions(model_input)]

    def predict_stream(
        self,
        context: Any,
        model_input,
        params: dict[str, Any] | None = None,
    ):
        """Yield predictions for MLflow streaming interfaces."""

        yield from self.predict(context, model_input, params)


def _extract_questions(model_input: Any) -> list[str]:
    """Normalize MLflow-supported input shapes into a list of questions."""

    if isinstance(model_input, str):
        return [model_input]
    if isinstance(model_input, list):
        return [_extract_question_from_item(item) for item in model_input]
    if hasattr(model_input, "to_dict"):
        rows = model_input.to_dict(orient="records")
        return [_extract_question_from_item(row) for row in rows]
    raise TypeError("model_input must be a string, list, or DataFrame-like object")


def _extract_question_from_item(item: Any) -> str:
    """Extract one question string from a scalar or mapping input item."""

    if isinstance(item, str):
        return item
    if isinstance(item, dict) and "question" in item:
        return str(item["question"])
    raise TypeError("Each model input item must be a string or contain a question field")
