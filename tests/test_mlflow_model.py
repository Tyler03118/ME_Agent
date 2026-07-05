from pathlib import Path

import pandas as pd

from me_agent.core.config import AgentConfig
from me_agent.tracking.mlflow_logging import MODEL_CODE_PATHS, MODEL_REQUIREMENTS, model_tracking_params
from me_agent.tracking.mlflow_model import MEEngineeringAssistantModel


def test_mlflow_model_predict_accepts_list_input() -> None:
    model = MEEngineeringAssistantModel()

    predictions = model.predict(None, ["How much RAM does ECU-850 have?"])

    assert len(predictions) == 1
    assert predictions[0]["question"] == "How much RAM does ECU-850 have?"
    assert "answer" in predictions[0]


def test_mlflow_model_predict_accepts_dataframe_input() -> None:
    model = MEEngineeringAssistantModel()

    predictions = model.predict(None, pd.DataFrame({"question": ["今天天气如何"]}))

    assert predictions[0]["route_category"] == "general"
    assert "outside the ECU manual scope" in predictions[0]["answer"]


def test_load_context_uses_packaged_artifacts(monkeypatch, tmp_path) -> None:
    import me_agent.tracking.mlflow_model as mlflow_model

    manuals = tmp_path / "manuals"
    eval_questions = tmp_path / "test-questions.csv"
    manuals.mkdir()
    eval_questions.write_text("Question_ID,Question,Expected_Answer,Evaluation_Criteria\n", encoding="utf-8")
    captured = {}

    def fake_from_config(config=None):
        captured["config"] = config
        return object()

    monkeypatch.setattr(
        mlflow_model.EngineeringAssistant,
        "from_config",
        staticmethod(fake_from_config),
    )

    class Context:
        artifacts = {
            "manuals": manuals.as_posix(),
            "eval_questions": eval_questions.as_posix(),
        }

    model = MEEngineeringAssistantModel()
    model.load_context(Context())

    assert captured["config"].manual_dir == manuals
    assert captured["config"].eval_path == eval_questions


def test_mlflow_requirements_are_resolvable_packages() -> None:
    joined = "\n".join(MODEL_REQUIREMENTS)

    assert "me-agent" not in joined
    assert "langchain" in joined
    assert "sentence-transformers" in joined
    assert "faiss-cpu" in joined


def test_mlflow_logging_includes_package_code_path() -> None:
    assert any(
        Path(path).name == "me_agent" and (Path(path) / "__init__.py").exists()
        for path in MODEL_CODE_PATHS
    )


def test_mlflow_tracking_params_include_versioning_metadata() -> None:
    params = model_tracking_params(
        AgentConfig(
            model_name="deepseek-v4-flash",
            retriever_mode="hybrid",
            embedding_backend="hashing",
        )
    )

    assert params["package_version"]
    assert params["model_name"] == "deepseek-v4-flash"
    assert params["retriever_mode"] == "hybrid"
    assert params["embedding_backend"] == "hashing"
    assert params["api_key_env_var"] == "DEEPSEEK_API_KEY"
    assert not any("key_value" in key.lower() for key in params)
