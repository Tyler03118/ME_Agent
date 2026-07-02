from me_agent.mlflow_model import MEEngineeringAssistantModel


def test_mlflow_model_predict_accepts_list_input() -> None:
    model = MEEngineeringAssistantModel()

    predictions = model.predict(None, ["How much RAM does ECU-850 have?"])

    assert len(predictions) == 1
    assert predictions[0]["question"] == "How much RAM does ECU-850 have?"
    assert "answer" in predictions[0]


def test_load_context_uses_packaged_artifacts(monkeypatch, tmp_path) -> None:
    import me_agent.mlflow_model as mlflow_model

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
