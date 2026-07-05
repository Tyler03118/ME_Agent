import builtins

from me_agent.core.config import AgentConfig, PROJECT_ROOT, _load_dotenv_if_available


def test_agent_config_defaults_to_deepseek_v4_flash() -> None:
    config = AgentConfig()

    assert config.model_name == "deepseek-v4-flash"
    assert config.openai_base_url == "https://api.deepseek.com"
    assert config.anthropic_base_url == "https://api.deepseek.com/anthropic"
    assert config.api_key_env_var == "DEEPSEEK_API_KEY"
    assert config.retriever_mode == "hybrid"
    assert config.embedding_backend == "auto"


def test_agent_config_reads_deepseek_overrides_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ME_AGENT_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setenv("ME_AGENT_OPENAI_BASE_URL", "https://example.com/openai")
    monkeypatch.setenv("ME_AGENT_ANTHROPIC_BASE_URL", "https://example.com/anthropic")
    monkeypatch.setenv("ME_AGENT_API_KEY_ENV_VAR", "CUSTOM_DEEPSEEK_KEY")
    monkeypatch.setenv("ME_AGENT_RETRIEVER_MODE", "vector")

    config = AgentConfig.from_env()

    assert config.model_name == "deepseek-v4-flash"
    assert config.openai_base_url == "https://example.com/openai"
    assert config.anthropic_base_url == "https://example.com/anthropic"
    assert config.api_key_env_var == "CUSTOM_DEEPSEEK_KEY"
    assert config.retriever_mode == "vector"


def test_agent_config_resolves_relative_data_paths_from_project_root(monkeypatch) -> None:
    monkeypatch.setenv("ME_AGENT_MANUAL_DIR", "data/manuals")
    monkeypatch.setenv("ME_AGENT_EVAL_PATH", "data/eval/test-questions.csv")

    config = AgentConfig.from_env()

    assert config.manual_dir == PROJECT_ROOT / "data" / "manuals"
    assert config.eval_path == PROJECT_ROOT / "data" / "eval" / "test-questions.csv"


def test_dotenv_loading_soft_fails_when_dependency_is_unavailable(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    _load_dotenv_if_available()
