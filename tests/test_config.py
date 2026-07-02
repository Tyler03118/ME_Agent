from me_agent.config import AgentConfig


def test_agent_config_defaults_to_deepseek_v4_flash() -> None:
    config = AgentConfig()

    assert config.model_name == "deepseek-v4-flash"
    assert config.openai_base_url == "https://api.deepseek.com"
    assert config.anthropic_base_url == "https://api.deepseek.com/anthropic"
    assert config.api_key_env_var == "DEEPSEEK_API_KEY"
    assert config.retriever_mode == "hybrid"
    assert config.vector_weight == 0.5
    assert config.keyword_weight == 0.5


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
