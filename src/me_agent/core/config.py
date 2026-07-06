"""Runtime configuration for the ME Engineering Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _project_relative_path(value: str | Path) -> Path:
    """Resolve relative project config paths against the package project root."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_dotenv_if_available() -> None:
    """Load a local .env file when python-dotenv is installed."""

    try:
        from dotenv import load_dotenv  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class AgentConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration values used by the local RAG workflow."""

    manual_dir: Path = PROJECT_ROOT / "data" / "manuals"
    eval_path: Path = PROJECT_ROOT / "data" / "eval" / "test-questions.csv"
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 8
    retriever_mode: str = "hybrid"
    confidence_threshold: float = 0.65
    retrieval_retry_threshold: float = 0.08
    eval_pass_threshold: float = 0.60
    model_timeout_seconds: float = 8.0
    model_max_retries: int = 0
    model_name: str = "deepseek-v4-flash"
    openai_base_url: str = "https://api.deepseek.com"
    anthropic_base_url: str = "https://api.deepseek.com/anthropic"
    api_key_env_var: str = "DEEPSEEK_API_KEY"
    embedding_backend: str = "auto"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Build configuration from environment variables without requiring .env support."""

        _load_dotenv_if_available()
        defaults = cls()
        return cls(
            manual_dir=_project_relative_path(
                os.getenv("ME_AGENT_MANUAL_DIR", defaults.manual_dir.as_posix())
            ),
            eval_path=_project_relative_path(
                os.getenv("ME_AGENT_EVAL_PATH", defaults.eval_path.as_posix())
            ),
            chunk_size=int(os.getenv("ME_AGENT_CHUNK_SIZE", str(defaults.chunk_size))),
            chunk_overlap=int(os.getenv("ME_AGENT_CHUNK_OVERLAP", str(defaults.chunk_overlap))),
            top_k=int(os.getenv("ME_AGENT_TOP_K", str(defaults.top_k))),
            retriever_mode=os.getenv("ME_AGENT_RETRIEVER_MODE", defaults.retriever_mode),
            confidence_threshold=float(
                os.getenv("ME_AGENT_CONFIDENCE_THRESHOLD", str(defaults.confidence_threshold))
            ),
            retrieval_retry_threshold=float(
                os.getenv(
                    "ME_AGENT_RETRIEVAL_RETRY_THRESHOLD",
                    str(defaults.retrieval_retry_threshold),
                )
            ),
            eval_pass_threshold=float(
                os.getenv("ME_AGENT_EVAL_PASS_THRESHOLD", str(defaults.eval_pass_threshold))
            ),
            model_timeout_seconds=float(
                os.getenv("ME_AGENT_MODEL_TIMEOUT_SECONDS", str(defaults.model_timeout_seconds))
            ),
            model_max_retries=int(
                os.getenv("ME_AGENT_MODEL_MAX_RETRIES", str(defaults.model_max_retries))
            ),
            model_name=os.getenv("ME_AGENT_MODEL_NAME", defaults.model_name),
            openai_base_url=os.getenv("ME_AGENT_OPENAI_BASE_URL", defaults.openai_base_url),
            anthropic_base_url=os.getenv(
                "ME_AGENT_ANTHROPIC_BASE_URL",
                defaults.anthropic_base_url,
            ),
            api_key_env_var=os.getenv("ME_AGENT_API_KEY_ENV_VAR", defaults.api_key_env_var),
            embedding_backend=os.getenv("ME_AGENT_EMBEDDING_BACKEND", defaults.embedding_backend),
            embedding_model_name=os.getenv(
                "ME_AGENT_EMBEDDING_MODEL_NAME",
                defaults.embedding_model_name,
            ),
        )
