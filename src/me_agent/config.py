"""Runtime configuration for the ME Engineering Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AgentConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration values used by the local RAG workflow."""

    manual_dir: Path = PROJECT_ROOT / "data" / "manuals"
    eval_path: Path = PROJECT_ROOT / "data" / "eval" / "test-questions.csv"
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 4
    retriever_mode: str = "hybrid"
    keyword_weight: float = 0.5
    vector_weight: float = 0.5
    confidence_threshold: float = 0.65
    model_timeout_seconds: float = 8.0
    model_max_retries: int = 0
    model_name: str = "deepseek-v4-flash"
    openai_base_url: str = "https://api.deepseek.com"
    anthropic_base_url: str = "https://api.deepseek.com/anthropic"
    api_key_env_var: str = "DEEPSEEK_API_KEY"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Build configuration from environment variables."""

        load_dotenv(PROJECT_ROOT / ".env")
        defaults = cls()
        return cls(
            manual_dir=Path(os.getenv("ME_AGENT_MANUAL_DIR", defaults.manual_dir.as_posix())),
            eval_path=Path(os.getenv("ME_AGENT_EVAL_PATH", defaults.eval_path.as_posix())),
            chunk_size=int(os.getenv("ME_AGENT_CHUNK_SIZE", str(defaults.chunk_size))),
            chunk_overlap=int(os.getenv("ME_AGENT_CHUNK_OVERLAP", str(defaults.chunk_overlap))),
            top_k=int(os.getenv("ME_AGENT_TOP_K", str(defaults.top_k))),
            retriever_mode=os.getenv("ME_AGENT_RETRIEVER_MODE", defaults.retriever_mode),
            keyword_weight=float(
                os.getenv("ME_AGENT_KEYWORD_WEIGHT", str(defaults.keyword_weight))
            ),
            vector_weight=float(
                os.getenv("ME_AGENT_VECTOR_WEIGHT", str(defaults.vector_weight))
            ),
            confidence_threshold=float(
                os.getenv("ME_AGENT_CONFIDENCE_THRESHOLD", str(defaults.confidence_threshold))
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
        )
