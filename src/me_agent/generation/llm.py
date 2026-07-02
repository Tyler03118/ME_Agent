"""DeepSeek integration plus corpus-agnostic grounded fallback generation."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from me_agent.core.config import AgentConfig
from me_agent.retrieval.embeddings import tokenize
from me_agent.core.schemas import RetrievalResult, RouteDecision

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "much", "of", "on", "or", "the", "to", "what", "which", "with", "does", "do",
}
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class GenerationResult:
    """Answer generation output."""

    answer: str
    used_llm: bool
    fallback_reason: str = ""


class DeepSeekAnswerGenerator:
    """Generate grounded answers with DeepSeek, falling back to extraction."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def generate(
        self,
        *,
        question: str,
        route: RouteDecision,
        retrieved_context: Sequence[RetrievalResult],
    ) -> GenerationResult:
        """Generate an answer from retrieved context without hardcoded question paths."""

        if not retrieved_context:
            return GenerationResult(
                (
                    "The retrieved documents do not contain enough information "
                    "to answer this question."
                ),
                used_llm=False,
                fallback_reason="no_context",
            )

        prompt = _build_prompt(question, route, retrieved_context)
        client = self._build_client()
        if client is not None:
            try:
                response = client.invoke(prompt)
                content = _normalize_model_text(getattr(response, "content", str(response)))
                if content:
                    return GenerationResult(content, used_llm=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return GenerationResult(
                    synthesize_deterministic(question, retrieved_context),
                    used_llm=False,
                    fallback_reason=type(exc).__name__,
                )

        return GenerationResult(
            synthesize_deterministic(question, retrieved_context),
            used_llm=False,
            fallback_reason="missing_deepseek_client_or_key",
        )

    def _build_client(self):
        api_key = os.getenv(self.config.api_key_env_var)
        if not api_key:
            return None
        try:
            from langchain_openai import ChatOpenAI  # pylint: disable=import-outside-toplevel
        except ImportError:
            return None
        return ChatOpenAI(
            model=self.config.model_name,
            api_key=api_key,
            base_url=self.config.openai_base_url,
            temperature=0,
            timeout=self.config.model_timeout_seconds,
            max_retries=self.config.model_max_retries,
        )


def synthesize_deterministic(
    question: str,
    retrieved_context: Sequence[RetrievalResult],
) -> str:
    """Create a short extractive answer from retrieved context only."""

    sentences = _ranked_sentences(question, retrieved_context)
    if not sentences:
        return "The retrieved documents do not contain enough information to answer this question."
    answer = " ".join(sentence for sentence, _score in sentences[:3])
    return _with_sources(answer, _sources(retrieved_context))


def _ranked_sentences(
    question: str,
    retrieved_context: Sequence[RetrievalResult],
) -> list[tuple[str, float]]:
    query_tokens = set(_content_tokens(question))
    ranked: list[tuple[str, float]] = []
    seen: set[str] = set()
    for result in retrieved_context:
        for sentence in _split_sentences(result.chunk.content):
            normalized = " ".join(sentence.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            sentence_tokens = set(_content_tokens(normalized))
            overlap = len(query_tokens & sentence_tokens) / max(len(query_tokens), 1)
            score = overlap + 0.10 * result.score
            if score > 0:
                ranked.append((normalized, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "|" in stripped:
            table_text = " ".join(part.strip() for part in stripped.split("|") if part.strip())
            if set(table_text.replace(" ", "")) <= {"-", ":"}:
                continue
            sentences.append(table_text)
            continue
        sentences.extend(SENTENCE_PATTERN.split(stripped))
    return [part.strip(" -•\t") for part in sentences if part.strip(" -•\t")]


def _content_tokens(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if token not in STOPWORDS and token != "ecu" and not token.isdigit()
    ]


def _normalize_model_text(answer: str) -> str:
    return " ".join(answer.replace(" ", " ").replace(" ", " ").split())


def _build_prompt(
    question: str,
    route: RouteDecision,
    retrieved_context: Sequence[RetrievalResult],
) -> str:
    context = "\n\n".join(
        f"SOURCE: {result.chunk.metadata.get('source')}\n{result.chunk.content}"
        for result in retrieved_context
    )
    evidence = "\n".join(_evidence_lines(question, retrieved_context))
    route_instruction = (
        "For comparison or feature-availability questions, compare every relevant "
        "source and model present in the evidence. Prefer explicit specification "
        "table rows over inheritance statements or summaries."
        if route.category in {"comparison", "feature_availability"}
        else "Prefer exact specification values from evidence lines and table rows."
    )
    return (
        "You are a concise ECU engineering assistant. Answer only from the provided "
        "context. Do not dump raw context. Synthesize the exact facts needed and cite "
        "source filenames in brackets. If context is insufficient, say so. "
        f"{route_instruction}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE_LINES:\n{evidence}\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:"
    )


def _evidence_lines(
    question: str,
    retrieved_context: Sequence[RetrievalResult],
    *,
    limit: int = 32,
) -> list[str]:
    query_tokens = set(_content_tokens(question))
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for result in retrieved_context:
        source = result.chunk.metadata.get("source", "unknown")
        for sentence in _split_sentences(result.chunk.content):
            normalized = " ".join(sentence.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            tokens = set(_content_tokens(normalized))
            overlap = len(query_tokens & tokens) / max(len(query_tokens), 1)
            table_bonus = 0.35 if "|" in sentence or "**" in sentence else 0.0
            evidence_line = f"- [{source}] {normalized}"
            scored.append((overlap + table_bonus + 0.05 * result.score, evidence_line))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [line for _score, line in scored[:limit]]


def _sources(retrieved_context: Sequence[RetrievalResult]) -> tuple[str, ...]:
    seen: list[str] = []
    for result in retrieved_context:
        source = result.chunk.metadata.get("source")
        if isinstance(source, str) and source not in seen:
            seen.append(source)
    return tuple(seen)


def _with_sources(answer: str, sources: tuple[str, ...]) -> str:
    if not sources:
        return answer
    return f"{answer} Sources: {', '.join(sources)}."
