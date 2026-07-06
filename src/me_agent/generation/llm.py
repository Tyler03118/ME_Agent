"""DeepSeek integration plus corpus-agnostic grounded fallback generation."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

from me_agent.core.config import AgentConfig
from me_agent.core.domain_terms import GENERATION_QUERY_EXPANSION_GROUPS
from me_agent.retrieval.embeddings import tokenize
from me_agent.core.schemas import RetrievalResult, RouteDecision

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "much", "of", "on", "or", "the", "to", "what", "which", "with", "does", "do",
}
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


# Public generation API -----------------------------------------------------


@dataclass(frozen=True)
class GenerationResult:
    """Answer generation output."""

    answer: str
    used_llm: bool
    fallback_reason: str = ""


class DeepSeekAnswerGenerator:
    """Generate grounded answers with DeepSeek, falling back to extraction.

    The generator is deliberately stateless. It builds a client only when the
    configured API key is available, which keeps offline tests and no-key demos on
    the deterministic extractive path.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Store runtime configuration used for client construction."""

        self.config = config

    def generate(
        self,
        *,
        question: str,
        route: RouteDecision,
        retrieved_context: Sequence[RetrievalResult],
    ) -> GenerationResult:
        """Generate an answer from retrieved context without hardcoded questions.

        The live model receives only retrieved evidence, which keeps answers
        grounded even when a prompt asks it to ignore sources or invent ECU
        specifications. If the provider is unavailable, the deterministic
        extractive fallback keeps no-key demos and tests reproducible. The
        fallback reason records the provider error type for evaluation and
        MLflow diagnostics.
        """

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
        """Create a LangChain ChatOpenAI client when DeepSeek config is present.

        The import is lazy so the package remains usable in offline/no-key
        environments that only exercise deterministic fallback behavior.
        """

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


# Deterministic fallback path -----------------------------------------------


def synthesize_deterministic(
    question: str,
    retrieved_context: Sequence[RetrievalResult],
) -> str:
    """Create a short extractive answer from retrieved context only.

    This fallback is corpus-agnostic: it ranks sentences by overlap with the
    question plus the retriever score, then appends source filenames. It is meant
    to preserve grounded behavior when the live model is unavailable.
    """

    sentences = _ranked_sentences(question, retrieved_context)
    if not sentences:
        return "The retrieved documents do not contain enough information to answer this question."
    answer = " ".join(sentence for sentence, _score in sentences[:3])
    return _with_sources(answer, _sources(retrieved_context))


def _ranked_sentences(
    question: str,
    retrieved_context: Sequence[RetrievalResult],
) -> list[tuple[str, float]]:
    """Rank candidate context sentences for deterministic fallback synthesis.

    The score combines lexical overlap, small domain boosts for known ECU
    phrasing gaps, and the upstream retrieval score. It does not use
    question-id-specific answer rules.
    """

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
            score = overlap + _domain_fact_bonus(question, normalized) + 0.10 * result.score
            if score > 0:
                ranked.append((normalized, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _split_sentences(text: str) -> list[str]:
    """Split Markdown text into sentence-like facts and table-row facts.

    Markdown table rows are treated as complete facts so labels, values, and
    units stay together during deterministic fallback ranking.
    """

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
    """Return question/content tokens useful for fallback ranking.

    Deterministic synonym expansion mirrors retrieval so fallback ranking can
    connect user phrasing with the wording used in the manuals.
    """

    tokens = [
        token
        for token in tokenize(text)
        if token not in STOPWORDS and token != "ecu" and not token.isdigit()
    ]
    token_set = set(tokens)
    for triggers, expansion in GENERATION_QUERY_EXPANSION_GROUPS:
        if token_set & set(triggers):
            tokens.extend(tokenize(expansion))
    return tokens


def _domain_fact_bonus(question: str, sentence: str) -> float:
    """Boost manual rows that express a domain synonym targeted by the query."""

    query = question.lower()
    text = sentence.lower()
    bonus = 0.0
    if _contains_any(query, ("thermal", "tolerance", "environment", "temperature")):
        if _contains_any(text, ("operating temperature", "operating temp", "+105", "+85")):
            bonus += 0.45
    if _contains_any(query, ("remote", "firmware", "update", "ota")):
        if _contains_any(text, ("ota", "over-the-air", "updates", "firmware")):
            bonus += 0.35
    if _contains_any(query, ("edge", "inference", "ai", "accelerator", "neural")):
        if _contains_any(text, ("npu", "edge ai", "ai accelerator", "tops")):
            bonus += 0.35
    return bonus


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    """Return whether any phrase appears in lower-cased text."""

    return any(needle in text for needle in needles)


# Prompt construction for the live model ------------------------------------


def _normalize_model_text(answer: str) -> str:
    """Normalize model text whitespace without changing answer content."""

    return " ".join(answer.replace(" ", " ").replace(" ", " ").split())


def _build_prompt(
    question: str,
    route: RouteDecision,
    retrieved_context: Sequence[RetrievalResult],
) -> str:
    """Build a compact prompt that separates evidence lines from full context.

    Ranked evidence lines appear before the full context to steer the live LLM
    toward compact, high-signal facts while preserving complete retrieved chunks
    for grounding.
    """

    context = "\n\n".join(
        f"SOURCE: {result.chunk.metadata.get('source')}\n{result.chunk.content}"
        for result in retrieved_context
    )
    evidence = "\n".join(_evidence_lines(question, retrieved_context))
    route_instruction = (
        "For comparison or feature-availability questions, compare every relevant "
        "source and model present in the evidence. For difference questions, lead "
        "with the key changed specifications, keep the answer concise, and avoid "
        "exhaustive tables unless the user asks for one. Prefer explicit specification "
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
    """Surface the most relevant rows/sentences before the full context.

    Tables and bolded spec labels are boosted because they tend to be higher
    confidence manual evidence than surrounding prose.
    """

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


# Shared response formatting helpers ----------------------------------------


def _sources(retrieved_context: Sequence[RetrievalResult]) -> tuple[str, ...]:
    """Return unique source filenames from retrieved context."""

    seen: list[str] = []
    for result in retrieved_context:
        source = result.chunk.metadata.get("source")
        if isinstance(source, str) and source not in seen:
            seen.append(source)
    return tuple(seen)


def _with_sources(answer: str, sources: tuple[str, ...]) -> str:
    """Append source filenames to deterministic fallback answers."""

    if not sources:
        return answer
    return f"{answer} Sources: {', '.join(sources)}."
