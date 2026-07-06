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
    """Return value from the answer generator.

    Fields:
    - ``answer``: final text shown to the user;
    - ``used_llm``: ``True`` only when the live provider returned text;
    - ``fallback_reason``: why deterministic extraction was used, if any.
    """

    answer: str
    used_llm: bool
    fallback_reason: str = ""


class DeepSeekAnswerGenerator:
    """Answer ECU questions from retrieved chunks.

    Runtime behavior:
    - with a configured API key, call DeepSeek through ``ChatOpenAI``;
    - without a key, use deterministic extraction from retrieved text;
    - if the provider errors, fall back instead of crashing the graph.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Store model name, endpoint, timeout, retry, and API-key settings."""

        self.config = config

    def generate(
        self,
        *,
        question: str,
        route: RouteDecision,
        retrieved_context: Sequence[RetrievalResult],
    ) -> GenerationResult:
        """Generate one answer from one question and its retrieved chunks.

        Flow:
        - if retrieval returned no chunks, report insufficient context;
        - build a prompt from the question, route, and retrieved chunks;
        - call the live model when a client can be built;
        - normalize live model whitespace before returning;
        - on missing key/import error/provider error, return extractive fallback.
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
        """Build the DeepSeek-compatible LangChain client when possible.

        Return ``None`` when:
        - the configured API-key environment variable is missing;
        - ``langchain_openai`` is not installed in the current environment.
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
    """Create an answer without calling an LLM.

    Steps:
    - rank sentence/table-row facts from retrieved chunks;
    - join the top three facts into a compact answer;
    - append source filenames so the fallback still cites evidence.
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
    """Score each fact-like line from retrieved chunks.

    For each candidate sentence/table row:
    - normalize whitespace and skip duplicates;
    - compute token overlap with the question;
    - add small boosts for ECU terms like temperature, OTA, or NPU;
    - add a small amount of the retrieval score;
    - sort highest score first.
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
    """Split chunk text into reusable facts.

    Rules:
    - blank lines are ignored;
    - Markdown table rows stay as one fact so labels and values stay together;
    - non-table lines are split on sentence punctuation or newlines;
    - leading bullets and dashes are stripped from the final facts.
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
    """Return tokens that matter for fallback ranking.

    Processing:
    - tokenize text with the shared embedding tokenizer;
    - remove generic stopwords, ``ecu``, and standalone numbers;
    - add domain expansion terms when trigger words are present.
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
    """Add small ranking boosts for common ECU wording mismatches.

    Examples:
    - ``thermal tolerance`` in the question should find ``operating temperature``;
    - ``remote update`` should find ``OTA`` or ``firmware``;
    - ``edge inference`` should find ``NPU`` or ``TOPS``.
    """

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
    """Return ``True`` when any target phrase appears in ``text``."""

    return any(needle in text for needle in needles)


# Prompt construction for the live model ------------------------------------


def _normalize_model_text(answer: str) -> str:
    """Collapse unusual whitespace from a provider response into normal spaces."""

    return " ".join(answer.replace(" ", " ").replace(" ", " ").split())


def _build_prompt(
    question: str,
    route: RouteDecision,
    retrieved_context: Sequence[RetrievalResult],
) -> str:
    """Build the prompt sent to DeepSeek.

    Prompt sections:
    - ``QUESTION``: original user question;
    - ``EVIDENCE_LINES``: top ranked rows/sentences for easy model focus;
    - ``CONTEXT``: full retrieved chunks with source filenames;
    - ``ANSWER``: marker where the model starts generating.
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
    """Pick the highest-signal facts to place at the top of the prompt.

    Scoring for each fact:
    - token overlap with the user question;
    - bonus for Markdown tables or bold spec labels;
    - small boost from the original retrieval score.
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
    """Return source filenames once, preserving first-seen order."""

    seen: list[str] = []
    for result in retrieved_context:
        source = result.chunk.metadata.get("source")
        if isinstance(source, str) and source not in seen:
            seen.append(source)
    return tuple(seen)


def _with_sources(answer: str, sources: tuple[str, ...]) -> str:
    """Append source filenames to fallback answers when sources exist."""

    if not sources:
        return answer
    return f"{answer} Sources: {', '.join(sources)}."
