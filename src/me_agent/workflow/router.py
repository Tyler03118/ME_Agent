"""Deterministic router for ECU-manual and out-of-domain questions."""

from __future__ import annotations

import re

from me_agent.core.schemas import RouteDecision

ECU_700_SOURCE = "ECU-700_Series_Manual.md"
ECU_800_BASE_SOURCE = "ECU-800_Series_Base.md"
ECU_800_PLUS_SOURCE = "ECU-800_Series_Plus.md"
ALL_SOURCES = (ECU_700_SOURCE, ECU_800_BASE_SOURCE, ECU_800_PLUS_SOURCE)
IN_SCOPE_TERMS = (
    "ecu", "can", "ota", "over-the-air", "npu", "ram", "memory", "storage", "temperature",
    "power", "lpddr", "emmc", "driver", "me-driver", "cortex", "flash",
)


class DeterministicRouter:
    """Route queries with stable rules and an explicit out-of-scope category."""

    def route(self, query: str) -> RouteDecision:
        """Classify a query and infer required source files."""

        normalized = query.strip().lower()
        if not normalized:
            raise ValueError("query must not be empty")
        if not _is_ecu_manual_query(normalized):
            return RouteDecision("general", (), "Question is outside the ECU manual scope.")

        sources = self._required_sources(normalized)
        category = "general"
        required_sources = sources
        rationale = "No specific ECU route matched."

        if _contains_any(normalized, ("enable", "configuration", "driver command", "npu")):
            category = (
                "configuration"
                if "enable" in normalized or "driver" in normalized
                else "ecu_850b_lookup"
            )
            required_sources = sources or {ECU_800_PLUS_SOURCE}
            rationale = "NPU and driver-command queries require the ECU-800 plus addendum."
        elif _contains_any(normalized, ("ota", "over-the-air")):
            category = "feature_availability"
            required_sources = set(ALL_SOURCES)
            rationale = "Feature availability requires positive and negative evidence."
        elif _contains_any(
            normalized,
            (
                "compare", "comparison", "difference", "differences", "across", "all models",
                "which ecu", "harshest", "storage capacity",
            ),
        ):
            category = "comparison"
            required_sources = sources or set(ALL_SOURCES)
            rationale = "Comparison queries need coverage across all relevant models."
        elif re.search(r"\becu-850b\b", normalized):
            category = "ecu_850b_lookup"
            required_sources = sources or {ECU_800_BASE_SOURCE, ECU_800_PLUS_SOURCE}
            rationale = "ECU-850b details live in the plus addendum and inherit ECU-850 features."
        elif re.search(r"\becu-850\b", normalized):
            category = "ecu_800_lookup"
            required_sources = sources or {ECU_800_BASE_SOURCE}
            rationale = "ECU-850 details live in the ECU-800 base manual."
        elif re.search(r"\becu-750\b", normalized):
            category = "ecu_700_lookup"
            required_sources = sources or {ECU_700_SOURCE}
            rationale = "ECU-750 details live in the ECU-700 manual."

        return RouteDecision(category, _ordered_sources(required_sources), rationale)

    @staticmethod
    def _required_sources(normalized_query: str) -> set[str]:
        sources: set[str] = set()
        if re.search(r"\becu-750\b|\becu-700\b", normalized_query):
            sources.add(ECU_700_SOURCE)
        if re.search(r"\becu-850\b|\becu-800\b", normalized_query):
            sources.add(ECU_800_BASE_SOURCE)
        if re.search(r"\becu-850b\b", normalized_query):
            sources.add(ECU_800_PLUS_SOURCE)
        if _contains_any(normalized_query, ("all models", "storage", "harshest", "which ecu")):
            sources.update(ALL_SOURCES)
        if "temperature" in normalized_query and not sources:
            sources.update(ALL_SOURCES)
        return sources


def _is_ecu_manual_query(text: str) -> bool:
    return _contains_any(text, IN_SCOPE_TERMS) or bool(re.search(r"\becu[- ]?\d+\w*\b", text))


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _ordered_sources(sources: set[str]) -> tuple[str, ...]:
    return tuple(source for source in ALL_SOURCES if source in sources)
