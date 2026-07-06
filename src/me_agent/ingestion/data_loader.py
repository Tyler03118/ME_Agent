"""Markdown manual loading and metadata extraction."""

from __future__ import annotations

import re
from pathlib import Path

from me_agent.core.schemas import ManualDocument


DOCUMENT_ID_PATTERN = re.compile(r"Document ID:\s*([A-Z0-9_.-]+)", re.IGNORECASE)
ECU_MODEL_PATTERN = re.compile(r"\bECU-\d{3}[A-Za-z]?\b")


class MarkdownManualLoader:
    """Load Markdown ECU manuals from a local directory."""

    def __init__(self, manual_dir: str | Path, pattern: str = "*.md") -> None:
        """Store the manual directory and glob pattern used by ``load``."""

        self.manual_dir = Path(manual_dir)
        self.pattern = pattern

    def load(self) -> list[ManualDocument]:
        """Read Markdown files in deterministic order and attach metadata."""

        if not self.manual_dir.exists():
            raise FileNotFoundError(f"Manual directory does not exist: {self.manual_dir}")
        if not self.manual_dir.is_dir():
            raise NotADirectoryError(f"Manual path is not a directory: {self.manual_dir}")

        documents: list[ManualDocument] = []
        for path in sorted(self.manual_dir.glob(self.pattern)):
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                documents.append(
                    ManualDocument(content=content, metadata=self._metadata(path, content))
                )
        return documents

    @staticmethod
    def _metadata(path: Path, content: str) -> dict[str, str | None]:
        """Build citation, routing, and reporting metadata for one manual.

        The filename remains the primary citation key. Product family and model
        are best-effort labels used for filtering and diagnostics.
        """

        text = f"{path.name}\n{content}"
        document_id_match = DOCUMENT_ID_PATTERN.search(content)
        product_family = _extract_product_family(text)
        return {
            "source": path.name,
            "document_id": document_id_match.group(1) if document_id_match else None,
            "product_family": product_family,
            "model": _extract_model(text, product_family),
        }


def _extract_product_family(text: str) -> str | None:
    """Return the ECU product family mentioned in file name or content."""

    for family in ("ECU-700", "ECU-800"):
        if family.lower() in text.lower():
            return family
    return None


def _extract_model(text: str, product_family: str | None) -> str | None:
    """Return the most specific ECU model identifier found in the text.

    Specific models such as ECU-850b are preferred over family names such as
    ECU-800 because model-level routing is more selective.
    """

    matches = ECU_MODEL_PATTERN.findall(text)
    for model in matches:
        normalized = model.upper().replace("B", "b")
        if normalized != product_family:
            return normalized
    return matches[0] if matches else None
