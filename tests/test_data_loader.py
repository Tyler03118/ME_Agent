from pathlib import Path

from me_agent.data_loader import MarkdownManualLoader


def test_loader_reads_markdown_manual_with_metadata(tmp_path: Path) -> None:
    manual_path = tmp_path / "ECU-700_Series_Manual.md"
    manual_path.write_text(
        "# Product Manual: ECU-700 Series\n\n"
        "Document ID: DOC-700-REV1.2\n\n"
        "This document covers the ECU-750.\n",
        encoding="utf-8",
    )

    documents = MarkdownManualLoader(tmp_path).load()

    assert len(documents) == 1
    document = documents[0]
    assert document.content.startswith("# Product Manual")
    assert document.metadata["source"] == "ECU-700_Series_Manual.md"
    assert document.metadata["document_id"] == "DOC-700-REV1.2"
    assert document.metadata["product_family"] == "ECU-700"
    assert document.metadata["model"] == "ECU-750"


def test_loader_raises_for_missing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    try:
        MarkdownManualLoader(missing_dir).load()
    except FileNotFoundError as exc:
        assert "Manual directory does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing manual directory")
