"""Render an existing evaluation JSON artifact as a self-contained HTML report."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from me_agent.evaluation import load_eval_payload, write_html_report


def _parse_args():
    parser = ArgumentParser(description="Render an ME Agent eval JSON artifact to HTML.")
    parser.add_argument("input_json", type=Path, help="Evaluation JSON artifact path.")
    parser.add_argument("output_html", type=Path, help="HTML report output path.")
    parser.add_argument(
        "--markdown-report",
        type=Path,
        help="Optional Markdown report to embed in the HTML report.",
    )
    parser.add_argument(
        "--title",
        default="ME Agent Evaluation Report",
        help="HTML report title.",
    )
    return parser.parse_args()


def main() -> None:
    """Render a saved eval artifact to HTML."""

    args = _parse_args()
    payload = load_eval_payload(args.input_json)
    path = write_html_report(
        payload,
        args.output_html,
        title=args.title,
        markdown_report_path=args.markdown_report,
    )
    print(path)


if __name__ == "__main__":
    main()
