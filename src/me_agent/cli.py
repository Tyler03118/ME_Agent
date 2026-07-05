"""Console entry points for local ME Engineering Assistant workflows."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import replace
import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd

from me_agent.core.config import AgentConfig
from me_agent.evaluation import (
    evaluate_cases,
    load_eval_payload,
    load_evaluation_cases,
    log_evaluation_to_mlflow,
    write_html_report,
    write_evaluation_results,
)
from me_agent.workflow.graph import EngineeringAssistant
from me_agent.tracking.mlflow_logging import configure_mlflow_tracking, log_assistant_model


def ask_cli() -> None:
    """Answer a single question from command-line arguments."""

    if len(sys.argv) < 2:
        raise SystemExit('Usage: me-agent "question"')
    assistant = EngineeringAssistant.from_config()
    response = assistant.ask(" ".join(sys.argv[1:]))
    print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))


def run_eval_cli() -> None:
    """Run the golden evaluation set and log metrics to MLflow."""

    args = _eval_parser().parse_args()
    configure_mlflow_tracking()
    config = AgentConfig.from_env()
    if args.eval_path is not None:
        config = replace(config, eval_path=args.eval_path)
    cases = load_evaluation_cases(config.eval_path)
    assistant = EngineeringAssistant.from_config(config)
    with mlflow.start_run(run_name="me-agent-evaluation"):
        results = evaluate_cases(assistant, cases)
        summary = write_evaluation_results(results, args.output)
        log_evaluation_to_mlflow(summary, args.output)
        if args.html_report is not None:
            payload = load_eval_payload(args.output)
            write_html_report(
                payload,
                args.html_report,
                title=args.title,
                markdown_report_path=args.markdown_report,
            )
            mlflow.log_artifact(str(args.html_report))
    output = {**summary, "eval_results_path": str(args.output)}
    if args.html_report is not None:
        output["html_report_path"] = str(args.html_report)
    print(json.dumps(output, indent=2, ensure_ascii=False))


def render_eval_report_cli() -> None:
    """Render an existing evaluation JSON artifact as HTML."""

    args = _render_report_parser().parse_args()
    payload = load_eval_payload(args.input_json)
    path = write_html_report(
        payload,
        args.output_html,
        title=args.title,
        markdown_report_path=args.markdown_report,
    )
    print(path)


def log_model_cli() -> None:
    """Log the assistant as a self-contained MLflow pyfunc model."""

    log_assistant_model()


def load_model_cli() -> None:
    """Load the latest logged MLflow model and run a smoke prediction."""

    configure_mlflow_tracking()
    model_uri = os.getenv("ME_AGENT_MODEL_URI")
    if not model_uri:
        run_id_path = Path("outputs") / "latest_run_id.txt"
        if not run_id_path.exists():
            raise FileNotFoundError("Set ME_AGENT_MODEL_URI or run me-agent-log-model first.")
        run_id = run_id_path.read_text(encoding="utf-8").strip()
        model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    print(model.predict(pd.DataFrame({"question": ["How much RAM does the ECU-850 have?"]})))


def _eval_parser() -> ArgumentParser:
    """Build the CLI parser for evaluation runs."""

    parser = ArgumentParser(description="Run ME Agent evaluation and log MLflow metrics.")
    parser.add_argument(
        "--eval-path",
        type=Path,
        help="CSV evaluation set. Defaults to ME_AGENT_EVAL_PATH or the challenge CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results.json"),
        help="JSON artifact path. Defaults to eval_results.json.",
    )
    parser.add_argument(
        "--html-report",
        type=Path,
        help="Optional self-contained HTML report path.",
    )
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
    return parser


def _render_report_parser() -> ArgumentParser:
    """Build the CLI parser for rendering existing eval JSON artifacts."""

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
    return parser
