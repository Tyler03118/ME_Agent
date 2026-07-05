"""Run the local ECU evaluation set and log metrics to MLflow."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import mlflow

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
from me_agent.tracking.mlflow_logging import configure_mlflow_tracking


def _parse_args():
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
    return parser.parse_args()


def main() -> None:
    """Evaluate the assistant against configured golden questions."""

    args = _parse_args()
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


if __name__ == "__main__":
    main()
