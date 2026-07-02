"""Run the local ECU evaluation set."""

from __future__ import annotations

import json

from me_agent.config import AgentConfig
from me_agent.evaluation import evaluate_cases, load_evaluation_cases, write_evaluation_results
from me_agent.graph import EngineeringAssistant


def main() -> None:
    """Evaluate the assistant against configured golden questions."""

    config = AgentConfig.from_env()
    cases = load_evaluation_cases(config.eval_path)
    assistant = EngineeringAssistant.from_config(config)
    results = evaluate_cases(assistant, cases)
    summary = write_evaluation_results(results, "eval_results.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
