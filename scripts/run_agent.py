"""Run the ME Engineering Assistant for a single question."""

from __future__ import annotations

import argparse
import json

from me_agent.graph import EngineeringAssistant


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Ask a question over ECU manuals.")
    parser.add_argument("question")
    args = parser.parse_args()

    assistant = EngineeringAssistant.from_config()
    response = assistant.ask(args.question)
    print(json.dumps(response.to_dict(), indent=2))


if __name__ == "__main__":
    main()
