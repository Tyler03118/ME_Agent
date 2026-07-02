# Evaluation Strategy

The evaluation set is a CSV of golden ECU questions with expected answers and
domain-specific criteria. Automated evaluation should track:

- Answer correctness against golden expectations.
- Required source coverage.
- Verifier support status.
- Human-review rate.
- Per-query latency.

For production validation, subject-matter experts should review sampled answers,
especially low-confidence cases, comparison questions, and configuration
commands.
