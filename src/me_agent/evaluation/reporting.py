"""Self-contained HTML reports for evaluation JSON artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


def load_eval_payload(path: str | Path) -> dict[str, Any]:
    """Load an evaluation JSON artifact produced by ``write_evaluation_results``."""

    payload_path = Path(path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation payload must be a JSON object: {payload_path}")
    if not isinstance(payload.get("summary"), dict) or not isinstance(
        payload.get("results"), list
    ):
        raise ValueError(
            "Evaluation payload must contain a summary object and results list: "
            f"{payload_path}"
        )
    return payload


def write_html_report(
    payload: dict[str, Any],
    output_path: str | Path,
    *,
    title: str = "ME Agent Evaluation Report",
    markdown_report_path: str | Path | None = None,
) -> Path:
    """Render and write a self-contained HTML report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown_text = _read_optional_markdown(markdown_report_path)
    html = render_html_report(payload, title=title, markdown_text=markdown_text)
    path.write_text(html, encoding="utf-8")
    return path


def render_html_report(
    payload: dict[str, Any],
    *,
    title: str = "ME Agent Evaluation Report",
    markdown_text: str | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Render an evaluation payload into a standalone HTML document."""

    summary = payload.get("summary", {})
    results = payload.get("results", [])
    timestamp = generated_at or datetime.now(timezone.utc)
    category_rows = _render_category_rows(results)
    case_cards = "\n".join(_render_case_card(case) for case in results)
    failure_rows = _render_attention_rows(
        [case for case in results if not case.get("passed")],
        empty_message="No failed cases.",
    )
    review_rows = _render_attention_rows(
        [case for case in results if case.get("needs_human_review")],
        empty_message="No cases flagged for human review.",
    )
    markdown_section = _render_markdown_section(markdown_text)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
{_CSS}
  </style>
</head>
<body>
  <header class="hero">
    <div>
      <p class="eyebrow">Evaluation Report</p>
      <h1>{escape(title)}</h1>
      <p class="subtitle">Generated {escape(timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"))}</p>
    </div>
    <div class="status-pill {_status_class(summary)}">{_status_label(summary)}</div>
  </header>

  <main>
    <section class="metric-grid" aria-label="Summary metrics">
      {_metric_card("Accuracy", _percent(summary.get("accuracy")), summary)}
      {_metric_card("Passed", _count_ratio(summary, "passed_cases", "total_cases"), summary)}
      {_metric_card("used_llm_rate", _percent(summary.get("used_llm_rate")), summary)}
      {_metric_card("Avg latency", _seconds(summary.get("avg_latency_seconds")), summary)}
      {_metric_card("Max latency", _seconds(summary.get("max_latency_seconds")), summary)}
      {_metric_card("Fallback cases", _plain(summary.get("fallback_cases", 0)), summary)}
      {_metric_card("Fact recall", _percent(summary.get("mean_required_fact_recall")), summary)}
      {_metric_card("Source match", _percent(summary.get("mean_source_match")), summary)}
      {_metric_card("Route match", _percent(summary.get("mean_route_match")), summary)}
      {_metric_card("Forbidden facts", _plain(summary.get("forbidden_fact_violations", 0)), summary)}
    </section>

    <section class="panel">
      <div class="section-heading">
        <h2>Score Breakdown</h2>
        <p>Core quality, routing, source, and latency signals from the JSON artifact.</p>
      </div>
      <div class="bars">
        {_bar("Accuracy", summary.get("accuracy"))}
        {_bar("Semantic similarity", summary.get("mean_semantic_similarity"))}
        {_bar("Token coverage", summary.get("mean_token_coverage"))}
        {_bar("Required fact recall", summary.get("mean_required_fact_recall"))}
        {_bar("Source match", summary.get("mean_source_match"))}
        {_bar("Route match", summary.get("mean_route_match"))}
      </div>
    </section>

    <section class="panel">
      <div class="section-heading">
        <h2>Category Results</h2>
        <p>Pass rate and average combined score grouped by evaluation category.</p>
      </div>
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th>Pass Rate</th>
            <th>Cases</th>
            <th>Avg Score</th>
            <th>Avg Latency</th>
          </tr>
        </thead>
        <tbody>{category_rows}</tbody>
      </table>
    </section>

    <section class="attention-grid">
      <div class="panel">
        <div class="section-heading">
          <h2>Failed Cases</h2>
          <p>Cases below the configured pass threshold.</p>
        </div>
        {failure_rows}
      </div>
      <div class="panel">
        <div class="section-heading">
          <h2>Human Review</h2>
          <p>Low-confidence or borderline cases flagged by the workflow.</p>
        </div>
        {review_rows}
      </div>
    </section>

    <section class="panel">
      <div class="section-heading">
        <h2>Case Details</h2>
        <p>Question-level evidence for reviewing answers, sources, routes, and scores.</p>
      </div>
      <div class="case-list">{case_cards}</div>
    </section>

    {markdown_section}
  </main>
</body>
</html>
"""


def _read_optional_markdown(path: str | Path | None) -> str | None:
    """Load optional Markdown appendix text for embedding in the HTML report."""

    if path is None:
        return None
    markdown_path = Path(path)
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown report does not exist: {markdown_path}")
    return markdown_path.read_text(encoding="utf-8")


def _metric_card(label: str, value: str, summary: dict[str, Any]) -> str:
    """Render one top-level summary metric card."""

    return f"""
      <article class="metric-card">
        <span>{escape(label)}</span>
        <strong>{escape(value)}</strong>
        {_mini_status(summary)}
      </article>"""


def _mini_status(summary: dict[str, Any]) -> str:
    """Render the small pass/fail status label shown inside metric cards."""

    total = int(summary.get("total_cases", 0) or 0)
    failed = int(summary.get("failed_cases_count", 0) or 0)
    if total == 0:
        return '<small class="muted">no cases</small>'
    if failed == 0:
        return '<small class="ok">all passing</small>'
    return f'<small class="warn">{failed} failed</small>'


def _render_category_rows(results: list[dict[str, Any]]) -> str:
    """Render category-level pass-rate rows for the summary table."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in results:
        groups[str(case.get("category") or "uncategorized")].append(case)
    if not groups:
        return '<tr><td colspan="5" class="empty">No case results.</td></tr>'
    rows = []
    for category in sorted(groups):
        cases = groups[category]
        total = len(cases)
        passed = sum(1 for case in cases if case.get("passed"))
        avg_score = _mean(case.get("combined_score") for case in cases)
        avg_latency = _mean(case.get("latency_seconds") for case in cases)
        rows.append(
            "<tr>"
            f"<td>{escape(category)}</td>"
            f"<td>{escape(_percent(passed / total if total else 0.0))}</td>"
            f"<td>{passed}/{total}</td>"
            f"<td>{avg_score:.4f}</td>"
            f"<td>{avg_latency:.4f}s</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _render_attention_rows(cases: list[dict[str, Any]], *, empty_message: str) -> str:
    """Render failed-case or human-review rows."""

    if not cases:
        return f'<p class="empty">{escape(empty_message)}</p>'
    rows = []
    for case in cases:
        rows.append(
            '<div class="attention-row">'
            f'<strong>{escape(str(case.get("question_id", "")))}</strong>'
            f'<span>{escape(str(case.get("category", "")))}</span>'
            f'<p>{escape(str(case.get("question", "")))}</p>'
            f'<small>score {escape(_score(case.get("combined_score")))} | '
            f'route {escape(str(case.get("route_category", "")))}</small>'
            "</div>"
        )
    return "\n".join(rows)


def _render_case_card(case: dict[str, Any]) -> str:
    """Render one expandable per-question detail card."""

    status = "pass" if case.get("passed") else "fail"
    sources = ", ".join(str(source) for source in case.get("sources", [])) or "none"
    expected_sources = (
        ", ".join(str(source) for source in case.get("expected_sources", [])) or "not specified"
    )
    return f"""
        <details class="case-card {status}">
          <summary>
            <span class="case-id">{escape(str(case.get("question_id", "")))}</span>
            <span>{escape(str(case.get("category", "")))}</span>
            <strong>{escape(str(case.get("question", "")))}</strong>
            <span class="case-score">{escape(_score(case.get("combined_score")))}</span>
          </summary>
          <div class="case-body">
            <div class="answer-grid">
              <div>
                <h3>Expected</h3>
                <p>{escape(str(case.get("expected_answer", "")))}</p>
              </div>
              <div>
                <h3>Actual</h3>
                <p>{escape(str(case.get("actual_answer", "")))}</p>
              </div>
            </div>
            <dl class="case-metrics">
              {_detail("Passed", str(case.get("passed")))}
              {_detail("Combined score", _score(case.get("combined_score")))}
              {_detail("Semantic similarity", _score(case.get("semantic_similarity")))}
              {_detail("Token coverage", _score(case.get("token_coverage")))}
              {_detail("Required fact recall", _score(case.get("required_fact_recall")))}
              {_detail("Source match", _score(case.get("source_match")))}
              {_detail("Route match", _score(case.get("route_match")))}
              {_detail("Forbidden violations", str(case.get("forbidden_fact_violations", 0)))}
              {_detail("Route", str(case.get("route_category", "")))}
              {_detail("Expected route", str(case.get("expected_route", "") or "not specified"))}
              {_detail("Sources", sources)}
              {_detail("Expected sources", expected_sources)}
              {_detail("Latency", _seconds(case.get("latency_seconds")))}
              {_detail("Used LLM", str(case.get("used_llm")))}
              {_detail("Fallback reason", str(case.get("fallback_reason") or "none"))}
              {_detail("Human review", str(case.get("needs_human_review")))}
            </dl>
          </div>
        </details>"""


def _detail(label: str, value: str) -> str:
    """Render one definition-list key/value pair."""

    return f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>"


def _render_markdown_section(markdown_text: str | None) -> str:
    """Render the optional Markdown appendix section."""

    if markdown_text is None:
        return ""
    return f"""
    <section class="panel">
      <div class="section-heading">
        <h2>Markdown Report</h2>
        <p>Companion report embedded for one-file review.</p>
      </div>
      <pre class="markdown-report">{escape(markdown_text)}</pre>
    </section>"""


def _bar(label: str, value: Any) -> str:
    """Render one percentage bar in the score breakdown section."""

    numeric = _float(value)
    width = max(0.0, min(1.0, numeric)) * 100
    return f"""
        <div class="bar-row">
          <div class="bar-label"><span>{escape(label)}</span><strong>{escape(_percent(numeric))}</strong></div>
          <div class="bar-track"><div class="bar-fill" style="width: {width:.2f}%"></div></div>
        </div>"""


def _status_class(summary: dict[str, Any]) -> str:
    """Return the CSS class for the report-level pass/fail status pill."""

    return "good" if int(summary.get("failed_cases_count", 0) or 0) == 0 else "needs-work"


def _status_label(summary: dict[str, Any]) -> str:
    """Return the text for the report-level pass/fail status pill."""

    failed = int(summary.get("failed_cases_count", 0) or 0)
    return "All passing" if failed == 0 else f"{failed} failed"


def _count_ratio(summary: dict[str, Any], numerator: str, denominator: str) -> str:
    """Format two integer summary fields as a ratio."""

    return f"{int(summary.get(numerator, 0) or 0)}/{int(summary.get(denominator, 0) or 0)}"


def _percent(value: Any) -> str:
    """Format a 0-to-1 score as a percentage string."""

    return f"{_float(value) * 100:.1f}%"


def _seconds(value: Any) -> str:
    """Format a numeric duration in seconds."""

    return f"{_float(value):.4f}s"


def _score(value: Any) -> str:
    """Format a metric score to four decimal places."""

    return f"{_float(value):.4f}"


def _plain(value: Any) -> str:
    """Convert a value to display text without numeric formatting."""

    return str(value)


def _float(value: Any) -> float:
    """Coerce a value to float, using 0.0 for missing or invalid values."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values) -> float:
    """Return the arithmetic mean for numeric-ish values."""

    collected = [_float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


_CSS = """
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #172026;
  --muted: #68737d;
  --line: #dfe4ea;
  --accent: #2563eb;
  --good: #0f766e;
  --good-soft: #ccfbf1;
  --warn: #b45309;
  --bad: #b91c1c;
  --bad-soft: #fee2e2;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
  line-height: 1.5;
}

.hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  padding: 40px clamp(20px, 5vw, 72px) 28px;
  border-bottom: 1px solid var(--line);
  background: #ffffff;
}

.eyebrow {
  margin: 0 0 8px;
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1 {
  margin-bottom: 8px;
  font-size: clamp(28px, 4vw, 44px);
  line-height: 1.08;
}

h2 {
  margin-bottom: 4px;
  font-size: 20px;
}

h3 {
  margin-bottom: 8px;
  font-size: 14px;
}

.subtitle,
.section-heading p,
.muted {
  color: var(--muted);
}

.status-pill {
  flex: 0 0 auto;
  border-radius: 999px;
  padding: 9px 14px;
  font-weight: 700;
}

.status-pill.good {
  background: var(--good-soft);
  color: var(--good);
}

.status-pill.needs-work {
  background: var(--bad-soft);
  color: var(--bad);
}

main {
  display: grid;
  gap: 20px;
  max-width: 1180px;
  margin: 0 auto;
  padding: 24px clamp(16px, 4vw, 40px) 56px;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
}

.metric-card,
.panel,
.case-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}

.metric-card {
  display: grid;
  gap: 6px;
  padding: 16px;
}

.metric-card span {
  color: var(--muted);
  font-size: 13px;
}

.metric-card strong {
  font-size: 24px;
  line-height: 1;
}

.metric-card small {
  font-size: 12px;
}

.ok {
  color: var(--good);
}

.warn {
  color: var(--warn);
}

.panel {
  padding: 20px;
}

.section-heading {
  margin-bottom: 16px;
}

.bars {
  display: grid;
  gap: 14px;
}

.bar-label {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 6px;
  font-size: 13px;
}

.bar-track {
  height: 10px;
  overflow: hidden;
  border-radius: 999px;
  background: #edf1f5;
}

.bar-fill {
  height: 100%;
  border-radius: inherit;
  background: var(--accent);
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

th,
td {
  padding: 10px 8px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}

th {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

.attention-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
}

.attention-row {
  padding: 12px 0;
  border-top: 1px solid var(--line);
}

.attention-row:first-of-type {
  border-top: 0;
}

.attention-row strong,
.attention-row span {
  display: inline-block;
  margin-right: 8px;
}

.attention-row span,
.attention-row small {
  color: var(--muted);
}

.attention-row p {
  margin: 6px 0;
}

.empty {
  color: var(--muted);
}

.case-list {
  display: grid;
  gap: 10px;
}

.case-card {
  overflow: hidden;
}

.case-card.fail {
  border-color: #fca5a5;
}

.case-card summary {
  display: grid;
  grid-template-columns: minmax(60px, auto) minmax(110px, 170px) 1fr auto;
  gap: 12px;
  align-items: center;
  padding: 14px 16px;
  cursor: pointer;
}

.case-card summary strong {
  font-weight: 600;
}

.case-id,
.case-score {
  font-variant-numeric: tabular-nums;
  font-weight: 700;
}

.case-score {
  color: var(--accent);
}

.case-body {
  padding: 0 16px 16px;
  border-top: 1px solid var(--line);
}

.answer-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
  padding: 16px 0;
}

.answer-grid p {
  white-space: pre-wrap;
}

.case-metrics {
  display: grid;
  grid-template-columns: minmax(140px, 190px) 1fr;
  gap: 8px 12px;
  margin: 0;
  font-size: 13px;
}

.case-metrics dt {
  color: var(--muted);
  font-weight: 700;
}

.case-metrics dd {
  margin: 0;
  overflow-wrap: anywhere;
}

.markdown-report {
  max-height: 640px;
  overflow: auto;
  margin: 0;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfcfe;
  color: #24313a;
  font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  white-space: pre-wrap;
}

@media (max-width: 720px) {
  .hero {
    display: block;
  }

  .status-pill {
    display: inline-block;
    margin-top: 16px;
  }

  .case-card summary {
    grid-template-columns: 1fr;
  }
}
"""
