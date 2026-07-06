#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
RUN_LIVE_EVAL="${RUN_LIVE_EVAL:-0}"

pick_python() {
  local candidates=()
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  candidates+=(python3.13 python3.12 python3.11)

  for candidate in "${candidates[@]}"; do
    command -v "${candidate}" >/dev/null 2>&1 || continue
    if "${candidate}" - <<'PY' >/dev/null 2>&1
import sys

raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
    then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python)" || {
  echo "ERROR: Python >= 3.11 is required, but no compatible interpreter was found." >&2
  echo "Recommended with uv: uv venv --python 3.11 .venv" >&2
  echo "Or with Homebrew: brew install python@3.11" >&2
  echo "Or with pyenv: pyenv install 3.11.9 && pyenv local 3.11.9" >&2
  exit 1
}

cd "${ROOT_DIR}"

echo "== ME Engineering Assistant checks =="
echo "Repository: ${ROOT_DIR}"
echo "Virtual environment: ${VENV_DIR}"
echo "Python interpreter: ${PYTHON_BIN}"
echo

echo "== Create virtual environment =="
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

echo "== Install package and dependencies =="
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'

echo "== Import check =="
python -c 'import me_agent; print("me_agent", me_agent.__version__)'

echo "== Unit tests =="
python -m pytest -q -p no:cacheprovider

echo "== Static analysis =="
python -m pylint src/me_agent

echo "== Bytecode compile check =="
python -m compileall -q src tests scripts

echo "== Offline evaluation smoke test =="
ME_AGENT_API_KEY_ENV_VAR=ME_AGENT_MISSING_API_KEY python scripts/run_eval.py \
  --eval-path data/eval/test-questions.csv \
  --output reports/eval_results_offline.json \
  --title "ME Agent Offline Evaluation"

if [[ "${RUN_LIVE_EVAL}" == "1" ]]; then
  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: RUN_LIVE_EVAL=1 requires DEEPSEEK_API_KEY." >&2
    exit 1
  fi

  echo "== Live default evaluation =="
  python scripts/run_eval.py \
    --eval-path data/eval/test-questions.csv \
    --output reports/eval_results.json \
    --html-report reports/eval_report.html \
    --title "ME Agent Evaluation Report"

  echo "== Live stress evaluation =="
  python scripts/run_eval.py \
    --eval-path data/eval/stress-questions.csv \
    --output reports/stress_eval_results.json \
    --html-report reports/stress_eval_report.html \
    --title "ME Agent Stress Evaluation Report"

  echo "== Live evaluation summary =="
  python - <<'PY'
import json
from pathlib import Path

for path in [Path("reports/eval_results.json"), Path("reports/stress_eval_results.json")]:
    payload = json.loads(path.read_text())
    summary = payload.get("summary", payload)
    print(
        f"{path}: {summary['passed_cases']}/{summary['total_cases']} "
        f"accuracy={summary['accuracy']} used_llm_rate={summary.get('used_llm_rate')}"
    )
PY
else
  echo "== Live evaluation skipped =="
  echo "Set RUN_LIVE_EVAL=1 and DEEPSEEK_API_KEY to run live model evaluation."
fi

echo
echo "All checks completed."
