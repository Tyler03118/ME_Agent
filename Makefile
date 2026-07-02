.PHONY: install test compile lint eval

install:
	pip install -e ".[dev]"

compile:
	python -m compileall src tests scripts

test:
	pytest -q

lint:
	pylint src/me_agent

eval:
	python scripts/run_eval.py
