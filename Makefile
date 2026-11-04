# Thin wrapper: every target delegates to tasks.py (works without make on Windows too).
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
CONFIG ?= configs/mmotu_v1.yaml

.PHONY: setup dev api web worker test e2e lint data-audit verify-mirror fetch-models train evaluate bundle report check-no-data provider-status

setup dev api web worker test e2e lint data-audit verify-mirror fetch-models check-no-data provider-status:
	$(PY) tasks.py $@

train evaluate bundle report:
	$(PY) tasks.py $@ $(CONFIG)
