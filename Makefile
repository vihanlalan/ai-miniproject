# SynthDx task runner.  On Windows without `make`, use the equivalent:
#   python tasks.py <target>
PYTHON ?= python

.PHONY: install test test-fast train experiments report demo all clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest --cov=synthdx --cov-report=term-missing

test-fast:
	$(PYTHON) -m pytest -m "not slow"

train:
	$(PYTHON) -m synthdx.experiments.train

experiments:
	$(PYTHON) -m synthdx.experiments.run

report:
	$(PYTHON) -m synthdx.experiments.report

demo:
	$(PYTHON) -m streamlit run synthdx/app/main.py

all: train experiments report test

clean:
	rm -rf .pytest_cache .coverage htmlcov
