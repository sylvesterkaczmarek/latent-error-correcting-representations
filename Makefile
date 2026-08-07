.PHONY: install test sweep adversarial drift all

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

sweep:
	python -m experiments.corruption_sweep

adversarial:
	python -m experiments.adversarial_flip

drift:
	python -m experiments.coherent_drift

all:
	python -m experiments.run_all
