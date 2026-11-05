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
	python -m experiments.run_all --out runs/reference

.PHONY: demo selective verify-evidence
demo:
	python -m experiments.showcase --out runs/demo

selective:
	python -m experiments.selective_decoding run --out runs/selective

verify-evidence:
	python -m experiments.selective_decoding verify --out results/selective-v1
