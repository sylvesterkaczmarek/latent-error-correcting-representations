# Reproducibility

The reference benchmark is designed for CPU execution.

## Reference command

```bash
python -m experiments.run_all --seeds 7 17 29 41 53 --epochs 5 --out results
```

The checked reference suite uses five fixed seeds and regenerates the JSON summaries and figures in `results/`.

## Controls

- explicit Python, NumPy, and PyTorch seeds,
- deterministic synthetic data generation,
- deterministic PyTorch algorithms where available,
- fixed train/test generation per seed,
- exact corruption budgets,
- deterministic adversarial single-bit search,
- deterministic nearest-valid-codeword drift construction,
- same-seed regression test,
- GitHub Actions smoke experiment.

## Environment

The code targets Python 3.10+ and PyTorch 2.x. The default experiments do not require a GPU.
