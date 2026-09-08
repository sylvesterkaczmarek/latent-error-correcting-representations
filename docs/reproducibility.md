# Reproducibility

The reference benchmark is designed for CPU execution.

## Reference command

```bash
python -m experiments.run_all --seeds 7 17 29 41 53 --epochs 5 --out results
```

The checked reference suite uses five fixed seeds and regenerates the JSON summaries and figures in `results/`.

Use a separate `--out` directory when comparing a new run with the checked reference
results. The same `python -m experiments...` commands work from a source checkout
after installation and from a wheel installation outside the checkout.

The command-line options are `--seeds`, `--epochs` and `--out`. Seeds must be
distinct integers from 0 through 4294967295, and epochs must be a positive integer.
The complete argument list is checked before any experiment starts or output is
written. Repeating a seed would repeat the same experiment and cannot be counted
as an independent run.

`configs/reference.yaml` records the reference settings for inspection. The
experiment commands do not load it; editing it does not change a run. The suite settings are passed explicitly by `run_seed` in `experiment.py`.
Lower-level API defaults are defined in `data.py`, `model.py` and `training.py`.
Corruption budgets and methods are fixed in `experiment.py` and `codes.py`.

## Controls

New result files retain the original metric paths. Those metrics measure
preservation of the encoder's predicted message. The additional `end_to_end`
measurements compare the decoded message with ground truth, including any encoder
mistakes. Each run and summary includes `metadata` recording the configuration,
metric references and software environment. Aggregation rejects repeated seeds or
incompatible run metadata.

The complete suite uses paths such as
`random_corruption.hamming74_repair.1.task_accuracy` for integrity and
`end_to_end.random_corruption.hamming74_repair.1.task_accuracy` for ground-truth
accuracy. Individual experiment summaries retain method names at the top level,
with `seeds`, `metadata` and `end_to_end` alongside them. For example, the corruption
sweep retains `hamming74_repair.1.task_accuracy` and adds
`end_to_end.hamming74_repair.1.task_accuracy`. Their per-seed files remain complete
run records containing all three challenges.

- explicit Python, NumPy, and PyTorch seeds,
- deterministic synthetic data generation,
- deterministic PyTorch algorithms where available,
- fixed train/test generation per seed,
- exact corruption budgets,
- deterministic adversarial single-bit search,
- deterministic nearest-valid-codeword drift construction,
- same-seed regression test,
- GitHub Actions smoke experiment.
- wheel build and installed-command smoke run outside the source checkout.

## Environment

The code targets Python 3.10+ and PyTorch 2.x. The default experiments do not require a GPU.

The reference shell script and CI use one OpenMP/MKL thread for this small CPU
workload. For the same setting when invoking a module directly, prefix the command
with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. The actual PyTorch thread count is
recorded in each run's metadata.

CI exercises Python 3.10, 3.11, 3.12 and 3.13 with CPU PyTorch. Dependency versions
are bounded rather than fully pinned, so identical seeds do not establish
bit-for-bit reproducibility across different PyTorch versions or platforms; see
the [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).

New seed files and full summaries include schema version 2 metadata with settings, software versions, thread count and metric references. `summarize` accepts legacy run records together, but rejects mixing legacy and new records or combining different configurations or recorded environments. A single-run standard deviation is stored as zero by convention; it does not estimate variability across seeds.
