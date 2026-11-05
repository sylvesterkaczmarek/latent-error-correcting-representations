# Independent verification record

## Scope and original candidate

The review began from a fresh checkout at
077e016372a338d55487dae85f2c97eb43cc0704 and a fresh Python environment. The
candidate passed all 307 existing tests before additional cases were written.
The base before implementation was 5c59cca89cd8ab5891d71502b988c4ac01e042a1.
This is a separate engineering review pass, not external certification.

A new 39-case review produced 34 failures and five passing controls. The expanded
59-case suite was independently replayed against the unchanged candidate: 47
failed and 12 passed. All 59 pass after repair, and the original external 39-case
file passes unchanged. The original implementation audit was also reproduced:
all 22 cases failed at the old base and pass after correction. Original logs,
probes and checkouts remain preserved in the local completion bundle.

## Findings and corrections

### Study plan validation

The verifier accepted duplicate declared seeds, duplicate methods or challenges,
invalid schema types and incompatible model/calibration settings when downstream
hashes and summaries were rebuilt. A duplicate seed could increase the apparent
number of independent runs. Validation now requires the complete supported
predeclared design, exact typed fields and unique integer seed identities before
checkpoint loading or aggregation. Historical source/runtime identities are
validated without incorrectly requiring them to equal the current installation.

### Trace and summary contracts

Recorded integer bits could be accepted after conversion from booleans or floats.
Malformed digest identifiers were also retained. Direct summary calls accepted
duplicate or missing cells, inconsistent integer partitions and rates, and
out-of-range encoder measurements. The returned seed list aliased the mutable
input plan. Validation now checks canonical trace bits, digest structure,
complete unique cells, integer denominators, outcome partitions and recomputed
rates. Undefined conditional accuracy remains null at zero coverage. Summaries
own their seed list and aggregate in declared-seed order.

### Checkpoint consistency

The earlier verifier checked checkpoint bytes against the manifest but did not
inspect them. A malformed or different checkpoint could pass when its file hash
was updated while the trace still named a different model. Restricted weights-only
CPU loading now checks the small benchmark checkpoint's schema, seed, plan,
parameter names, shapes, precision, finiteness and state digest against the trace.
The verifier does not use arbitrary checkpoint classes or unsafe fallback loading.
CPU random state is preserved while constructing a shape reference. File reads
are bounded and special entries in the evidence tree are rejected.

This verifies internal consistency, not provenance authenticity. An author able
to replace an entire coherent study can recompute its unkeyed hashes. Restricted
loading is not a general sandbox for untrusted external model files.

## Mathematical verification

A separately written integer-only oracle constructs the Hamming codebook by
solving parity constraints. Expected decoding uses nearest-codeword radius-one
search or majority decisions rather than the implementation's parity formulas.
It agrees with the implementation on all 4,752 possible received words across
six policies, all 76,032 message/mask cases, and 192 exact one/two-bit adversarial
choices on valid messages. The oracle is runnable as:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/independent_coding_oracle.py
```

Single-bit SECDED correction and double-bit abstention are preserved. Triple-bit
miscorrection and undetected wrong-valid-codeword substitution remain explicit.
Zero wrong acceptance with zero coverage is not successful task completion.

## Training and checkpoint reproduction

A fresh independent five-seed retraining reproduced all 19 archived study files
byte-for-byte in the recorded environment. Another run after the verifier repairs
retained all 240 policy/challenge cells, predictions, decisions, losses, counts,
statistics and model tensors. Provenance hashes changed only as expected for the
changed source and the checkpoint's bound plan. The original four-method runs
also reproduced the five baseline seed records exactly.

Separate saved-state inference used regenerated train/test data and direct
linear/tanh operations, without MessageEncoder.forward, on both sets of five
checkpoints. All 8,000 saved predictions matched. Neither this replay nor the
bootstrap increases the independent sample count beyond five trained seeds.

The original results/summary.json, configs/reference.yaml and all 19 files in
results/selective-v1 are unchanged. Current verification accepts these historical
artifacts under their recorded source identity. Cross-platform bitwise training
reproduction is not promised; the tests separately cover the supported runtimes.

## Release checks

The final local source suite passes 366 tests. A fresh non-editable package is
checked outside the source tree; runtime and experiment modules must match the
reviewed source. CI covers Linux Python 3.10-3.14, macOS, the recorded dependency
advisory audit and isolated installed-package verification. Exact completed job
results are available in the linked PR and release workflow, rather than inferred
from this document.

The GitHub-only release workflow repeats exact-commit CI, checks the package
proof and reviewed-main ancestry, refuses replacement of an existing version,
publishes the tested files and verifies public downloads. It does not upload to
PyPI or create a package-index publisher. The namespace and account-side publishing
configuration were not established for this research project.

## Reproduction and limitations

```bash
python -m pip install -e ".[dev]"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -m pytest -q
python -m ruff check src experiments scripts tests
python scripts/independent_coding_oracle.py
python -m experiments.selective_decoding verify --out results/selective-v1
python -m experiments.selective_decoding run --out /tmp/latent-release-replay-new
python -m experiments.showcase --out /tmp/latent-release-demo-new
python -m build
python -m twine check --strict dist/*
python scripts/verify_distribution.py dist/*.whl dist/*.tar.gz --report /tmp/latent-install-proof.json
```

Output directories must be absent. Routine verification recomputes trace outcomes
and checks model identity; it does not retrain or replay inference. This remains
a synthetic four-bit supervised bottleneck and classical coding experiment. It
provides no human-influence result, novel coding theorem, general model
self-monitoring guarantee or production safety certification.

References checked for the release:

- PyTorch restricted serialization: https://docs.pytorch.org/docs/stable/notes/serialization
- PyTorch reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
- AMD SECDED scope: https://docs.amd.com/r/en-US/am011-versal-acap-trm/Memory-Error-Detection-and-Correction

## Hosted runner allocation

The first final-review CI attempt completed its Linux Python matrix and dependency
audit, but the macos-latest job remained unassigned. That attempt was cancelled
before merge. The macOS job was pinned to the supported macos-15 ARM64 image
without removing or weakening any test. The final exact-head CI and release
workflow retain the full macOS suite and installed-package gates. Independent
local macOS installations on Python 3.13 and 3.14 also passed all 366 tests.
