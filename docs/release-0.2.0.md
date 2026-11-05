# Latent Error-Correcting Representations 0.2.0

This research release adds explicit selective decoding, complete trace evidence,
and independently tested plan, outcome and checkpoint consistency checks.

## Changes

The historical four coding methods are preserved. Extended Hamming(8,4) SECDED
and a Hamming detection-and-abstention control report accepted correct, accepted
wrong and abstained outcomes separately. The complete five-seed study contains
240 policy/challenge cells, checkpoints and per-example traces. The integer
reference check covers all 76,032 message/mask cases.

The review repaired plan/seed validation, canonical trace types, summary
partitions and rates, mutable metadata and checkpoint/model consistency. The
source suite contains 366 tests, including 59 separately authored review cases.
Independent retraining and checkpoint inference reproduced the recorded findings.
The original study artifacts and their historical source identities are preserved.

## Installation

The wheel is available in this GitHub release. Install the downloaded file with:

```bash
python -m pip install latent_error_correcting_representations-0.2.0-py3-none-any.whl
python -m experiments.showcase --out /tmp/latent-demo-new
```

The source archive also includes tests and complete checked research evidence.
No PyPI package is published by this release.

## Compatibility and interpretation

Output destinations must be new. Invalid repair controls, incomplete or
contradictory records and unsupported study plans are rejected explicitly.
The four historical method names and metrics remain intact. Conditional accuracy
is null when the policy accepts no samples. Valid old selective-study artifacts
remain readable under their recorded source identity.

SECDED's double-error rejection has zero coverage; triple errors can be
miscorrected and coherent movement to another valid codeword remains undetected.
The study does not establish general AI alignment, human-influence or real-model
safety. Hashes establish consistency rather than authenticated provenance.
See INDEPENDENT_REVIEW.md for test methods, repairs and limitations.
