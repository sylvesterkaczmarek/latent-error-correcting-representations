# Selective decoding study

## Question and scope

When a decoder detects corruption, should it return its best guess or abstain?
This study adds two explicit acceptance policies to the historical four-method
benchmark: Hamming(7,4) detection with abstention, and extended Hamming(8,4) SECDED.
The latter is a standard classical code, not a new learned correction mechanism.
It adds one overall parity bit, corrects single-bit errors and rejects detected
double-bit errors. The study also tests three/four-bit errors and valid-codeword
substitution, beyond that bounded guarantee.

The neural encoder predicts a supervised four-bit message before this classical
coding stage. No LLM, continuous transformer latent, human participant or physical
sensor is evaluated. The outcome measures distinguish preserved encoder prediction
from correct ground-truth task identity. Both can diverge when the encoder errs.

## Predeclared design

Five seeds 7, 17, 29, 41 and 53 each train an MLP for five epochs on 2,500 samples
and evaluate 800 independently sampled test observations from the same synthetic
projection. Shared semantic classes and projection define the task; evaluation
observations are not used for training or selecting settings. Architecture and
hyperparameters match the historical reference. There is no hyperparameter search.

Every trained encoder is evaluated under six policies and eight challenges:
0/1/2/3/4 exact random flips, exhaustive one/two-bit attacks, and nearest different
valid-codeword substitution. That gives 240 seed/policy/challenge cells and 192,000
example-level decisions, drawn from five trained encoders and 4,000 base test
observations. Those decisions are not independent training repetitions.

Equal-width policies share corruption masks. Different widths use independently
specified deterministic streams. Equal flip counts do not equalize code rate or
per-bit error probability: uncoded uses 4 bits, Hamming uses 7, extended Hamming
uses 8, and repetition uses 12. No resource-matched superiority is claimed.
Attacks prefer an accepted wrong answer, then abstention, then correct acceptance.
Their reference is the encoder prediction, so end-to-end attack results are not
worst-case attacks against ground truth.

## Metrics

Every cell records integer counts as well as rates. Accepted correct answers,
accepted wrong answers and abstentions partition all examples. Coverage is the
accepted fraction. Task success is accepted correct / all examples. Wrong
acceptance is accepted wrong / all examples. Conditional accuracy divides accepted
correct by accepted count and is null when no answer is accepted. Abstention is
never counted as a correct answer. Candidate bits remain available for diagnosis;
`task_labels` masks rejected outputs to -1.

Repair activation records an attempted correction, not successful repair. The
`repair_to_reference` count means that an attempted repair ended at the named
reference; it is not a causal success claim. Detection includes syndromes and
parity checks, not a learned anomaly detector.

## Executed results

All five encoders reached 1.000 held-out task accuracy. The table shows two-bit
random corruption, mean ± sample standard deviation across the five seeds.

| Policy | Encoded bits | Coverage | Correct accepted / all | Wrong accepted / all |
| --- | ---: | ---: | ---: | ---: |
| Uncoded | 4 | 1.000 | 0.000 | 1.000 |
| Repetition with majority | 12 | 1.000 | 0.82525 ± 0.01740 | 0.17475 ± 0.01740 |
| Hamming detect and still return | 7 | 1.000 | 0.13500 ± 0.00606 | 0.86500 ± 0.00606 |
| Hamming single-error repair | 7 | 1.000 | 0.000 | 1.000 |
| Hamming detection and abstention | 7 | 0.000 | 0.000 | 0.000 |
| Extended Hamming SECDED | 8 | 0.000 | 0.000 | 0.000 |

The zero-error-acceptance result for SECDED under double errors comes with zero
coverage, not perfect task completion. In contrast, SECDED corrects and accepts
all single-bit errors. Every three-bit error is accepted incorrectly under the
standard SECDED decoder in this four-bit-message construction. All policies also
accept the deliberately substituted wrong valid codeword without detecting it.

The artifact contains descriptive paired bootstrap intervals based on seeds,
with 5,000 resamples and a fixed bootstrap seed. They are not multiplicity-adjusted
significance tests or generalisation claims. Undefined conditional accuracy is
not replaced with zero. Exact code-space counts below do not need a confidence
interval. The five-seed replay reused identical seeds, so it adds no independent
samples.

## Exhaustive mathematical evidence

The enumeration covers every four-bit message and every possible corruption mask
for each of the six policies: 76,032 message/mask combinations in total. Different
policies can share a code and still count as separate evaluations.
For SECDED specifically:

| Exact flipped bits | Cases | Accepted correct | Accepted wrong | Abstained |
| --- | ---: | ---: | ---: | ---: |
| 0 | 16 | 16 | 0 | 0 |
| 1 | 128 | 128 | 0 | 0 |
| 2 | 448 | 0 | 0 | 448 |
| 3 | 896 | 0 | 896 | 0 |
| 4 | 1,120 | 0 | 224 | 896 |

The checked artifact includes every budget up to the encoded width. Tests also
compare all 256 possible SECDED received words against a separately written
nearest-codeword, radius-one oracle and verify minimum distance four using an
independent generator matrix. This verifies software and known coding properties;
it is not evidence of a new coding theorem.

## Inspect and reproduce

The complete plan, source hashes, model checkpoints, per-example traces, raw
seed results, summary and completion manifest are in `results/selective-v1/`.
The historical `results/summary.json` remains byte-for-byte unchanged. The new
random streams differ from the historical method-name-based streams, so sampled
repetition/detect-only rates need not equal the historical table.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python -m experiments.selective_decoding verify --out results/selective-v1
python -m experiments.selective_decoding run --out /tmp/latent-ecc-study-new
python -m experiments.showcase --out /tmp/latent-ecc-demo-new
```

Use absent output destinations. Verification recomputes corruptions and decisions
from the recorded traces, aggregates the seed results, checks file hashes, and
repeats finite enumeration. Version 0.2.0 also validates the complete declared
design, unique cells, integer counts and rates, canonical recorded bit types,
and restricted checkpoint contents against the traced model state. It does not
retrain or replay model inference during routine verification. Hashes are integrity
checks, not authentication. A complete replay with the same environment reproduced
all five seeds, all 240 cells, traces, checkpoints and statistics exactly. Cross-
version or cross-hardware bitwise training reproducibility is not promised.

## References

- SECDED implementation documentation: https://kevinpt.github.io/vhdl-extras/rst/modules/secded_edac.html
- AMD memory ECC documentation: https://docs.amd.com/r/en-US/ug573-ultrascale-memory-resources/Built-in-Error-Detection-and-Correction
- PyTorch reproducibility guidance: https://docs.pytorch.org/docs/stable/notes/randomness.html

## Independent release reproduction

A separate fresh checkout reproduced the original five-seed execution byte for
byte across all 19 files. After the verifier repairs, another five-seed run
retained every prediction, decision, count, loss, data fingerprint and model
tensor. Source/plan hashes and checkpoint file hashes changed because the saved
checkpoint binds its plan. These expected provenance changes do not alter the
results. Saved-state inference was additionally evaluated through direct linear
and tanh operations for both sets of five checkpoints, confirming 8,000
predictions against the traces. This remains five independent training seeds.
