# Method

## Research question

Can structured redundancy inside a neural representation provide an endogenous signal that detects and repairs internal corruption, and where does that signal fail?

The benchmark intentionally separates two failure classes:

1. **Channel-like corruption** changes bits after a semantic message has been formed.
2. **Code-consistent drift** replaces one valid semantic message with another valid message.

Error-correcting codes are designed for the first class. The second class is included as a deliberate boundary test.

## Learned semantic bottleneck

A small MLP receives a 16-dimensional noisy observation generated from an underlying four-bit semantic message. The encoder is trained to recover those four bits.

The four-bit message is also treated as one of 16 task identities, so any message-bit change changes the downstream task identity.

The checked reference suite reports clean encoder accuracy separately from post-encoding corruption experiments. The existing corruption `task_accuracy` and `message_bit_accuracy` fields measure preservation of the encoder's **predicted message before corruption**. They can be perfect even when the encoder is wrong.

New `end_to_end` sections report both accuracies against the dataset's **true message** for the same corrupted examples. The attacks still target the pre-corruption identity, so the adversarial end-to-end score is not a worst-case attack against ground truth. A constructed drift may accidentally repair an encoder error. Metadata names both references explicitly.

## Representation methods

### Uncoded

The four message bits are used directly. There is no redundancy, detection signal, or repair.

### Repetition-3

Each message bit is repeated three times. A disagreement inside a triplet provides a detection signal; majority vote repairs one corrupted copy per triplet.

### Hamming(7,4) detect

The four message bits are encoded as a systematic Hamming(7,4) codeword. A non-zero parity-check syndrome detects an invalid codeword, but the representation is decoded without correction.

### Hamming(7,4) repair

The same code is used, but a non-zero syndrome is interpreted as a single-bit error location and the corresponding bit is flipped before decoding.

Hamming(7,4) is a single-error-correcting code. The benchmark intentionally includes two-bit corruption to show the limit of blindly applying a single-error repair rule. Exhaustive regression checks cover every four-bit message and every one-bit and two-bit corruption for both coded methods.

## Random corruption

For each representation, the benchmark flips exactly 0, 1, or 2 randomly selected latent bits per test sample. The public flip API rejects negative, non-integer or over-width budgets rather than silently changing the requested experiment.

Reported metrics include:

- message-bit accuracy,
- task identity accuracy,
- corruption detection rate,
- repair activation rate (`correction_rate`), which does not establish successful repair.

Exact bit counts are used instead of a per-bit corruption probability so the single-error correction guarantee is directly visible.

## Adversarial single-bit corruption

For each test sample, the benchmark tests every possible one-bit latent flip and selects a flip that causes downstream task failure when one exists.

This is a white-box discrete latent attack, not an input-space adversarial example.

## Code-consistent drift

For each internal message, the benchmark replaces the encoded state with the nearest **valid** codeword corresponding to a different task identity.

Because the new representation is itself a valid codeword:

- the Hamming syndrome is zero,
- repetition triplets are internally consistent,
- repair logic has no reason to activate.

This is intentionally adversarial and does not model spontaneous semantic drift. It tests a narrower question: whether code validity is sufficient for semantic validity.

## Interpretation

A positive result under bit corruption supports error-correcting redundancy as a mechanism for low-level latent integrity.

Failure under code-consistent drift shows that local representational consistency cannot by itself establish that the representation still means the right thing.

## Search implementation

The adversarial search evaluates every bit position in batches of at most 1,024 examples, retaining the first position on a tie. The valid-codeword search retains the lowest task label on a distance tie. Batching reduces repeated Python and tensor-operation overhead; it preserves the corruption choices and uses bounded intermediate memory.
