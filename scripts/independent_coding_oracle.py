"""Integer-only reference construction, independent of tensor coding algorithms."""

import itertools, json
import torch
from latent_error_correcting_representations.selective import (
    SELECTIVE_METHODS,
    selective_encode,
    selective_decode,
    adversarial_exact_flip,
)
from latent_error_correcting_representations.exhaustive import enumerate_codes

WIDTH = {
    "uncoded": 4,
    "repetition3": 12,
    "hamming74_detect": 7,
    "hamming74_repair": 7,
    "hamming74_abstain": 7,
    "hamming84_secded": 8,
}


def bits(word, width):
    return [(word >> p) & 1 for p in range(width)]


def label(word):
    return sum(((word >> position) & 1) << i for i, position in enumerate((2, 4, 5, 6)))


def hcode(message):
    initial = sum(
        ((message >> i) & 1) << position for i, position in enumerate((2, 4, 5, 6))
    )
    # Solve H*c=0 by enumerating the three parity bits, rather than reusing encoder formulas.
    solutions = []
    for parity in range(8):
        word = initial | sum(((parity >> i) & 1) << p for i, p in enumerate((0, 1, 3)))
        syndrome = 0
        for position in range(7):
            if (word >> position) & 1:
                syndrome ^= position + 1
        if syndrome == 0:
            solutions.append(word)
    assert len(solutions) == 1
    return solutions[0]


def code(method, message):
    if method == "uncoded":
        return message
    if method == "repetition3":
        return sum(
            ((message >> i) & 1) << (3 * i + j) for i in range(4) for j in range(3)
        )
    word = hcode(message)
    return (
        word | ((word.bit_count() % 2) << 7) if method == "hamming84_secded" else word
    )


def decision(method, word):
    book = [code(method, m) for m in range(16)]
    detected = word not in book
    if method == "uncoded":
        return word, False, False, True
    if method == "repetition3":
        return (
            sum(
                (sum((word >> (3 * i + j)) & 1 for j in range(3)) >= 2) << i
                for i in range(4)
            ),
            detected,
            detected,
            True,
        )
    if method in ("hamming74_detect", "hamming74_abstain"):
        return (
            label(word),
            detected,
            False,
            (not detected if method == "hamming74_abstain" else True),
        )
    distance = [(word ^ c).bit_count() for c in book]
    near = [i for i, d in enumerate(distance) if d <= 1]
    if method == "hamming74_repair":
        assert len(near) == 1
    assert len(near) <= 1
    return (
        (near[0], detected, detected, True)
        if near
        else (label(word), True, False, False)
    )


def verify():
    record = enumerate_codes()
    total = received = attacks = 0
    for method in SELECTIVE_METHODS:
        width = WIDTH[method]
        nwords = 1 << width
        table = [decision(method, w) for w in range(nwords)]
        input_tensor = torch.tensor([bits(w, width) for w in range(nwords)])
        actual = selective_decode(method, input_tensor)
        ref = torch.tensor([bits(m, 4) for m, d, r, a in table])
        assert torch.equal(actual.candidate_message, ref), method
        for index, field in enumerate(("detected", "repaired", "accepted"), 1):
            assert getattr(actual, field).tolist() == [r[index] for r in table]
        received += nwords
        truth = torch.tensor([bits(m, 4) for m in range(16)])
        assert selective_encode(method, truth).tolist() == [
            bits(code(method, m), width) for m in range(16)
        ]
        rows = {
            str(k): dict(
                n=0,
                accepted=0,
                abstained=0,
                correct_accepted=0,
                wrong_accepted=0,
                detected=0,
                repair_attempted=0,
                repair_to_reference=0,
                candidate_correct=0,
            )
            for k in range(width + 1)
        }
        for message in range(16):
            for mask in range(nwords):
                m, d, r, a = table[code(method, message) ^ mask]
                correct = m == message
                row = rows[str(mask.bit_count())]
                for key, value in dict(
                    n=1,
                    accepted=a,
                    abstained=not a,
                    correct_accepted=a and correct,
                    wrong_accepted=a and not correct,
                    detected=d,
                    repair_attempted=r,
                    repair_to_reference=r and correct,
                    candidate_correct=correct,
                ).items():
                    row[key] += int(value)
                total += 1
        for k, row in rows.items():
            for key, value in row.items():
                assert value == record["methods"][method]["exact_bit_flips"][k][key], (
                    method,
                    k,
                    key,
                )
        # Independent adversarial choice for every valid message at both exact budgets.
        for k in (1, 2):
            damaged = adversarial_exact_flip(
                method, selective_encode(method, truth), truth, k
            ).tolist()
            for message in range(16):
                candidates = []
                for positions in itertools.combinations(range(width), k):
                    w = code(method, message) ^ sum(1 << p for p in positions)
                    m, d, r, a = table[w]
                    score = 2 * int(a and m != message) + int(not a)
                    candidates.append((score, w))
                best = max(score for score, w in candidates)
                expected = next(w for score, w in candidates if score == best)
                assert damaged[message] == bits(expected, width)
                attacks += 1
    report = {
        "status": "PASS",
        "message_mask_cases": total,
        "all_received_words": received,
        "adversarial_valid_message_cases": attacks,
        "reference": "Integer parity-constraint codebook, nearest-codeword and majority decoding; no implementation encoder/decoder used to construct expected outcomes.",
    }
    assert total == 76032
    return report


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
