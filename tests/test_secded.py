from itertools import combinations, product
import pytest
import torch
from latent_error_correcting_representations.codes import METHODS
from latent_error_correcting_representations.selective import (
    SELECTIVE_METHODS,
    encode_hamming84,
    decode_hamming84,
    messages,
    selective_encode,
    selective_decode,
    outcome_counts,
    nearest_valid_replacement,
    adversarial_exact_flip,
)
from latent_error_correcting_representations.exhaustive import enumerate_codes


def reference_codeword(bits):
    # Independent linear generator columns, including overall parity.
    generators = [
        (1, 1, 1, 0, 0, 0, 0, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (0, 1, 0, 1, 0, 1, 0, 1),
        (1, 1, 0, 1, 0, 0, 1, 0),
    ]
    return tuple(
        sum(bits[j] * generators[j][i] for j in range(4)) % 2 for i in range(8)
    )


def test_extended_codebook_matches_linear_generator_and_distance_four():
    truth = messages()
    expected = [reference_codeword(bits) for bits in truth.tolist()]
    assert encode_hamming84(truth).tolist() == [list(x) for x in expected]
    assert (
        min(
            sum(a != b for a, b in zip(x, y))
            for i, x in enumerate(expected)
            for j, y in enumerate(expected)
            if i != j
        )
        == 4
    )


def test_every_received_word_matches_independent_radius_one_decoder():
    truth = messages().tolist()
    book = [reference_codeword(bits) for bits in truth]
    words = list(product((0, 1), repeat=8))
    decoded = decode_hamming84(torch.tensor(words))
    for i, word in enumerate(words):
        near = [
            (sum(a != b for a, b in zip(word, code)), label)
            for label, code in enumerate(book)
        ]
        best = min(near)
        accepted = best[0] <= 1
        assert bool(decoded.accepted[i]) == accepted
        assert bool(decoded.detected[i]) == (word not in book)
        assert bool(decoded.repaired[i]) == (best[0] == 1)
        if accepted:
            assert decoded.task_labels[i].item() == best[1]
        else:
            assert decoded.task_labels[i].item() == -1


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_all_messages_and_every_corruption_budget(k):
    truth = messages()
    base = encode_hamming84(truth)
    for pattern in combinations(range(8), k):
        damaged = base.clone()
        if pattern:
            damaged[:, list(pattern)] ^= 1
        before = damaged.clone()
        result = decode_hamming84(damaged)
        assert torch.equal(damaged, before)
        if k <= 1:
            assert result.accepted.all() and torch.equal(
                result.candidate_message, truth
            )
        if k == 2:
            assert (
                not result.accepted.any()
                and result.detected.all()
                and not result.repaired.any()
            )
        if k == 3:
            assert (
                result.accepted.all()
                and (result.candidate_message != truth).any(dim=1).all()
            )


@pytest.mark.parametrize("method", SELECTIVE_METHODS)
def test_clean_and_coherent_drift_controls(method):
    truth = messages()
    clean = selective_decode(method, selective_encode(method, truth))
    assert clean.accepted.all() and not clean.detected.any()
    drift = selective_decode(method, nearest_valid_replacement(method, truth))
    assert drift.accepted.all() and not drift.detected.any()
    assert (drift.candidate_message != truth).any(dim=1).all()


@pytest.mark.parametrize("method", SELECTIVE_METHODS)
@pytest.mark.parametrize("k", [1, 2])
def test_attacks_exhaustively_maximize_accepted_error_before_abstention(method, k):
    truth = messages()
    base = selective_encode(method, truth)
    actual = adversarial_exact_flip(method, base, truth, k)
    rows = []
    for i, word in enumerate(base):
        choices = []
        for pattern in combinations(range(len(word)), k):
            x = word.clone()
            x[list(pattern)] ^= 1
            dec = selective_decode(method, x[None])
            score = 2 * int(
                dec.accepted[0] and not torch.equal(dec.candidate_message[0], truth[i])
            ) + int(not dec.accepted[0])
            choices.append((score, x))
        maximum = max(score for score, x in choices)
        rows.append(next(x for score, x in choices if score == maximum))
    assert (
        torch.equal(actual, torch.stack(rows))
        and ((actual != base).sum(dim=1) == k).all()
    )


def test_zero_coverage_has_no_conditional_accuracy_or_success():
    truth = messages()
    code = encode_hamming84(truth)
    code[:, [0, 1]] ^= 1
    result = outcome_counts(decode_hamming84(code), truth)
    assert result["conditional_accuracy"] is None
    assert (
        result["task_success_rate"]
        == result["wrong_acceptance_rate"]
        == result["coverage"]
        == 0
    )
    assert result["abstained"] == 16


@pytest.mark.parametrize("value", [-1, 2, 0.5, float("nan"), float("inf")])
def test_extended_code_rejects_invalid_bits(value):
    code = torch.zeros((2, 8))
    code[0, 1] = value
    with pytest.raises(ValueError):
        decode_hamming84(code)


def test_empty_and_noncontiguous_inputs():
    empty = decode_hamming84(torch.empty((0, 8)))
    assert empty.task_labels.shape == (0,)
    code = encode_hamming84(messages()).T.contiguous().T
    assert torch.equal(decode_hamming84(code).candidate_message, messages())


def test_exhaustive_counts_are_exact_and_legacy_registry_unchanged():
    result = enumerate_codes()
    extended = result["methods"]["hamming84_secded"]["exact_bit_flips"]
    assert result["message_mask_pairs"] == 76032
    assert extended["1"]["correct_accepted"] == 128
    assert extended["2"]["abstained"] == 448
    assert extended["3"]["wrong_accepted"] == 896
    assert extended["4"]["wrong_accepted"] == 224 and extended["4"]["abstained"] == 896
    assert list(METHODS) == [
        "uncoded",
        "repetition3",
        "hamming74_detect",
        "hamming74_repair",
    ]
