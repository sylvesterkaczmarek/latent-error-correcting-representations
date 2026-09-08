import itertools

import pytest
import torch

from latent_error_correcting_representations.codes import METHODS, encode
from latent_error_correcting_representations.corruption import (
    adversarial_single_flip,
    flip_exactly_k,
    nearest_opposite_valid_codeword,
)

WIDTHS = {"uncoded": 4, "repetition3": 12, "hamming74_detect": 7, "hamming74_repair": 7}
MESSAGES = torch.tensor(
    [[(value >> bit) & 1 for bit in range(4)] for value in range(16)]
)


def _oracle_label(method, bits):
    bits = list(bits)
    if method == "uncoded":
        message = bits
    elif method == "repetition3":
        message = [int(sum(bits[start : start + 3]) >= 2) for start in range(0, 12, 3)]
    else:
        if method == "hamming74_repair":
            # An independent parity-check matrix interpretation over GF(2).
            syndrome = 0
            for position, bit in enumerate(bits, start=1):
                if bit:
                    syndrome ^= position
            if syndrome:
                bits[syndrome - 1] ^= 1
        message = [bits[position] for position in (2, 4, 5, 6)]
    return sum(bit * 2**position for position, bit in enumerate(message))


@pytest.mark.parametrize("width", [4, 7, 12])
def test_exact_k_counts_and_rng_stream_match_previous_valid_contract(width):
    source = torch.tensor([[index % 2 for index in range(width)]]).repeat(11, 1)
    generator = torch.Generator().manual_seed(351)
    reference_generator = torch.Generator().manual_seed(351)
    before = source.clone()
    for count in range(width + 1):
        expected = source.clone()
        if count:
            for row in range(len(expected)):
                indices = torch.randperm(width, generator=reference_generator)[:count]
                expected[row, indices] ^= 1
        actual = flip_exactly_k(source, count, generator)
        assert torch.equal(actual, expected)
        assert ((actual != source).sum(dim=1) == count).all()
        assert torch.equal(generator.get_state(), reference_generator.get_state())
    assert torch.equal(source, before)


@pytest.mark.parametrize("count", [-1, 8, 3.5, True])
def test_invalid_flip_counts_fail_without_advancing_rng(count):
    generator = torch.Generator().manual_seed(29)
    before = generator.get_state().clone()
    with pytest.raises((ValueError, TypeError), match="k must"):
        flip_exactly_k(torch.zeros((3, 7)), count, generator)
    assert torch.equal(generator.get_state(), before)


@pytest.mark.parametrize("method", METHODS)
def test_adversarial_search_all_bit_patterns_and_all_labels(method):
    width = WIDTHS[method]
    rows = []
    labels = []
    expected = []
    for bits in itertools.product((0, 1), repeat=width):
        candidates = []
        candidate_labels = []
        for position in range(width):
            candidate = list(bits)
            candidate[position] ^= 1
            candidates.append(candidate)
            candidate_labels.append(_oracle_label(method, candidate))
        for label in range(16):
            selected = next(
                (
                    position
                    for position, pred in enumerate(candidate_labels)
                    if pred != label
                ),
                0,
            )
            rows.append(bits)
            labels.append(label)
            expected.append(candidates[selected])
    source = torch.tensor(rows)
    before = source.clone()
    actual = adversarial_single_flip(method, source, torch.tensor(labels))
    assert torch.equal(actual, torch.tensor(expected))
    assert torch.equal(source, before)
    assert ((actual != source).sum(dim=1) == 1).all()


@pytest.mark.parametrize("method", METHODS)
def test_nearest_valid_search_has_minimum_distance_and_lowest_label_ties(method):
    all_codes = encode(method, MESSAGES).tolist()
    expected = []
    for label, codeword in enumerate(all_codes):
        alternatives = [
            (
                sum(left != right for left, right in zip(codeword, candidate)),
                other_label,
                candidate,
            )
            for other_label, candidate in enumerate(all_codes)
            if label != other_label
        ]
        expected.append(min(alternatives)[2])
    # Cross several search chunks, including a partial final chunk.
    messages = MESSAGES.repeat(130, 1)
    assert torch.equal(
        nearest_opposite_valid_codeword(method, messages),
        torch.tensor(expected).repeat(130, 1),
    )


@pytest.mark.parametrize("method", METHODS)
def test_empty_corruption_batches(method):
    codes = torch.empty((0, WIDTHS[method]), dtype=torch.int64)
    assert (
        adversarial_single_flip(method, codes, torch.empty((0,))).shape == codes.shape
    )
    assert (
        nearest_opposite_valid_codeword(method, torch.empty((0, 4))).shape
        == codes.shape
    )
    assert flip_exactly_k(codes, 1, torch.Generator()).shape == codes.shape


@pytest.mark.parametrize(
    "labels",
    [
        torch.zeros((1, 1)),
        torch.tensor([0.5]),
        torch.tensor([-1]),
        torch.tensor([16]),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([0j]),
    ],
)
def test_invalid_task_labels(labels):
    with pytest.raises(ValueError, match="true_label"):
        adversarial_single_flip("uncoded", torch.zeros((1, 4)), labels)


def test_invalid_corruption_input_bits_and_shape():
    generator = torch.Generator()
    with pytest.raises(ValueError, match="binary"):
        flip_exactly_k(torch.full((2, 4), 0.5), 0, generator)
    with pytest.raises(ValueError, match="shape"):
        flip_exactly_k(torch.zeros((4,)), 0, generator)
    with pytest.raises(ValueError, match="at least one"):
        flip_exactly_k(torch.zeros((2, 0)), 0, generator)
    with pytest.raises(ValueError, match="shape"):
        adversarial_single_flip("uncoded", torch.empty((0, 7)), torch.empty((0,)))
