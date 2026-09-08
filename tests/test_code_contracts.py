import itertools

import pytest
import torch

from latent_error_correcting_representations.codes import (
    METHODS,
    decode,
    encode,
    hamming74_syndrome,
    task_label,
)

MESSAGES = torch.tensor(
    [[(value >> bit) & 1 for bit in range(4)] for value in range(16)], dtype=torch.int64
)
WIDTHS = {"uncoded": 4, "repetition3": 12, "hamming74_detect": 7, "hamming74_repair": 7}


def test_hamming_codebook_has_independent_parity_checks_and_distance_three():
    codewords = encode("hamming74_repair", MESSAGES)
    # H has the distinct nonzero three-bit columns 1 through 7 over GF(2).
    parity_check = torch.tensor(
        [[(column >> bit) & 1 for column in range(1, 8)] for bit in range(3)]
    )
    assert not (codewords @ parity_check.T % 2).any()
    distances = (codewords[:, None] != codewords[None, :]).sum(dim=-1)
    assert distances[~torch.eye(16, dtype=torch.bool)].min().item() == 3
    assert torch.equal(task_label(MESSAGES), torch.arange(16))


@pytest.mark.parametrize("repair", [False, True])
def test_hamming_every_one_and_two_bit_pattern(repair):
    method = "hamming74_repair" if repair else "hamming74_detect"
    codewords = encode(method, MESSAGES)
    for count in (1, 2):
        for positions in itertools.combinations(range(7), count):
            damaged = codewords.clone()
            damaged[:, positions] ^= 1
            before = damaged.clone()
            decoded = decode(method, damaged)
            expected_syndrome = 0
            for position in positions:
                expected_syndrome ^= position + 1
            assert torch.equal(
                hamming74_syndrome(damaged), torch.full((16,), expected_syndrome)
            )
            assert decoded.detected.all()
            assert torch.equal(decoded.corrected, torch.full((16,), repair))
            assert torch.equal(damaged, before)
            if repair:
                if count == 1:
                    assert torch.equal(decoded.message, MESSAGES)
                else:
                    # A two-bit error is miscorrected to a different valid word by SEC Hamming.
                    assert (decoded.message != MESSAGES).any(dim=1).all()
                    repaired = damaged.clone()
                    repaired[:, expected_syndrome - 1] ^= 1
                    assert not hamming74_syndrome(repaired).any()
                    assert torch.equal(encode(method, decoded.message), repaired)
            else:
                assert torch.equal(decoded.message, damaged[:, [2, 4, 5, 6]])


def test_repetition_every_one_and_two_bit_pattern_all_messages():
    codewords = encode("repetition3", MESSAGES)
    for count in (1, 2):
        for positions in itertools.combinations(range(12), count):
            damaged = codewords.clone()
            damaged[:, positions] ^= 1
            decoded = decode("repetition3", damaged)
            expected = MESSAGES.clone()
            if count == 2 and positions[0] // 3 == positions[1] // 3:
                expected[:, positions[0] // 3] ^= 1
            assert torch.equal(decoded.message, expected)
            assert decoded.detected.all()
            assert decoded.corrected.all()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.int32, torch.float32, torch.float64]
)
def test_binary_dtypes_round_trip_without_input_mutation(method, dtype):
    messages = MESSAGES.to(dtype)
    before = messages.clone()
    decoded = decode(method, encode(method, messages).to(dtype))
    assert torch.equal(decoded.message, MESSAGES)
    assert decoded.message.dtype == torch.int64
    assert torch.equal(messages, before)


@pytest.mark.parametrize("method", METHODS)
def test_empty_batches_have_consistent_shapes(method):
    messages = torch.empty((0, 4))
    codewords = encode(method, messages)
    result = decode(method, codewords)
    assert codewords.shape == (0, WIDTHS[method])
    assert result.message.shape == (0, 4)
    assert result.detected.shape == result.corrected.shape == (0,)
    assert task_label(messages).shape == (0,)


@pytest.mark.parametrize("bad_value", [-1, 2, 0.5, float("nan"), float("inf")])
@pytest.mark.parametrize("method", METHODS)
def test_non_binary_inputs_are_rejected_before_conversion(method, bad_value):
    message = MESSAGES.float()
    message[0, 0] = bad_value
    with pytest.raises(ValueError, match="binary"):
        encode(method, message)
    with pytest.raises(ValueError, match="binary"):
        task_label(message)
    codeword = encode(method, MESSAGES).float()
    codeword[0, 0] = bad_value
    with pytest.raises(ValueError, match="binary"):
        decode(method, codeword)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 5), (1, 1, 4)])
def test_message_shape_contract(method, shape):
    with pytest.raises(ValueError, match="shape"):
        encode(method, torch.zeros(shape))
    with pytest.raises(ValueError, match="shape"):
        task_label(torch.zeros(shape))


@pytest.mark.parametrize("method", METHODS)
def test_codeword_shape_contract(method):
    for shape in [(WIDTHS[method],), (2, WIDTHS[method] + 1), (1, 1, WIDTHS[method])]:
        with pytest.raises(ValueError, match="shape"):
            decode(method, torch.zeros(shape))


def test_complex_bits_are_rejected():
    with pytest.raises(ValueError, match="binary"):
        encode("uncoded", MESSAGES.to(torch.complex64))


def test_noncontiguous_repetition_input():
    message = MESSAGES[:3]
    codeword = encode("repetition3", message).T.contiguous().T
    assert not codeword.is_contiguous()
    assert torch.equal(decode("repetition3", codeword).message, message)
