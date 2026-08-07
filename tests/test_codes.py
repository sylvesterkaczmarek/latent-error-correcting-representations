import torch

from latent_error_correcting_representations.codes import (
    encode_hamming74,
    decode_hamming74,
    encode_repetition3,
    decode_repetition3,
)


def test_hamming_round_trip_all_messages():
    messages = torch.tensor([[ (i >> b) & 1 for b in range(4)] for i in range(16)], dtype=torch.int64)
    code = encode_hamming74(messages)
    dec = decode_hamming74(code, repair=True)
    assert torch.equal(dec.message, messages)
    assert not dec.detected.any()


def test_hamming_corrects_every_single_bit():
    messages = torch.tensor([[ (i >> b) & 1 for b in range(4)] for i in range(16)], dtype=torch.int64)
    code = encode_hamming74(messages)
    for j in range(7):
        corrupt = code.clone()
        corrupt[:, j] ^= 1
        dec = decode_hamming74(corrupt, repair=True)
        assert torch.equal(dec.message, messages)
        assert dec.detected.all()


def test_repetition_corrects_every_single_bit():
    messages = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.int64)
    code = encode_repetition3(messages)
    for j in range(code.shape[1]):
        corrupt = code.clone()
        corrupt[:, j] ^= 1
        dec = decode_repetition3(corrupt, repair=True)
        assert torch.equal(dec.message, messages)
        assert dec.detected.all()
