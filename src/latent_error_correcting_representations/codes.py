from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecodeResult:
    """Decoder output; corrected records an attempted repair, not its success."""

    message: torch.Tensor
    detected: torch.Tensor
    corrected: torch.Tensor


def _binary_batch(
    value: torch.Tensor, width: int | None = None, *, name: str
) -> torch.Tensor:
    """Validate before integer conversion so invalid bit values cannot be truncated."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or (width is not None and value.shape[1] != width):
        expected = f"(N, {width})" if width is not None else "(N, width)"
        raise ValueError(f"{name} must have shape {expected}")
    if value.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one bit per row")
    if value.is_complex() or not bool(((value == 0) | (value == 1)).all()):
        raise ValueError(f"{name} must contain only binary values 0 or 1")
    return value.to(torch.int64)


def task_label(message: torch.Tensor) -> torch.Tensor:
    """Map the four-bit semantic message to one of 16 task identities."""
    m = _binary_batch(message, 4, name="message")
    weights = torch.tensor([1, 2, 4, 8], dtype=torch.int64, device=m.device)
    return (m * weights).sum(dim=-1)


def encode_uncoded(message: torch.Tensor) -> torch.Tensor:
    return _binary_batch(message, 4, name="message").clone()


def decode_uncoded(codeword: torch.Tensor, repair: bool = False) -> DecodeResult:
    c = _binary_batch(codeword, 4, name="codeword")
    n = c.shape[0]
    false = torch.zeros(n, dtype=torch.bool, device=codeword.device)
    return DecodeResult(c.clone(), false, false)


def encode_repetition3(message: torch.Tensor) -> torch.Tensor:
    m = _binary_batch(message, 4, name="message")
    return m.repeat_interleave(3, dim=-1)


def decode_repetition3(codeword: torch.Tensor, repair: bool = True) -> DecodeResult:
    c = _binary_batch(codeword, 12, name="codeword")
    groups = c.reshape(c.shape[0], 4, 3)
    sums = groups.sum(dim=-1)
    message = (sums >= 2).to(torch.int64)
    detected = ((sums != 0) & (sums != 3)).any(dim=-1)
    corrected = detected.clone() if repair else torch.zeros_like(detected)
    if not repair:
        message = groups[:, :, 0]
    return DecodeResult(message, detected, corrected)


def encode_hamming74(message: torch.Tensor) -> torch.Tensor:
    """Systematic Hamming(7,4), parity positions 1,2,4."""
    m = _binary_batch(message, 4, name="message")
    d1, d2, d3, d4 = [m[:, i] for i in range(4)]
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4
    return torch.stack([p1, p2, d1, p4, d2, d3, d4], dim=1)


def hamming74_syndrome(codeword: torch.Tensor) -> torch.Tensor:
    c = _binary_batch(codeword, 7, name="codeword")
    s1 = c[:, 0] ^ c[:, 2] ^ c[:, 4] ^ c[:, 6]
    s2 = c[:, 1] ^ c[:, 2] ^ c[:, 5] ^ c[:, 6]
    s4 = c[:, 3] ^ c[:, 4] ^ c[:, 5] ^ c[:, 6]
    return s1 + 2 * s2 + 4 * s4


def decode_hamming74(codeword: torch.Tensor, repair: bool = True) -> DecodeResult:
    syndrome = hamming74_syndrome(codeword)
    c = codeword.to(torch.int64).clone()
    detected = syndrome != 0
    corrected = torch.zeros_like(detected)
    if repair:
        rows = torch.nonzero(detected, as_tuple=False).flatten()
        if rows.numel():
            positions = syndrome[rows] - 1
            c[rows, positions] ^= 1
            corrected[rows] = True
    message = c[:, [2, 4, 5, 6]]
    return DecodeResult(message, detected, corrected)


METHODS = {
    "uncoded": (encode_uncoded, decode_uncoded, False),
    "repetition3": (encode_repetition3, decode_repetition3, True),
    "hamming74_detect": (encode_hamming74, decode_hamming74, False),
    "hamming74_repair": (encode_hamming74, decode_hamming74, True),
}


def encode(method: str, message: torch.Tensor) -> torch.Tensor:
    return METHODS[method][0](message)


def decode(method: str, codeword: torch.Tensor) -> DecodeResult:
    _, decoder, repair = METHODS[method]
    return decoder(codeword, repair=repair)
