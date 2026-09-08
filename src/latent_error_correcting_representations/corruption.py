from __future__ import annotations

import operator

import torch

from .codes import _binary_batch, decode, encode, task_label

_SEARCH_BATCH_SIZE = 1024


def flip_exactly_k(
    codeword: torch.Tensor, k: int, generator: torch.Generator
) -> torch.Tensor:
    """Flip exactly k distinct bits per row, preserving the supplied RNG stream."""
    out = _binary_batch(codeword, name="codeword").clone()
    width = out.shape[1]
    if isinstance(k, bool):
        raise TypeError("k must be an integer bit count")
    try:
        k = operator.index(k)
    except TypeError as error:
        raise TypeError("k must be an integer bit count") from error
    if not 0 <= k <= width:
        raise ValueError(f"k must lie between 0 and codeword width {width}")
    if k == 0:
        return out
    for i in range(out.shape[0]):
        idx = torch.randperm(width, generator=generator, device=generator.device)[:k]
        out[i, idx.to(out.device)] ^= 1
    return out


def adversarial_single_flip(
    method: str, codeword: torch.Tensor, true_label: torch.Tensor
) -> torch.Tensor:
    """Maximise decoded task error, choosing the first bit position on a tie."""
    base = _binary_batch(codeword, name="codeword")
    n, width = base.shape
    # Validate method-specific width even when the input contains no rows.
    decode(method, base[:0])
    if not isinstance(true_label, torch.Tensor):
        raise TypeError("true_label must be a torch.Tensor")
    if true_label.shape != (n,):
        raise ValueError("true_label must have shape (N,)")
    if true_label.is_complex() or not bool(
        (
            (true_label >= 0)
            & (true_label <= 15)
            & (true_label == true_label.to(torch.int64))
        ).all()
    ):
        raise ValueError("true_label must contain integer task identities from 0 to 15")
    labels = true_label.to(device=base.device, dtype=torch.int64)
    out = torch.empty_like(base)
    positions = torch.arange(width, device=base.device)
    for start in range(0, n, _SEARCH_BATCH_SIZE):
        batch = base[start : start + _SEARCH_BATCH_SIZE]
        candidates = batch[:, None, :].expand(-1, width, -1).clone()
        candidates[:, positions, positions] ^= 1
        predictions = task_label(decode(method, candidates.reshape(-1, width)).message)
        losses = (
            predictions.reshape(-1, width) != labels[start : start + len(batch), None]
        )
        selected = losses.to(torch.int64).argmax(dim=1)
        out[start : start + len(batch)] = candidates[
            torch.arange(len(batch), device=base.device), selected
        ]
    return out


def nearest_opposite_valid_codeword(method: str, message: torch.Tensor) -> torch.Tensor:
    """Choose the nearest different task identity, breaking ties by lowest label."""
    m = _binary_batch(message, 4, name="message")
    device = m.device
    all_messages = torch.tensor(
        [[(i >> b) & 1 for b in range(4)] for i in range(16)],
        dtype=torch.int64,
        device=device,
    )
    all_codes = encode(method, all_messages)
    all_labels = task_label(all_messages)
    original_codes = encode(method, m)
    original_labels = task_label(m)
    out = torch.empty_like(original_codes)
    for start in range(0, m.shape[0], _SEARCH_BATCH_SIZE):
        batch = original_codes[start : start + _SEARCH_BATCH_SIZE]
        distances = (batch[:, None, :] != all_codes[None, :, :]).sum(dim=-1)
        same_label = (
            original_labels[start : start + len(batch), None] == all_labels[None, :]
        )
        distances.masked_fill_(same_label, all_codes.shape[1] + 1)
        out[start : start + len(batch)] = all_codes[distances.argmin(dim=1)]
    return out
