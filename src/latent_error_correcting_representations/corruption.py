from __future__ import annotations

import torch

from .codes import decode, task_label, encode


def flip_exactly_k(codeword: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    out = codeword.clone().to(torch.int64)
    if k <= 0:
        return out
    width = out.shape[1]
    k = min(k, width)
    for i in range(out.shape[0]):
        idx = torch.randperm(width, generator=generator)[:k]
        out[i, idx] ^= 1
    return out


def adversarial_single_flip(method: str, codeword: torch.Tensor, true_label: torch.Tensor) -> torch.Tensor:
    """Choose a one-bit corruption that maximizes task error after decoding."""
    base = codeword.to(torch.int64)
    n, width = base.shape
    out = base.clone()
    for i in range(n):
        best = None
        best_loss = -1
        for j in range(width):
            candidate = base[i : i + 1].clone()
            candidate[0, j] ^= 1
            pred = task_label(decode(method, candidate).message)[0]
            loss = int(pred.item() != int(true_label[i].item()))
            if loss > best_loss:
                best_loss = loss
                best = candidate
                if loss == 1:
                    break
        out[i] = best[0]
    return out


def nearest_opposite_valid_codeword(method: str, message: torch.Tensor) -> torch.Tensor:
    """Move to the nearest valid codeword whose decoded parity label is opposite."""
    m = message.to(torch.int64)
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
    out = original_codes.clone()
    for i in range(m.shape[0]):
        mask = all_labels != original_labels[i]
        candidates = all_codes[mask]
        distances = (candidates != original_codes[i]).sum(dim=1)
        out[i] = candidates[torch.argmin(distances)]
    return out
