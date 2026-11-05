"""Explicit decoder acceptance policies and extended Hamming(8,4) SECDED.

The decoder's accepted flag is a bounded-noise assumption, not semantic truth.
Triple errors can be miscorrected. Another valid codeword is accepted unchanged.
The four historical METHODS and DecodeResult interfaces remain unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import torch
from .codes import (
    METHODS,
    _binary_batch,
    encode,
    decode,
    encode_hamming74,
    hamming74_syndrome,
    task_label,
)

SELECTIVE_METHODS = (*METHODS, "hamming74_abstain", "hamming84_secded")


@dataclass(frozen=True)
class SelectiveDecode:
    candidate_message: torch.Tensor
    detected: torch.Tensor
    repaired: torch.Tensor
    accepted: torch.Tensor

    @property
    def task_labels(self):
        """Return -1 for abstention; candidate bits must never imply acceptance."""
        return torch.where(self.accepted, task_label(self.candidate_message), -1)


def encode_hamming84(message):
    code = encode_hamming74(message)
    return torch.cat((code, (code.sum(dim=1) % 2)[:, None]), dim=1)


def decode_hamming84(codeword):
    code = _binary_batch(codeword, 8, name="codeword").clone()
    syndrome = hamming74_syndrome(code[:, :7])
    odd = code.sum(dim=1) % 2 == 1
    detected = (syndrome != 0) | odd
    uncorrectable = (syndrome != 0) & ~odd
    rows = torch.nonzero(odd, as_tuple=False).flatten()
    if rows.numel():
        positions = torch.where(syndrome[rows] == 0, 7, syndrome[rows] - 1)
        code[rows, positions] ^= 1
    return SelectiveDecode(code[:, [2, 4, 5, 6]], detected, odd.clone(), ~uncorrectable)


def selective_encode(method, message):
    if method not in SELECTIVE_METHODS:
        raise ValueError("unknown selective decoding method")
    if method == "hamming84_secded":
        return encode_hamming84(message)
    return encode(
        "hamming74_detect" if method == "hamming74_abstain" else method, message
    )


def selective_decode(method, codeword):
    if method not in SELECTIVE_METHODS:
        raise ValueError("unknown selective decoding method")
    if method == "hamming84_secded":
        return decode_hamming84(codeword)
    base = decode(
        "hamming74_detect" if method == "hamming74_abstain" else method, codeword
    )
    accepted = (
        ~base.detected
        if method == "hamming74_abstain"
        else torch.ones_like(base.detected)
    )
    return SelectiveDecode(base.message, base.detected, base.corrected, accepted)


def outcome_counts(decision, reference):
    message = _binary_batch(decision.candidate_message, 4, name="candidate_message")
    reference = _binary_batch(reference, 4, name="reference")
    if (
        message.shape != reference.shape
        or not len(message)
        or message.device != reference.device
    ):
        raise ValueError(
            "reference and candidate must be nonempty, same-device matching batches"
        )
    n = len(message)
    for flags in (decision.detected, decision.repaired, decision.accepted):
        if (
            not isinstance(flags, torch.Tensor)
            or flags.dtype != torch.bool
            or flags.shape != (n,)
            or flags.device != message.device
        ):
            raise ValueError("decision flags must be same-device boolean vectors")
    if (decision.repaired & ~decision.detected).any() or (
        decision.repaired & ~decision.accepted
    ).any():
        raise ValueError("repair requires detected and accepted output")
    correct = (message == reference).all(dim=1)
    accepted = int(decision.accepted.sum().item())
    correct_accepted = int((correct & decision.accepted).sum().item())
    return {
        "n": n,
        "accepted": accepted,
        "abstained": n - accepted,
        "correct_accepted": correct_accepted,
        "wrong_accepted": accepted - correct_accepted,
        "detected": int(decision.detected.sum().item()),
        "repair_attempted": int(decision.repaired.sum().item()),
        "repair_to_reference": int((decision.repaired & correct).sum().item()),
        "candidate_correct": int(correct.sum().item()),
        "coverage": accepted / n,
        "task_success_rate": correct_accepted / n,
        "wrong_acceptance_rate": (accepted - correct_accepted) / n,
        "detection_rate": int(decision.detected.sum().item()) / n,
        "conditional_accuracy": correct_accepted / accepted if accepted else None,
    }


def messages(device="cpu"):
    return torch.tensor(
        [[(i >> bit) & 1 for bit in range(4)] for i in range(16)],
        device=device,
        dtype=torch.int64,
    )


def nearest_valid_replacement(method, message):
    m = _binary_batch(message, 4, name="message")
    codebook = selective_encode(method, messages(m.device))
    original = selective_encode(method, m)
    distances = (original[:, None] != codebook[None]).sum(dim=-1)
    distances[torch.arange(len(m), device=m.device), task_label(m)] = (
        codebook.shape[1] + 1
    )
    return codebook[distances.argmin(dim=1)]


def adversarial_exact_flip(method, codeword, reference, k):
    """Prefer an accepted wrong identity, then abstention, with first-pattern ties.

    This targets the supplied reference only. The study uses pre-corruption
    predictions, so ground-truth accuracy is not a worst-case ground-truth attack.
    """
    codeword = _binary_batch(codeword, name="codeword")
    reference = _binary_batch(reference, 4, name="reference")
    if len(codeword) != len(reference) or codeword.device != reference.device:
        raise ValueError("attack reference must match the codeword batch")
    selective_decode(method, codeword[:0])  # Validate width even for an empty batch.
    if type(k) is not int or k not in (1, 2):
        raise ValueError("attack supports exact one- or two-bit budgets")
    patterns = tuple(combinations(range(codeword.shape[1]), k))
    output = torch.empty_like(codeword)
    for start in range(0, len(codeword), 256):
        batch = codeword[start : start + 256]
        candidates = batch[:, None].expand(-1, len(patterns), -1).clone()
        for index, pattern in enumerate(patterns):
            candidates[:, index, list(pattern)] ^= 1
        decision = selective_decode(method, candidates.reshape(-1, codeword.shape[1]))
        correct = (
            decision.candidate_message.reshape(len(batch), len(patterns), 4)
            == reference[start : start + len(batch), None]
        ).all(dim=2)
        accepted = decision.accepted.reshape(len(batch), len(patterns))
        loss = 2 * (accepted & ~correct).long() + (~accepted).long()
        output[start : start + len(batch)] = candidates[
            torch.arange(len(batch), device=batch.device), loss.argmax(dim=1)
        ]
    return output
