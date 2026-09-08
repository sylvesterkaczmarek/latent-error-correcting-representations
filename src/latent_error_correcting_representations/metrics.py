from __future__ import annotations

from dataclasses import dataclass
import torch

from .codes import _binary_batch, task_label


@dataclass(frozen=True)
class EvalMetrics:
    message_bit_accuracy: float
    task_accuracy: float
    detection_rate: float
    correction_rate: float


def evaluate(decoded_message: torch.Tensor, true_message: torch.Tensor, detected: torch.Tensor, corrected: torch.Tensor) -> EvalMetrics:
    decoded_message = _binary_batch(decoded_message, 4, name="decoded_message")
    true_message = _binary_batch(true_message, 4, name="true_message")
    if decoded_message.shape != true_message.shape or decoded_message.shape[0] == 0:
        raise ValueError("message batches must have matching non-empty shapes")
    if decoded_message.device != true_message.device:
        raise ValueError("message batches must use the same device")
    for name, flags in (("detected", detected), ("corrected", corrected)):
        if not isinstance(flags, torch.Tensor) or flags.shape != (decoded_message.shape[0],):
            raise ValueError(f"{name} must contain one binary flag per message")
        if flags.is_complex() or not bool(((flags == 0) | (flags == 1)).all()):
            raise ValueError(f"{name} flags must be binary and finite")
    bit_acc = (decoded_message == true_message).float().mean().item()
    task_acc = (task_label(decoded_message) == task_label(true_message)).float().mean().item()
    return EvalMetrics(
        message_bit_accuracy=float(bit_acc),
        task_accuracy=float(task_acc),
        detection_rate=float(detected.float().mean().item()),
        correction_rate=float(corrected.float().mean().item()),
    )
