from __future__ import annotations

from dataclasses import dataclass
import torch

from .codes import task_label


@dataclass(frozen=True)
class EvalMetrics:
    message_bit_accuracy: float
    task_accuracy: float
    detection_rate: float
    correction_rate: float


def evaluate(decoded_message: torch.Tensor, true_message: torch.Tensor, detected: torch.Tensor, corrected: torch.Tensor) -> EvalMetrics:
    bit_acc = (decoded_message == true_message).float().mean().item()
    task_acc = (task_label(decoded_message) == task_label(true_message)).float().mean().item()
    return EvalMetrics(
        message_bit_accuracy=float(bit_acc),
        task_accuracy=float(task_acc),
        detection_rate=float(detected.float().mean().item()),
        correction_rate=float(corrected.float().mean().item()),
    )
