from __future__ import annotations

from numbers import Integral

import torch
from torch import nn


class MessageEncoder(nn.Module):
    """Small MLP that learns a four-bit semantic bottleneck."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 48) -> None:
        super().__init__()
        for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def hard_message(logits: torch.Tensor) -> torch.Tensor:
    """Threshold finite logits, assigning an exact zero to bit one."""
    if logits.is_complex() or not torch.isfinite(logits).all():
        raise ValueError("message logits must be finite real numbers")
    # Sigmoid can round a small negative logit to exactly 0.5.
    return (logits >= 0).to(torch.int64)
