from __future__ import annotations

import torch
from torch import nn


class MessageEncoder(nn.Module):
    """Small MLP that learns a four-bit semantic bottleneck."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 48) -> None:
        super().__init__()
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
    return (torch.sigmoid(logits) >= 0.5).to(torch.int64)
