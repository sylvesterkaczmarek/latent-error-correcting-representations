from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import MessageEncoder, hard_message
from .seed import validate_seed


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_batch(x: torch.Tensor, message: torch.Tensor, labels: torch.Tensor) -> None:
    if x.ndim != 2 or x.shape[1] == 0 or x.is_complex() or not torch.isfinite(x).all():
        raise ValueError("observations must be a finite real matrix with non-zero width")
    if message.shape != (x.shape[0], 4) or message.is_complex() or not torch.all((message == 0) | (message == 1)):
        raise ValueError("messages must contain four binary bits per observation")
    if labels.shape != (x.shape[0],) or labels.is_complex():
        raise ValueError("labels must contain one task identity per observation")
    expected = (message.to(torch.int64) * message.new_tensor([1, 2, 4, 8], dtype=torch.int64)).sum(dim=1)
    if not torch.equal(labels.to(expected.device), expected):
        raise ValueError("labels must match the four-bit task identities")


def _validate_logits(logits: torch.Tensor, n: int) -> None:
    if logits.shape != (n, 4) or logits.is_complex() or not torch.isfinite(logits).all():
        raise ValueError("encoder must produce four finite real logits per observation")


def train_encoder(
    model: MessageEncoder,
    dataset,
    seed: int,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 2e-3,
    device: str = "cpu",
) -> list[float]:
    _positive_integer(epochs, "epochs")
    _positive_integer(batch_size, "batch_size")
    seed = validate_seed(seed)
    if isinstance(lr, bool) or not isinstance(lr, Real) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if len(dataset) == 0:
        raise ValueError("training dataset must not be empty")
    model.to(device)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    history: list[float] = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for x, message, labels in loader:
            _validate_batch(x, message, labels)
            x = x.to(device)
            message = message.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            _validate_logits(logits, x.shape[0])
            loss = loss_fn(logits, message)
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss is not finite")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("training gradients are not finite")
            opt.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("training produced non-finite model parameters")
            total += float(loss.item()) * x.shape[0]
            count += x.shape[0]
        history.append(total / count)
    return history


@torch.no_grad()
def predict_messages(model: MessageEncoder, dataset, batch_size: int = 512, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _positive_integer(batch_size, "batch_size")
    if len(dataset) == 0:
        raise ValueError("prediction dataset must not be empty")
    training_modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval().to(device)
        # Even an unshuffled DataLoader draws a base seed when iterated.
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator().manual_seed(0))
        pred, true, labels = [], [], []
        for x, message, y in loader:
            _validate_batch(x, message, y)
            logits = model(x.to(device))
            _validate_logits(logits, x.shape[0])
            pred.append(hard_message(logits).cpu())
            true.append(message.to(torch.int64).cpu())
            labels.append(y.cpu())
        return torch.cat(pred), torch.cat(true), torch.cat(labels)
    finally:
        for module, training in training_modes:
            module.training = training
