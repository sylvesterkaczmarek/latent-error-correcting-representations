from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import MessageEncoder, hard_message


def train_encoder(
    model: MessageEncoder,
    dataset,
    seed: int,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 2e-3,
    device: str = "cpu",
) -> list[float]:
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
        for x, message, _ in loader:
            x = x.to(device)
            message = message.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, message)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.shape[0]
            count += x.shape[0]
        history.append(total / max(count, 1))
    return history


@torch.no_grad()
def predict_messages(model: MessageEncoder, dataset, batch_size: int = 512, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred, true, labels = [], [], []
    for x, message, y in loader:
        logits = model(x.to(device))
        pred.append(hard_message(logits).cpu())
        true.append(message.to(torch.int64).cpu())
        labels.append(y.cpu())
    return torch.cat(pred), torch.cat(true), torch.cat(labels)
