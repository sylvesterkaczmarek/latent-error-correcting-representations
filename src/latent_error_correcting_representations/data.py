from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class DatasetBundle:
    train: TensorDataset
    test: TensorDataset
    projection: torch.Tensor


def _make_split(
    n: int,
    projection: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    message = rng.integers(0, 2, size=(n, 4), dtype=np.int64)
    bipolar = (2 * message - 1).astype(np.float32)
    x = bipolar @ projection.T
    x += rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    y = (message * np.array([1, 2, 4, 8], dtype=np.int64)).sum(axis=1).astype(np.int64)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(message, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def make_dataset(
    seed: int,
    n_train: int = 2500,
    n_test: int = 800,
    input_dim: int = 16,
    noise_std: float = 0.35,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, size=(input_dim, 4)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=0, keepdims=True) + 1e-8
    projection = raw * 3.0
    tr = _make_split(n_train, projection, noise_std, rng)
    te = _make_split(n_test, projection, noise_std, rng)
    return DatasetBundle(
        train=TensorDataset(*tr),
        test=TensorDataset(*te),
        projection=torch.tensor(projection, dtype=torch.float32),
    )
