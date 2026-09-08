from __future__ import annotations

import random
from numbers import Integral

import numpy as np
import torch


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer from 0 through 2**32 - 1")
    return int(seed)


def seed_everything(seed: int) -> None:
    seed = validate_seed(seed)
    # Fail explicitly if deterministic execution cannot be requested, before
    # changing any generator state.
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
