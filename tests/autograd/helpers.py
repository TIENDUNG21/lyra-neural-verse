"""Shared helpers for autograd tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def rand(*shape: int, seed: int = 0, low: float = -1.0, high: float = 1.0) -> NDArray[Any]:
    """Deterministic float64 array for gradient checks."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=shape).astype(np.float64)
