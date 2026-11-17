"""Global gradient-norm clipping."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from lyra.nn.module import Parameter


def clip_grad_norm(params: Sequence[Parameter], max_norm: float) -> float:
    """Scale all gradients so their joint L2 norm is at most ``max_norm``.

    Returns the norm before clipping (useful for logging).
    """
    grads = [p.grad for p in params if p.grad is not None]
    total = math.sqrt(sum(float(np.square(g, dtype=np.float64).sum()) for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g *= scale
    return total
