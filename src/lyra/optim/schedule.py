"""Learning-rate schedule: linear warmup followed by cosine decay."""

from __future__ import annotations

import math


def warmup_cosine(
    step: int, *, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int
) -> float:
    """Learning rate for a zero-based ``step``.

    Rises linearly to ``base_lr`` over ``warmup_steps``, then follows half a
    cosine down to ``min_lr`` at ``total_steps`` and stays there.
    """
    if total_steps < warmup_steps or warmup_steps < 0:
        raise ValueError("total_steps must be >= warmup_steps >= 0")
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    span = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / span
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
