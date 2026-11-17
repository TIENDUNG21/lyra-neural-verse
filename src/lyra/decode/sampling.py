"""Next-token sampling with temperature, top-k and top-p (nucleus) filtering.

The arithmetic is written so that the TypeScript port performs the same
floating-point operations in the same order: candidates are sorted by
(logit desc, id asc), probabilities are unnormalized ``exp`` values summed
left to right, and the draw compares ``u * total`` against running sums.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lyra.decode.rng import Mulberry32


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float = 0.9
    top_k: int = 40
    top_p: float = 0.95

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables it)")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


def ranked_candidates(logits: NDArray[Any], allowed: NDArray[np.bool_]) -> list[int]:
    """Allowed token ids sorted by logit (descending), ties by id (ascending)."""
    ids = np.flatnonzero(allowed)
    if ids.size == 0:
        raise ValueError("no token is allowed at this step")
    values = logits[ids].astype(np.float64)
    order = np.lexsort((ids, -values))
    return [int(i) for i in ids[order]]


def sample_token(
    logits: NDArray[Any],
    allowed: NDArray[np.bool_],
    settings: SamplingSettings,
    rng: Mulberry32,
) -> int:
    """Pick the next token among ``allowed``; consumes one random number unless greedy."""
    ranked = ranked_candidates(logits, allowed)
    if settings.temperature == 0.0:
        return ranked[0]
    if settings.top_k:
        ranked = ranked[: settings.top_k]
    top = float(logits[ranked[0]])
    weights = [math.exp((float(logits[i]) - top) / settings.temperature) for i in ranked]
    running: list[float] = []
    total = 0.0
    for w in weights:
        total += w
        running.append(total)
    cutoff = len(ranked)
    if settings.top_p < 1.0:
        for n, value in enumerate(running):
            if value >= settings.top_p * total:
                cutoff = n + 1
                break
    threshold = rng.random() * running[cutoff - 1]
    for n in range(cutoff):
        if threshold < running[n]:
            return ranked[n]
    return ranked[cutoff - 1]
