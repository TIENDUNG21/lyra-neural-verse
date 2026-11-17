"""Basic layers: Linear, Embedding, LayerNorm and Dropout."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from lyra.autograd import Tensor
from lyra.autograd import functional as F
from lyra.nn.module import Module, Parameter


class Linear(Module):
    """``y = x @ W + b`` with ``W`` of shape ``(in_features, out_features)``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rng: np.random.Generator,
        bias: bool = True,
        std: float = 0.02,
    ) -> None:
        super().__init__()
        self.weight = Parameter(
            (rng.normal(0.0, std, size=(in_features, out_features))).astype(np.float32)
        )
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class Embedding(Module):
    """A lookup table mapping integer ids to learned vectors."""

    def __init__(
        self, num_embeddings: int, dim: int, *, rng: np.random.Generator, std: float = 0.02
    ) -> None:
        super().__init__()
        self.weight = Parameter(rng.normal(0.0, std, size=(num_embeddings, dim)).astype(np.float32))

    def forward(self, ids: NDArray[np.integer[Any]]) -> Tensor:
        return F.embedding(self.weight, ids)


class LayerNorm(Module):
    """Layer normalization over the last axis with learned scale and shift."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = Parameter(np.ones(dim, dtype=np.float32))
        self.bias = Parameter(np.zeros(dim, dtype=np.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight, self.bias, self.eps)


class Dropout(Module):
    """Inverted dropout that is active only in training mode."""

    def __init__(self, p: float, *, rng: np.random.Generator) -> None:
        super().__init__()
        self.p = p
        self.rng = rng

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.rng, self.training)
