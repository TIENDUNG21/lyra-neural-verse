"""Causal multi-head self-attention."""

from __future__ import annotations

import math

import numpy as np

from lyra.autograd import Tensor
from lyra.autograd import functional as F
from lyra.nn.layers import Dropout, Linear
from lyra.nn.module import Module


class CausalSelfAttention(Module):
    """Multi-head attention where position ``t`` only attends to ``<= t``.

    Shapes: input ``(B, T, C)``; heads ``H``; head width ``D = C / H``.
    One fused projection produces queries, keys and values.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        block_size: int,
        dropout: float,
        rng: np.random.Generator,
        residual_std: float = 0.02,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = Linear(d_model, 3 * d_model, rng=rng)
        self.proj = Linear(d_model, d_model, rng=rng, std=residual_std)
        self.attn_drop = Dropout(dropout, rng=rng)
        self.resid_drop = Dropout(dropout, rng=rng)
        # True above the diagonal: the future positions that must be hidden.
        self.future = np.triu(np.ones((block_size, block_size), dtype=bool), k=1)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        qkv = self.qkv.forward(x).reshape(B, T, 3, H, D)
        q = qkv[:, :, 0].transpose(0, 2, 1, 3)  # (B, H, T, D)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        scores = (q @ k.swapaxes(-1, -2)) * (1.0 / math.sqrt(D))  # (B, H, T, T)
        scores = F.masked_fill(scores, self.future[:T, :T], -np.inf)
        weights = self.attn_drop.forward(F.softmax(scores, axis=-1))
        y = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.resid_drop.forward(self.proj.forward(y))
