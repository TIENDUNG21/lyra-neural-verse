"""A GPT-style decoder-only transformer language model."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lyra.autograd import Tensor
from lyra.autograd import functional as F
from lyra.nn.attention import CausalSelfAttention
from lyra.nn.layers import Dropout, Embedding, LayerNorm, Linear
from lyra.nn.module import Module, ModuleList


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters that define the network shape."""

    vocab_size: int
    block_size: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if min(self.vocab_size, self.block_size, self.d_model, self.n_layers) <= 0:
            raise ValueError("sizes must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown model config keys: {unknown}")
        return cls(**data)


class MLP(Module):
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, residual_std: float) -> None:
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.d_model
        self.fc = Linear(cfg.d_model, hidden, rng=rng)
        self.proj = Linear(hidden, cfg.d_model, rng=rng, std=residual_std)
        self.drop = Dropout(cfg.dropout, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop.forward(self.proj.forward(F.gelu(self.fc.forward(x))))


class Block(Module):
    """Pre-norm residual block: ``x + attn(ln(x))`` then ``x + mlp(ln(x))``."""

    def __init__(self, cfg: ModelConfig, rng: np.random.Generator) -> None:
        super().__init__()
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            block_size=cfg.block_size,
            dropout=cfg.dropout,
            rng=rng,
            residual_std=residual_std,
        )
        self.ln2 = LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg, rng, residual_std)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn.forward(self.ln1.forward(x))
        return x + self.mlp.forward(self.ln2.forward(x))


class LyraLM(Module):
    """Token + learned position embeddings, N blocks, final norm, tied output head."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        rng = np.random.default_rng(cfg.seed)
        self.tok_emb = Embedding(cfg.vocab_size, cfg.d_model, rng=rng)
        self.pos_emb = Embedding(cfg.block_size, cfg.d_model, rng=rng)
        self.drop = Dropout(cfg.dropout, rng=rng)
        self.blocks = ModuleList(Block(cfg, rng) for _ in range(cfg.n_layers))
        self.ln_f = LayerNorm(cfg.d_model)

    def forward(self, ids: NDArray[np.integer[Any]], last_only: bool = False) -> Tensor:
        """Return logits of shape ``(B, T, vocab)`` (or ``(B, 1, vocab)`` if ``last_only``)."""
        ids = np.asarray(ids)
        if ids.ndim != 2:
            raise ValueError("ids must have shape (batch, time)")
        seq_len = ids.shape[1]
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )
        x = self.tok_emb.forward(ids) + self.pos_emb.forward(np.arange(seq_len))
        x = self.drop.forward(x)
        for block in self.blocks:
            assert isinstance(block, Block)
            x = block.forward(x)
        x = self.ln_f.forward(x)
        if last_only:
            x = x[:, -1:]
        return x @ self.tok_emb.weight.T

    def loss(self, ids: NDArray[np.integer[Any]], targets: NDArray[np.integer[Any]]) -> Tensor:
        """Mean next-token cross-entropy; targets equal to -1 are ignored."""
        return F.cross_entropy(self.forward(ids), targets, ignore_index=-1)
