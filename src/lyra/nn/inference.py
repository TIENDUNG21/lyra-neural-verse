"""Fast inference for one sequence: plain NumPy, no autograd, with a KV cache.

The training model (``LyraLM``) recomputes the whole sequence on every call.
For generation that is wasteful: each new token only needs its own query
against the keys and values of the tokens before it. This engine caches the
per-layer keys and values, so feeding one token costs one row of work.

It computes exactly the same function as ``LyraLM`` in eval mode (tested to
float32 rounding), and the TypeScript engine in ``web/src/model`` mirrors it
line by line.

Context overflow: positions are learned embeddings, so the cache cannot slide.
When it is full, the engine re-reads the most recent ``block_size // 2``
tokens from scratch and continues from there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from lyra.nn.transformer import ModelConfig

FloatArray = NDArray[np.float32]
_GELU_C = math.sqrt(2.0 / math.pi)


def _layer_norm(x: FloatArray, weight: FloatArray, bias: FloatArray) -> FloatArray:
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = (centered * centered).mean(axis=-1, keepdims=True)
    return cast(FloatArray, centered / np.sqrt(var + 1e-5) * weight + bias)


def _gelu(x: FloatArray) -> FloatArray:
    return cast(FloatArray, 0.5 * x * (1.0 + np.tanh(_GELU_C * x * (1.0 + 0.044715 * x * x))))


class InferenceEngine:
    """Incremental decoder over float32 weights named like ``LyraLM.state_dict()``."""

    def __init__(self, config: ModelConfig, weights: dict[str, NDArray[Any]]) -> None:
        self.config = config
        self.w = {name: np.asarray(value, dtype=np.float32) for name, value in weights.items()}
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.tokens: list[int] = []
        self._k: list[FloatArray] = []
        self._v: list[FloatArray] = []
        self.reset()

    def reset(self) -> None:
        """Forget the context."""
        self.tokens = []
        shape = (self.n_heads, 0, self.head_dim)
        self._k = [np.zeros(shape, np.float32) for _ in range(self.config.n_layers)]
        self._v = [np.zeros(shape, np.float32) for _ in range(self.config.n_layers)]

    @property
    def length(self) -> int:
        return len(self.tokens)

    def feed(self, tokens: Sequence[int]) -> FloatArray:
        """Append ``tokens`` to the context; return logits for the next token."""
        if not tokens:
            raise ValueError("feed() needs at least one token")
        logits: FloatArray | None = None
        pending = list(tokens)
        while pending:
            room = self.config.block_size - self.length
            if room == 0:
                keep = self.tokens[-(self.config.block_size // 2) :]
                self.reset()
                self._forward(keep)
                room = self.config.block_size - self.length
            chunk, pending = pending[:room], pending[room:]
            logits = self._forward(chunk)
        assert logits is not None
        return logits

    def _forward(self, chunk: Sequence[int]) -> FloatArray:
        ids = np.asarray(chunk, dtype=np.int64)
        if int(np.min(ids)) < 0 or int(np.max(ids)) >= self.config.vocab_size:
            raise ValueError("token id out of range")
        start = self.length
        n = len(ids)
        w = self.w
        H, D = self.n_heads, self.head_dim
        x = w["tok_emb.weight"][ids] + w["pos_emb.weight"][start : start + n]
        # Query i (absolute position start+i) may see keys 0..start+i.
        future = np.triu(np.ones((n, start + n), dtype=bool), k=start + 1)
        for layer in range(self.config.n_layers):
            p = f"blocks.{layer}."
            h = _layer_norm(x, w[p + "ln1.weight"], w[p + "ln1.bias"])
            qkv = (h @ w[p + "attn.qkv.weight"] + w[p + "attn.qkv.bias"]).reshape(n, 3, H, D)
            q = qkv[:, 0].transpose(1, 0, 2)  # (H, n, D)
            k = np.concatenate([self._k[layer], qkv[:, 1].transpose(1, 0, 2)], axis=1)
            v = np.concatenate([self._v[layer], qkv[:, 2].transpose(1, 0, 2)], axis=1)
            self._k[layer], self._v[layer] = k, v
            scores = (q @ k.transpose(0, 2, 1)) * np.float32(1.0 / math.sqrt(D))
            scores = np.where(future, -np.inf, scores)
            scores = scores - scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=-1, keepdims=True)
            y = (weights @ v).transpose(1, 0, 2).reshape(n, H * D)
            x = x + (y @ w[p + "attn.proj.weight"] + w[p + "attn.proj.bias"])
            h = _layer_norm(x, w[p + "ln2.weight"], w[p + "ln2.bias"])
            h = _gelu(h @ w[p + "mlp.fc.weight"] + w[p + "mlp.fc.bias"])
            x = x + (h @ w[p + "mlp.proj.weight"] + w[p + "mlp.proj.bias"])
        self.tokens.extend(int(t) for t in ids)
        last = _layer_norm(x[-1], w["ln_f.weight"], w["ln_f.bias"])
        logits: FloatArray = (w["tok_emb.weight"] @ last).astype(np.float32)
        return logits
