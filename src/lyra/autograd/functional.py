"""Neural-network primitives with fused, hand-derived backward passes.

These could be composed from the elementary ``Tensor`` operations, but fusing
them keeps the graph small and the memory traffic low, which matters for a
NumPy-only training loop. Each one is verified against finite differences in
``tests/autograd``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lyra.autograd.tensor import Array, Tensor

_GELU_C = math.sqrt(2.0 / math.pi)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax along ``axis``."""
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    out = exp / exp.sum(axis=axis, keepdims=True)

    def grad_fn(g: Array) -> tuple[Array]:
        return (out * (g - (g * out).sum(axis=axis, keepdims=True)),)

    return Tensor.make(out, (x,), grad_fn, "softmax")


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable log-softmax along ``axis``."""
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    out = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))

    def grad_fn(g: Array) -> tuple[Array]:
        return (g - np.exp(out) * g.sum(axis=axis, keepdims=True),)

    return Tensor.make(out, (x,), grad_fn, "log_softmax")


def gelu(x: Tensor) -> Tensor:
    """GELU with the tanh approximation used by GPT-2.

    Written with in-place NumPy operations: this is the largest elementwise
    tensor in the model (``batch x time x 4*d_model``).
    """
    a = x.data
    a2 = a * a
    t = a2 * 0.044715
    t += 1.0
    t *= a
    t *= _GELU_C
    np.tanh(t, out=t)  # t = tanh(c * (a + 0.044715 a^3))
    out = t + 1.0
    out *= a
    out *= 0.5

    def grad_fn(g: Array) -> tuple[Array]:
        d = t * t
        np.subtract(1.0, d, out=d)
        d *= a
        d *= 0.5 * _GELU_C
        d *= 1.0 + 3.0 * 0.044715 * a2
        d += 0.5 * (1.0 + t)
        d *= g
        return (d,)

    return Tensor.make(out, (x,), grad_fn, "gelu")


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """Layer normalization over the last axis with an affine transform."""
    a = x.data
    mean = a.mean(axis=-1, keepdims=True)
    centered = a - mean
    var = (centered * centered).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = centered * rstd
    w, b = weight.data, bias.data
    out = xhat * w + b

    def grad_fn(g: Array) -> tuple[Array, Array, Array]:
        flat_g = g.reshape(-1, g.shape[-1])
        flat_xhat = xhat.reshape(-1, xhat.shape[-1])
        grad_w = (flat_g * flat_xhat).sum(axis=0)
        grad_b = flat_g.sum(axis=0)
        gx = g * w
        grad_x = rstd * (
            gx - gx.mean(axis=-1, keepdims=True) - xhat * (gx * xhat).mean(axis=-1, keepdims=True)
        )
        return grad_x, grad_w, grad_b

    return Tensor.make(out, (x, weight, bias), grad_fn, "layer_norm")


def embedding(weight: Tensor, ids: NDArray[np.integer[Any]]) -> Tensor:
    """Gather rows of ``weight``; the gradient scatters back with accumulation."""
    index = np.asarray(ids)
    rows = weight.shape[0]
    if index.size and (int(index.min()) < 0 or int(index.max()) >= rows):
        raise IndexError(f"token id out of range for embedding with {rows} rows")
    w = weight.data

    def grad_fn(g: Array) -> tuple[Array]:
        full = np.zeros_like(w)
        np.add.at(full, index.reshape(-1), g.reshape(-1, w.shape[1]))
        return (full,)

    return Tensor.make(w[index], (weight,), grad_fn, "embedding")


def cross_entropy(
    logits: Tensor, targets: NDArray[np.integer[Any]], ignore_index: int = -100
) -> Tensor:
    """Mean negative log-likelihood of ``targets`` under ``softmax(logits)``.

    ``logits`` has shape ``(..., vocab)`` and ``targets`` the leading shape.
    Positions whose target equals ``ignore_index`` do not contribute.
    """
    vocab = logits.shape[-1]
    flat = logits.data.reshape(-1, vocab)
    flat_targets = np.asarray(targets).reshape(-1)
    if flat_targets.shape[0] != flat.shape[0]:
        raise ValueError("targets must match the leading shape of logits")
    keep = flat_targets != ignore_index
    count = int(keep.sum())
    rows = np.nonzero(keep)[0]
    cols = flat_targets[keep]

    shifted = flat - flat.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    logp = shifted - log_z
    total = -logp[rows, cols].sum()
    loss = np.asarray(total / count if count else 0.0, dtype=logits.dtype)

    def grad_fn(g: Array) -> tuple[Array]:
        if count == 0:
            return (np.zeros_like(logits.data),)
        grad = np.exp(logp)
        grad[rows, cols] -= 1.0
        grad[~keep] = 0.0
        grad *= g / count
        return (grad.reshape(logits.shape),)

    return Tensor.make(loss, (logits,), grad_fn, "cross_entropy")


def dropout(x: Tensor, p: float, rng: np.random.Generator, training: bool) -> Tensor:
    """Inverted dropout: zero each element with probability ``p`` and rescale."""
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability must be in [0, 1), got {p}")
    if not training or p == 0.0:
        return x
    scale = 1.0 / (1.0 - p)
    mask = (rng.random(x.shape, dtype=np.float32) >= p).astype(x.dtype) * scale
    return Tensor.make(x.data * mask, (x,), lambda g: (g * mask,), "dropout")


def masked_fill(x: Tensor, mask: NDArray[np.bool_], value: float) -> Tensor:
    """Replace entries where ``mask`` is true by ``value`` (no gradient there)."""
    keep = ~np.broadcast_to(mask, x.shape)
    out = np.where(keep, x.data, np.asarray(value, dtype=x.dtype))
    return Tensor.make(out, (x,), lambda g: (g * keep,), "masked_fill")
