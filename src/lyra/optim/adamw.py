"""AdamW: Adam with decoupled weight decay (Loshchilov and Hutter, 2019)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lyra.nn.module import Parameter


class AdamW:
    """Adam with bias correction and decoupled weight decay.

    Weight decay applies only to parameters with two or more dimensions
    (matrices and embeddings); biases and layer-norm gains are not decayed.
    """

    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self, lr: float | None = None) -> None:
        """Apply one update using the gradients currently stored on the parameters."""
        lr = self.lr if lr is None else lr
        self.step_count += 1
        t = self.step_count
        correction1 = 1.0 - self.beta1**t
        correction2 = 1.0 - self.beta2**t
        for p, m, v in zip(self.params, self.m, self.v, strict=True):
            g = p.grad
            if g is None:
                continue
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
            if self.weight_decay and p.ndim >= 2:
                p.data *= 1.0 - lr * self.weight_decay
            denom = np.sqrt(v / correction2) + self.eps
            p.data -= (lr / correction1) * (m / denom)

    def state_dict(self) -> dict[str, NDArray[Any]]:
        state: dict[str, NDArray[Any]] = {"step": np.array(self.step_count, dtype=np.int64)}
        for i, (m, v) in enumerate(zip(self.m, self.v, strict=True)):
            state[f"m.{i}"] = m.copy()
            state[f"v.{i}"] = v.copy()
        return state

    def load_state_dict(self, state: dict[str, NDArray[Any]]) -> None:
        self.step_count = int(state["step"])
        for i, p in enumerate(self.params):
            m, v = np.asarray(state[f"m.{i}"]), np.asarray(state[f"v.{i}"])
            if m.shape != p.shape or v.shape != p.shape:
                raise ValueError(f"optimizer state shape mismatch for parameter {i}")
            self.m[i] = m.astype(p.dtype, copy=True)
            self.v[i] = v.astype(p.dtype, copy=True)
