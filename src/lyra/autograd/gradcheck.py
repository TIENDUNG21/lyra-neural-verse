"""Finite-difference gradient checking.

``gradcheck`` compares the analytical gradient produced by ``Tensor.backward``
with a central-difference estimate. Inputs are promoted to float64 so that the
numerical estimate is accurate. The output is reduced with a fixed random
projection, which checks the full vector-Jacobian product rather than only the
gradient of a plain sum.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lyra.autograd.tensor import Tensor, no_grad


class GradcheckError(AssertionError):
    """Raised when analytical and numerical gradients disagree."""


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[NDArray[Any]],
    *,
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-4,
    seed: int = 1234,
) -> None:
    """Check the gradient of ``fn(*tensors)`` with respect to every input.

    Raises ``GradcheckError`` naming the input and element with the largest error.
    """
    arrays = [np.array(x, dtype=np.float64) for x in inputs]
    tensors = [Tensor(a, requires_grad=True) for a in arrays]
    out = fn(*tensors)
    projection = np.random.default_rng(seed).normal(size=out.shape)
    (out * projection).sum().backward()

    def objective() -> float:
        with no_grad():
            value = fn(*[Tensor(a) for a in arrays])
        return float(np.sum(value.data * projection))

    for index, (array, tensor) in enumerate(zip(arrays, tensors, strict=True)):
        analytic = tensor.grad if tensor.grad is not None else np.zeros_like(array)
        numeric = np.zeros_like(array)
        flat = array.reshape(-1)
        numeric_flat = numeric.reshape(-1)
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + eps
            plus = objective()
            flat[i] = original - eps
            minus = objective()
            flat[i] = original
            numeric_flat[i] = (plus - minus) / (2.0 * eps)
        if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
            error = np.abs(analytic - numeric)
            worst = np.unravel_index(int(np.argmax(error)), error.shape)
            raise GradcheckError(
                f"input {index}: max error {error.max():.3e} at {worst}: "
                f"analytic={analytic[worst]:.6e} numeric={numeric[worst]:.6e}"
            )
