"""Reverse-mode automatic differentiation on top of NumPy."""

from lyra.autograd.gradcheck import GradcheckError, gradcheck
from lyra.autograd.tensor import Tensor, as_tensor, is_grad_enabled, no_grad, unbroadcast

__all__ = [
    "GradcheckError",
    "Tensor",
    "as_tensor",
    "gradcheck",
    "is_grad_enabled",
    "no_grad",
    "unbroadcast",
]
