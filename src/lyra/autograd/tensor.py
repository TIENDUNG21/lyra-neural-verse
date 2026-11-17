"""A NumPy-backed tensor with reverse-mode automatic differentiation.

Every operation that involves a tensor with ``requires_grad=True`` records its
parents and a backward function on the result. ``Tensor.backward`` walks that
graph in reverse topological order and accumulates gradients on the leaves.

Gradients are broadcasting-aware: when an operand was broadcast in the forward
pass, its gradient is summed back to the operand's original shape.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

Array: TypeAlias = NDArray[Any]
GradFn: TypeAlias = Callable[[Array], Sequence[Array | None]]
Index: TypeAlias = Any

_grad_enabled = True


def is_grad_enabled() -> bool:
    """Return whether new operations currently record a graph."""
    return _grad_enabled


@contextlib.contextmanager
def no_grad() -> Iterator[None]:
    """Context manager that disables graph recording (inference mode)."""
    global _grad_enabled
    previous = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = previous


def unbroadcast(grad: Array, shape: tuple[int, ...]) -> Array:
    """Sum ``grad`` over the axes that were broadcast to produce it from ``shape``."""
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, size in enumerate(shape) if size == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


def _is_basic_index(index: Index) -> bool:
    items = index if isinstance(index, tuple) else (index,)
    return all(
        isinstance(item, (int, np.integer, slice)) or item is None or item is Ellipsis
        for item in items
    )


class Tensor:
    """An n-dimensional array that can record operations for differentiation."""

    __slots__ = ("_grad_fn", "_parents", "data", "grad", "op", "requires_grad")
    __array_priority__ = 1000.0  # make ``ndarray (op) Tensor`` defer to Tensor

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        *,
        dtype: DTypeLike | None = None,
    ) -> None:
        array = np.asarray(data.data if isinstance(data, Tensor) else data, dtype=dtype)
        if not np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        self.data: Array = array
        self.grad: Array | None = None
        self.requires_grad = requires_grad
        self.op = "leaf"
        self._parents: tuple[Tensor, ...] = ()
        self._grad_fn: GradFn | None = None

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        return int(self.data.ndim)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    @property
    def size(self) -> int:
        return int(self.data.size)

    def __repr__(self) -> str:
        flag = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, op={self.op}{flag})"

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def item(self) -> float:
        return float(self.data.item())

    def numpy(self) -> Array:
        return self.data

    def detach(self) -> Tensor:
        """Return a tensor sharing data but cut from the graph."""
        return Tensor(self.data)

    def zero_grad(self) -> None:
        self.grad = None

    @staticmethod
    def make(data: Array, parents: Sequence[Tensor], grad_fn: GradFn, op: str) -> Tensor:
        """Create the result of an operation, recording the graph when needed."""
        out = Tensor(data)
        out.op = op
        if _grad_enabled and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = tuple(parents)
            out._grad_fn = grad_fn
        return out

    # ---------------------------------------------------------------- backward
    def backward(self, grad: ArrayLike | None = None) -> None:
        """Back-propagate from this tensor, accumulating ``.grad`` on leaves."""
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.size != 1:
                raise RuntimeError("grad must be given explicitly for a non-scalar output")
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(grad, dtype=self.dtype)
            if seed.shape != self.shape:
                raise ValueError(f"grad shape {seed.shape} does not match tensor {self.shape}")

        order = self._topological_order()
        grads: dict[int, Array] = {id(self): seed}
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._grad_fn is None:
                node.grad = g.copy() if node.grad is None else node.grad + g
                continue
            for parent, parent_grad in zip(node._parents, node._grad_fn(g), strict=True):
                if parent_grad is None or not parent.requires_grad:
                    continue
                key = id(parent)
                grads[key] = grads[key] + parent_grad if key in grads else parent_grad

    def _topological_order(self) -> list[Tensor]:
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if parent.requires_grad and id(parent) not in visited:
                    stack.append((parent, False))
        return order

    # ------------------------------------------------------------- arithmetic
    def __add__(self, other: TensorLike) -> Tensor:
        o = as_tensor(other, self.dtype)
        a_shape, b_shape = self.shape, o.shape
        return Tensor.make(
            self.data + o.data,
            (self, o),
            lambda g: (unbroadcast(g, a_shape), unbroadcast(g, b_shape)),
            "add",
        )

    def __radd__(self, other: TensorLike) -> Tensor:
        return as_tensor(other, self.dtype) + self

    def __sub__(self, other: TensorLike) -> Tensor:
        o = as_tensor(other, self.dtype)
        a_shape, b_shape = self.shape, o.shape
        return Tensor.make(
            self.data - o.data,
            (self, o),
            lambda g: (unbroadcast(g, a_shape), unbroadcast(-g, b_shape)),
            "sub",
        )

    def __rsub__(self, other: TensorLike) -> Tensor:
        return as_tensor(other, self.dtype) - self

    def __mul__(self, other: TensorLike) -> Tensor:
        o = as_tensor(other, self.dtype)
        a, b = self.data, o.data
        return Tensor.make(
            a * b,
            (self, o),
            lambda g: (unbroadcast(g * b, a.shape), unbroadcast(g * a, b.shape)),
            "mul",
        )

    def __rmul__(self, other: TensorLike) -> Tensor:
        return as_tensor(other, self.dtype) * self

    def __truediv__(self, other: TensorLike) -> Tensor:
        o = as_tensor(other, self.dtype)
        a, b = self.data, o.data
        return Tensor.make(
            a / b,
            (self, o),
            lambda g: (unbroadcast(g / b, a.shape), unbroadcast(-g * a / (b * b), b.shape)),
            "div",
        )

    def __rtruediv__(self, other: TensorLike) -> Tensor:
        return as_tensor(other, self.dtype) / self

    def __neg__(self) -> Tensor:
        return Tensor.make(-self.data, (self,), lambda g: (-g,), "neg")

    def __pow__(self, exponent: float) -> Tensor:
        a = self.data
        p = float(exponent)
        return Tensor.make(a**p, (self,), lambda g: (g * p * a ** (p - 1.0),), "pow")

    # ------------------------------------------------------------ elementwise
    def exp(self) -> Tensor:
        out = np.exp(self.data)
        return Tensor.make(out, (self,), lambda g: (g * out,), "exp")

    def log(self) -> Tensor:
        a = self.data
        return Tensor.make(np.log(a), (self,), lambda g: (g / a,), "log")

    def tanh(self) -> Tensor:
        out = np.tanh(self.data)
        return Tensor.make(out, (self,), lambda g: (g * (1.0 - out * out),), "tanh")

    # ------------------------------------------------------------- reductions
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        shape = self.shape
        axes = _normalize_axes(axis, len(shape))

        def grad_fn(g: Array) -> tuple[Array]:
            if not keepdims:
                g = np.expand_dims(g, axes)
            return (np.broadcast_to(g, shape),)

        if keepdims:
            total = np.sum(self.data, axis=axes, keepdims=True)
        else:
            total = np.asarray(np.sum(self.data, axis=axes))
        return Tensor.make(total, (self,), grad_fn, "sum")

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        axes = _normalize_axes(axis, self.ndim)
        count = int(np.prod([self.shape[a] for a in axes])) if axes else 1
        return self.sum(axis=axes, keepdims=keepdims) * (1.0 / count)

    # ----------------------------------------------------------------- matmul
    def __matmul__(self, other: TensorLike) -> Tensor:
        o = as_tensor(other, self.dtype)
        a, b = self.data, o.data
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("matmul operands must be at least 2-D")

        def grad_fn(g: Array) -> tuple[Array, Array]:
            grad_a = unbroadcast(g @ np.swapaxes(b, -1, -2), a.shape)
            if b.ndim == 2:
                # Common case x(..., k) @ W(k, n): fold the batch into one GEMM.
                grad_b = a.reshape(-1, a.shape[-1]).T @ g.reshape(-1, g.shape[-1])
            else:
                grad_b = unbroadcast(np.swapaxes(a, -1, -2) @ g, b.shape)
            return grad_a, grad_b

        return Tensor.make(a @ b, (self, o), grad_fn, "matmul")

    def __rmatmul__(self, other: TensorLike) -> Tensor:
        return as_tensor(other, self.dtype) @ self

    # ------------------------------------------------------------ shape ops
    def reshape(self, *shape: int) -> Tensor:
        original = self.shape
        return Tensor.make(
            self.data.reshape(shape), (self,), lambda g: (g.reshape(original),), "reshape"
        )

    def transpose(self, *axes: int) -> Tensor:
        order = tuple(axes) if axes else tuple(reversed(range(self.ndim)))
        inverse = tuple(int(i) for i in np.argsort(order))
        return Tensor.make(
            self.data.transpose(order), (self,), lambda g: (g.transpose(inverse),), "transpose"
        )

    def swapaxes(self, axis1: int, axis2: int) -> Tensor:
        return Tensor.make(
            np.swapaxes(self.data, axis1, axis2),
            (self,),
            lambda g: (np.swapaxes(g, axis1, axis2),),
            "swapaxes",
        )

    @property
    def T(self) -> Tensor:  # noqa: N802 - NumPy naming
        return self.transpose()

    def __getitem__(self, index: Index) -> Tensor:
        shape, dtype = self.shape, self.dtype
        basic = _is_basic_index(index)

        def grad_fn(g: Array) -> tuple[Array]:
            full = np.zeros(shape, dtype=dtype)
            if basic:
                full[index] = g
            else:
                np.add.at(full, index, g)
            return (full,)

        return Tensor.make(self.data[index], (self,), grad_fn, "getitem")


TensorLike: TypeAlias = Tensor | float | int | Array


def as_tensor(value: TensorLike, dtype: DTypeLike | None = None) -> Tensor:
    """Wrap ``value`` in a constant tensor unless it already is one."""
    if isinstance(value, Tensor):
        return value
    return Tensor(np.asarray(value, dtype=dtype))


def _normalize_axes(axis: int | tuple[int, ...] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else axis
    return tuple(sorted(a % ndim for a in axes))
