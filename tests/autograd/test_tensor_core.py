from __future__ import annotations

import numpy as np
import pytest

from lyra.autograd import Tensor, gradcheck, is_grad_enabled, no_grad
from tests.autograd.helpers import rand


def test_integer_input_is_promoted_to_float32() -> None:
    t = Tensor([1, 2, 3])
    assert t.data.dtype == np.float32
    assert t.shape == (3,)


def test_float64_input_keeps_dtype() -> None:
    t = Tensor(np.zeros(2, dtype=np.float64))
    assert t.dtype == np.float64


def test_add_mul_forward_matches_numpy() -> None:
    a, b = rand(2, 3, seed=1), rand(2, 3, seed=2)
    out = Tensor(a) * Tensor(b) + Tensor(a)
    np.testing.assert_allclose(out.data, a * b + a)


def test_scalar_and_reflected_operators() -> None:
    a = rand(3, seed=3)
    t = Tensor(a, requires_grad=True)
    out = (2.0 - t) / 4.0 + 1.0 / (t + 3.0) - (-t) * 0.5
    expected = (2.0 - a) / 4.0 + 1.0 / (a + 3.0) + a * 0.5
    np.testing.assert_allclose(out.data, expected)
    out.sum().backward()
    assert t.grad is not None
    np.testing.assert_allclose(t.grad, -0.25 - 1.0 / (a + 3.0) ** 2 + 0.5)


def test_grad_accumulates_when_tensor_is_reused() -> None:
    t = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    (t * t + t).sum().backward()
    assert t.grad is not None
    np.testing.assert_allclose(t.grad, 2 * t.data + 1)


def test_backward_twice_accumulates_into_leaf() -> None:
    t = Tensor(np.array([1.0, -2.0]), requires_grad=True)
    (t * 3.0).sum().backward()
    (t * 3.0).sum().backward()
    assert t.grad is not None
    np.testing.assert_allclose(t.grad, [6.0, 6.0])


def test_backward_on_non_scalar_requires_explicit_grad() -> None:
    t = Tensor(np.ones((2, 2)), requires_grad=True)
    with pytest.raises(RuntimeError, match="scalar"):
        (t * 2.0).backward()
    (t * 2.0).backward(np.ones((2, 2)))
    assert t.grad is not None


def test_backward_without_requires_grad_raises() -> None:
    with pytest.raises(RuntimeError, match="does not require grad"):
        Tensor([1.0]).sum().backward()


def test_no_grad_builds_no_graph() -> None:
    t = Tensor(np.ones(3), requires_grad=True)
    with no_grad():
        assert not is_grad_enabled()
        out = (t * 2.0).sum()
    assert is_grad_enabled()
    assert not out.requires_grad


def test_detach_cuts_the_graph() -> None:
    t = Tensor(np.ones(3), requires_grad=True)
    d = (t * 2.0).detach()
    assert not d.requires_grad
    np.testing.assert_allclose(d.data, 2.0)


def test_deep_chain_does_not_hit_recursion_limit() -> None:
    t = Tensor(np.array([1.0]), requires_grad=True)
    out = t
    for _ in range(5000):
        out = out + 0.0
    out.sum().backward()
    assert t.grad is not None
    np.testing.assert_allclose(t.grad, [1.0])


@pytest.mark.parametrize(
    ("name", "fn", "low"),
    [
        ("add", lambda a, b: a + b, -1.0),
        ("sub", lambda a, b: a - b, -1.0),
        ("mul", lambda a, b: a * b, -1.0),
        ("div", lambda a, b: a / b, 0.5),
    ],
)
def test_binary_ops_gradcheck(name: str, fn: object, low: float) -> None:
    del name
    gradcheck(fn, [rand(3, 4, seed=4), rand(3, 4, seed=5, low=low, high=2.0)])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fn", "low"),
    [
        (lambda a: a.exp(), -1.0),
        (lambda a: a.log(), 0.1),
        (lambda a: a.tanh(), -2.0),
        (lambda a: a**3, -1.0),
        (lambda a: a**0.5, 0.1),
        (lambda a: -a, -1.0),
        (lambda a: a.sum(), -1.0),
        (lambda a: a.sum(axis=1), -1.0),
        (lambda a: a.sum(axis=0, keepdims=True), -1.0),
        (lambda a: a.mean(), -1.0),
        (lambda a: a.mean(axis=-1, keepdims=True), -1.0),
        (lambda a: a.mean(axis=(0, 2)), -1.0),
    ],
)
def test_unary_ops_gradcheck(fn: object, low: float) -> None:
    gradcheck(fn, [rand(2, 3, 4, seed=6, low=low, high=2.0)])  # type: ignore[arg-type]


def test_broadcast_grad_sums_back_to_input_shape() -> None:
    a = Tensor(rand(4, 3, seed=7), requires_grad=True)
    b = Tensor(rand(3, seed=8), requires_grad=True)
    c = Tensor(rand(4, 1, seed=9), requires_grad=True)
    (a * b + c).sum().backward()
    assert b.grad is not None
    assert c.grad is not None
    assert b.grad.shape == (3,)
    assert c.grad.shape == (4, 1)
    np.testing.assert_allclose(b.grad, a.data.sum(axis=0))
    np.testing.assert_allclose(c.grad, np.ones((4, 1)) * 3)
