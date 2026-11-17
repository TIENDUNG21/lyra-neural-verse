from __future__ import annotations

import numpy as np
import pytest

from lyra.autograd import Tensor, gradcheck
from tests.autograd.helpers import rand


@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [
        ((3, 4), (4, 5)),
        ((2, 3, 4), (4, 5)),
        ((2, 3, 4), (2, 4, 5)),
        ((2, 2, 3, 4), (2, 2, 4, 3)),
        ((1, 3, 4), (2, 4, 2)),
    ],
)
def test_matmul_forward_and_gradcheck(a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> None:
    a, b = rand(*a_shape, seed=1), rand(*b_shape, seed=2)
    np.testing.assert_allclose((Tensor(a) @ Tensor(b)).data, a @ b)
    gradcheck(lambda x, y: x @ y, [a, b])


def test_matmul_rejects_vectors() -> None:
    with pytest.raises(ValueError, match="at least 2-D"):
        _ = Tensor(np.ones(3)) @ Tensor(np.ones((3, 2)))


def test_matmul_with_constant_array_operand() -> None:
    w = rand(4, 2, seed=3)
    x = Tensor(rand(3, 4, seed=4), requires_grad=True)
    (x @ w).sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, np.ones((3, 2)) @ w.T)


def test_reshape_and_transpose_gradcheck() -> None:
    a = rand(2, 3, 4, seed=5)
    gradcheck(lambda x: x.reshape(4, 6) * 2.0, [a])
    gradcheck(lambda x: x.reshape(-1), [a])
    gradcheck(lambda x: x.transpose(2, 0, 1), [a])
    gradcheck(lambda x: x.swapaxes(-1, -2), [a])
    assert Tensor(a).T.shape == (4, 3, 2)


@pytest.mark.parametrize(
    "index",
    [
        (slice(None), 1),
        (Ellipsis, slice(0, 2)),
        (np.array([0, 1, 1, 0]),),
        (np.array([1, 0, 1]), np.array([2, 2, 0])),
    ],
)
def test_getitem_gradcheck(index: tuple[object, ...]) -> None:
    gradcheck(lambda x: x[index], [rand(2, 3, 4, seed=6)])


def test_getitem_with_repeated_indices_accumulates() -> None:
    t = Tensor(np.zeros(3), requires_grad=True)
    t[np.array([0, 0, 2])].sum().backward()
    assert t.grad is not None
    np.testing.assert_allclose(t.grad, [2.0, 0.0, 1.0])
