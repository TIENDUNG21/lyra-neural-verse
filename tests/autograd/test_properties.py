"""Property-based checks: gradients stay correct for random broadcastable shapes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from lyra.autograd import Tensor, gradcheck, unbroadcast
from lyra.autograd import functional as F

dims = st.integers(min_value=1, max_value=4)


@st.composite
def broadcastable_pair(draw: st.DrawFn) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Two shapes that broadcast together: dims of b are 1, equal, or dropped."""
    shape_a = tuple(draw(st.lists(dims, min_size=1, max_size=3)))
    shape_b = tuple(draw(st.sampled_from([1, size])) for size in shape_a)
    drop = draw(st.integers(min_value=0, max_value=len(shape_b) - 1))
    if draw(st.booleans()):
        shape_a, shape_b = shape_b[drop:], shape_a
    return shape_a, shape_b[drop:]


OPS: list[Callable[[Tensor, Tensor], Tensor]] = [
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: a * b,
    lambda a, b: a / (b * b + 1.0),
    lambda a, b: (a * b).tanh() + a,
]


@settings(max_examples=60, deadline=None)
@given(pair=broadcastable_pair(), op=st.sampled_from(OPS), seed=st.integers(0, 2**16))
def test_binary_op_gradients_with_broadcasting(
    pair: tuple[tuple[int, ...], tuple[int, ...]],
    op: Callable[[Tensor, Tensor], Tensor],
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, size=pair[0])
    b = rng.uniform(-1, 1, size=pair[1])
    gradcheck(op, [a, b])


@settings(max_examples=60, deadline=None)
@given(pair=broadcastable_pair(), seed=st.integers(0, 2**16))
def test_unbroadcast_preserves_total(
    pair: tuple[tuple[int, ...], tuple[int, ...]], seed: int
) -> None:
    shape_a, shape_b = pair
    out_shape = np.broadcast_shapes(shape_a, shape_b)
    g = np.random.default_rng(seed).normal(size=out_shape)
    reduced = unbroadcast(g, shape_b)
    assert reduced.shape == shape_b
    np.testing.assert_allclose(reduced.sum(), g.sum())


@settings(max_examples=40, deadline=None)
@given(
    rows=st.integers(1, 5),
    vocab=st.integers(2, 7),
    scale=st.floats(0.1, 50.0),
    seed=st.integers(0, 2**16),
)
def test_cross_entropy_gradient_rows_sum_to_zero(
    rows: int, vocab: int, scale: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    logits = Tensor(rng.normal(size=(rows, vocab)) * scale, requires_grad=True)
    targets = rng.integers(0, vocab, size=rows)
    loss = F.cross_entropy(logits, targets)
    assert np.isfinite(loss.item())
    assert loss.item() >= 0.0
    loss.backward()
    assert logits.grad is not None
    np.testing.assert_allclose(logits.grad.sum(axis=-1), 0.0, atol=1e-12)
