from __future__ import annotations

import math

import numpy as np
import pytest

from lyra.autograd import Tensor, gradcheck
from lyra.autograd import functional as F
from tests.autograd.helpers import rand


def test_softmax_rows_sum_to_one_and_are_stable() -> None:
    logits = np.array([[1e4, 1e4 - 1.0, -1e4], [0.0, 0.0, 0.0]])
    probs = F.softmax(Tensor(logits), axis=-1).data
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0)
    np.testing.assert_allclose(probs[1], 1.0 / 3.0)


def test_log_softmax_matches_log_of_softmax() -> None:
    x = rand(3, 5, seed=1) * 10.0
    np.testing.assert_allclose(
        F.log_softmax(Tensor(x)).data, np.log(F.softmax(Tensor(x)).data), atol=1e-12
    )


def test_gelu_reference_values() -> None:
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    expected = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    np.testing.assert_allclose(F.gelu(Tensor(x)).data, expected)
    assert F.gelu(Tensor(np.array([0.0]))).item() == 0.0


def test_layer_norm_normalizes_last_axis() -> None:
    x = rand(4, 8, seed=2) * 5.0 + 3.0
    y = F.layer_norm(Tensor(x), Tensor(np.ones(8)), Tensor(np.zeros(8)), eps=1e-12).data
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-10)
    np.testing.assert_allclose(y.var(axis=-1), 1.0, atol=1e-8)


def test_embedding_gathers_rows() -> None:
    w = rand(5, 3, seed=3)
    ids = np.array([[4, 0], [0, 2]])
    np.testing.assert_allclose(F.embedding(Tensor(w), ids).data, w[ids])


def test_embedding_rejects_out_of_range_ids() -> None:
    with pytest.raises(IndexError, match="out of range"):
        F.embedding(Tensor(np.zeros((3, 2))), np.array([0, 3]))


def test_cross_entropy_matches_manual_value() -> None:
    logits = rand(4, 6, seed=4)
    targets = np.array([0, 5, 2, 2])
    logp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    expected = -logp[np.arange(4), targets].mean()
    assert F.cross_entropy(Tensor(logits), targets).item() == pytest.approx(expected)


def test_cross_entropy_ignore_index_excludes_positions() -> None:
    logits = rand(2, 3, 5, seed=5)
    targets = np.array([[1, -1, 2], [-1, -1, 4]])
    full = F.cross_entropy(Tensor(logits), targets, ignore_index=-1).item()
    keep = targets != -1
    manual = F.cross_entropy(Tensor(logits[keep]), targets[keep]).item()
    assert full == pytest.approx(manual)


def test_cross_entropy_all_ignored_is_zero_with_zero_grad() -> None:
    logits = Tensor(rand(2, 4, seed=6), requires_grad=True)
    loss = F.cross_entropy(logits, np.array([-1, -1]), ignore_index=-1)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    np.testing.assert_allclose(logits.grad, 0.0)


def test_dropout_is_identity_in_eval_and_scales_in_train() -> None:
    x = Tensor(np.ones((200, 50)))
    rng = np.random.default_rng(0)
    assert F.dropout(x, 0.5, rng, training=False) is x
    y = F.dropout(x, 0.25, rng, training=True).data
    kept = y != 0
    np.testing.assert_allclose(y[kept], 1.0 / 0.75)
    assert 0.7 < kept.mean() < 0.8


def test_dropout_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="probability"):
        F.dropout(Tensor(np.ones(2)), 1.0, np.random.default_rng(0), training=True)


def test_masked_fill_sets_values_and_blocks_gradient() -> None:
    x = Tensor(np.arange(4.0), requires_grad=True)
    mask = np.array([True, False, True, False])
    y = F.masked_fill(x, mask, -5.0)
    np.testing.assert_allclose(y.data, [-5.0, 1.0, -5.0, 3.0])
    y.sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, [0.0, 1.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: F.softmax(x, axis=-1),
        lambda x: F.softmax(x, axis=0),
        lambda x: F.log_softmax(x, axis=-1),
        F.gelu,
        lambda x: F.masked_fill(x, np.tril(np.ones((3, 4), dtype=bool)), 0.0),
    ],
)
def test_functional_gradcheck(fn: object) -> None:
    gradcheck(fn, [rand(2, 3, 4, seed=7) * 2.0])  # type: ignore[arg-type]


def test_layer_norm_gradcheck() -> None:
    gradcheck(
        lambda x, w, b: F.layer_norm(x, w, b),
        [rand(2, 3, 5, seed=8) * 3.0, rand(5, seed=9), rand(5, seed=10)],
        atol=1e-5,
    )


def test_embedding_gradcheck_with_repeated_ids() -> None:
    ids = np.array([[1, 1, 3], [0, 1, 3]])
    gradcheck(lambda w: F.embedding(w, ids), [rand(4, 3, seed=11)])


def test_cross_entropy_gradcheck_with_ignore_index() -> None:
    targets = np.array([[2, 0, -100], [1, 4, 4]])
    gradcheck(lambda x: F.cross_entropy(x, targets), [rand(2, 3, 5, seed=12) * 3.0])


def test_dropout_gradcheck_with_fixed_mask() -> None:
    def fn(x: Tensor) -> Tensor:
        return F.dropout(x, 0.3, np.random.default_rng(42), training=True)

    gradcheck(fn, [rand(4, 5, seed=13)])
