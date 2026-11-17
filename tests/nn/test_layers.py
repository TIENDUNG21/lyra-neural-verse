from __future__ import annotations

import numpy as np
import pytest

from lyra.autograd import Tensor, gradcheck
from lyra.nn import Dropout, Embedding, LayerNorm, Linear, Module, ModuleList


class _Pair(Module):
    def __init__(self) -> None:
        super().__init__()
        rng = np.random.default_rng(0)
        self.first = Linear(3, 4, rng=rng)
        self.layers = ModuleList([Linear(4, 2, rng=rng), LayerNorm(2)])


def test_linear_forward_matches_affine_map() -> None:
    layer = Linear(3, 2, rng=np.random.default_rng(1))
    x = np.random.default_rng(2).normal(size=(5, 3)).astype(np.float32)
    assert layer.bias is not None
    expected = x @ layer.weight.data + layer.bias.data
    np.testing.assert_allclose(layer.forward(Tensor(x)).data, expected, rtol=1e-6)


def test_linear_without_bias_has_one_parameter() -> None:
    layer = Linear(3, 2, bias=False, rng=np.random.default_rng(1))
    assert [name for name, _ in layer.named_parameters()] == ["weight"]


def test_named_parameters_walk_nested_modules_in_order() -> None:
    names = [name for name, _ in _Pair().named_parameters()]
    assert names == [
        "first.weight",
        "first.bias",
        "layers.0.weight",
        "layers.0.bias",
        "layers.1.weight",
        "layers.1.bias",
    ]
    assert _Pair().num_parameters() == 3 * 4 + 4 + 4 * 2 + 2 + 2 + 2


def test_state_dict_round_trip_copies_values() -> None:
    source, target = _Pair(), _Pair()
    target.first.weight.data[:] = 0.0
    target.load_state_dict(source.state_dict())
    np.testing.assert_array_equal(target.first.weight.data, source.first.weight.data)
    state = source.state_dict()
    state["first.weight"][:] = 7.0
    assert not np.any(source.first.weight.data == 7.0)


def test_load_state_dict_reports_missing_unexpected_and_shape_errors() -> None:
    model = _Pair()
    state = model.state_dict()
    del state["first.bias"]
    with pytest.raises(KeyError, match=r"missing.*first\.bias"):
        model.load_state_dict(state)
    state = model.state_dict()
    state["extra"] = np.zeros(1)
    with pytest.raises(KeyError, match=r"unexpected.*extra"):
        model.load_state_dict(state)
    state = model.state_dict()
    state["first.weight"] = np.zeros((4, 3))
    with pytest.raises(ValueError, match="shape"):
        model.load_state_dict(state)


def test_train_eval_propagates_to_children() -> None:
    model = _Pair()
    model.eval()
    assert not model.layers[1].training
    model.train()
    assert model.layers[0].training


def test_to_dtype_casts_every_parameter() -> None:
    model = _Pair().to(np.float64)
    assert all(p.dtype == np.float64 for p in model.parameters())


def test_zero_grad_clears_gradients() -> None:
    layer = Linear(2, 2, rng=np.random.default_rng(0))
    layer.forward(Tensor(np.ones((1, 2)))).sum().backward()
    assert layer.weight.grad is not None
    layer.zero_grad()
    assert layer.weight.grad is None


def test_embedding_and_layer_norm_shapes() -> None:
    emb = Embedding(10, 4, rng=np.random.default_rng(0))
    out = emb.forward(np.array([[1, 2, 3]]))
    assert out.shape == (1, 3, 4)
    assert LayerNorm(4).forward(out).shape == (1, 3, 4)


def test_dropout_module_respects_training_flag() -> None:
    drop = Dropout(0.5, rng=np.random.default_rng(0))
    x = Tensor(np.ones((10, 10)))
    assert np.any(drop.forward(x).data == 0.0)
    drop.eval()
    np.testing.assert_array_equal(drop.forward(x).data, x.data)


def test_linear_gradcheck_with_respect_to_input() -> None:
    layer = Linear(3, 2, rng=np.random.default_rng(0)).to(np.float64)
    gradcheck(layer.forward, [np.random.default_rng(1).normal(size=(4, 3))])
