from __future__ import annotations

import math

import numpy as np
import pytest

from lyra.autograd import Tensor, gradcheck
from lyra.nn import CausalSelfAttention, LyraLM, ModelConfig


def tiny_config(**overrides: object) -> ModelConfig:
    base: dict[str, object] = {
        "vocab_size": 11,
        "block_size": 8,
        "d_model": 16,
        "n_layers": 2,
        "n_heads": 4,
        "dropout": 0.0,
        "seed": 3,
    }
    base.update(overrides)
    return ModelConfig.from_dict(base)


def test_config_rejects_heads_that_do_not_divide_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        tiny_config(n_heads=3)


def test_config_round_trips_through_dict() -> None:
    cfg = tiny_config()
    assert ModelConfig.from_dict(cfg.to_dict()) == cfg


def test_config_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ModelConfig.from_dict({**tiny_config().to_dict(), "colour": 1})


def test_forward_shape_and_initial_loss_near_uniform() -> None:
    model = LyraLM(tiny_config())
    ids = np.random.default_rng(0).integers(0, 11, size=(3, 8))
    logits = model.forward(ids)
    assert logits.shape == (3, 8, 11)
    loss = model.loss(ids[:, :-1], ids[:, 1:])
    assert abs(loss.item() - math.log(11)) < 0.1


def test_forward_rejects_sequences_longer_than_block() -> None:
    with pytest.raises(ValueError, match="block_size"):
        LyraLM(tiny_config()).forward(np.zeros((1, 9), dtype=np.int64))


def test_embeddings_are_tied_with_the_output_head() -> None:
    model = LyraLM(tiny_config())
    names = [name for name, _ in model.named_parameters()]
    assert names.count("tok_emb.weight") == 1
    assert not any(name.startswith("head") and name != "head_bias" for name in names)
    cfg = tiny_config()
    d, v, t, layers = cfg.d_model, cfg.vocab_size, cfg.block_size, cfg.n_layers
    per_block = (
        2 * 2 * d + (d * 3 * d + 3 * d) + (d * d + d) + (d * 4 * d + 4 * d) + (4 * d * d + d)
    )
    assert model.num_parameters() == v * d + t * d + layers * per_block + 2 * d + v


def test_model_is_causal() -> None:
    model = LyraLM(tiny_config())
    model.eval()
    ids = np.random.default_rng(1).integers(0, 11, size=(1, 8))
    changed = ids.copy()
    changed[0, 5] = (changed[0, 5] + 1) % 11
    a, b = model.forward(ids).data, model.forward(changed).data
    np.testing.assert_allclose(a[0, :5], b[0, :5], atol=1e-6)
    assert not np.allclose(a[0, 5:], b[0, 5:])


def test_attention_gradcheck_float64() -> None:
    attn = CausalSelfAttention(
        d_model=8, n_heads=2, block_size=4, dropout=0.0, rng=np.random.default_rng(0)
    ).to(np.float64)
    gradcheck(attn.forward, [np.random.default_rng(1).normal(size=(2, 4, 8))], atol=1e-5)


def test_gradients_reach_every_parameter() -> None:
    model = LyraLM(tiny_config(dropout=0.1))
    ids = np.random.default_rng(2).integers(0, 11, size=(2, 8))
    model.loss(ids[:, :-1], ids[:, 1:]).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert p.grad.shape == p.shape, name
        assert np.all(np.isfinite(p.grad)), name


def test_eval_mode_is_deterministic_and_train_mode_uses_dropout() -> None:
    model = LyraLM(tiny_config(dropout=0.5))
    ids = np.arange(8).reshape(1, 8)
    model.eval()
    np.testing.assert_array_equal(model.forward(ids).data, model.forward(ids).data)
    model.train()
    assert not np.array_equal(model.forward(ids).data, model.forward(ids).data)


def test_logits_last_only_matches_full_forward() -> None:
    model = LyraLM(tiny_config())
    model.eval()
    ids = np.random.default_rng(3).integers(0, 11, size=(2, 6))
    full = model.forward(ids).data
    assert isinstance(model.forward(ids, last_only=True), Tensor)
    np.testing.assert_allclose(model.forward(ids, last_only=True).data, full[:, -1:], atol=1e-6)
