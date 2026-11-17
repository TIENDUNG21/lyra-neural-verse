from __future__ import annotations

import numpy as np
import pytest

from lyra.nn import LyraLM, ModelConfig
from lyra.nn.inference import InferenceEngine

CFG = ModelConfig(vocab_size=13, block_size=8, d_model=16, n_layers=2, n_heads=4, dropout=0.1)


@pytest.fixture(scope="module")
def model() -> LyraLM:
    m = LyraLM(CFG)
    m.eval()
    return m


def reference(model: LyraLM, tokens: list[int]) -> np.ndarray:
    return np.asarray(model.forward(np.array([tokens])).data[0])


def test_prefill_matches_training_model(model: LyraLM) -> None:
    tokens = [2, 5, 7, 1, 12, 3]
    engine = InferenceEngine(CFG, model.state_dict())
    np.testing.assert_allclose(engine.feed(tokens), reference(model, tokens)[-1], atol=1e-5)


def test_incremental_steps_match_full_forward(model: LyraLM) -> None:
    tokens = [2, 9, 4, 4, 11, 3, 6, 8]
    engine = InferenceEngine(CFG, model.state_dict())
    full = reference(model, tokens)
    for i, token in enumerate(tokens):
        np.testing.assert_allclose(engine.feed([token]), full[i], atol=1e-5)
    assert engine.length == 8


def test_mixed_chunks_give_the_same_result(model: LyraLM) -> None:
    tokens = [2, 9, 4, 4, 11, 3, 6]
    a = InferenceEngine(CFG, model.state_dict())
    b = InferenceEngine(CFG, model.state_dict())
    a.feed(tokens[:3])
    a.feed(tokens[3:5])
    np.testing.assert_allclose(a.feed(tokens[5:]), b.feed(tokens), atol=1e-5)


def test_overflow_reprefills_the_most_recent_half(model: LyraLM) -> None:
    tokens = list(np.random.default_rng(0).integers(0, 13, size=CFG.block_size + 3))
    engine = InferenceEngine(CFG, model.state_dict())
    for token in tokens:
        logits = engine.feed([int(token)])
    window = [int(t) for t in tokens[-(CFG.block_size // 2 + 3) :]]
    np.testing.assert_allclose(logits, reference(model, window)[-1], atol=1e-5)
    assert engine.length == len(window)


def test_long_feed_in_one_call_is_handled(model: LyraLM) -> None:
    engine = InferenceEngine(CFG, model.state_dict())
    logits = engine.feed([4] * 30)
    assert logits.shape == (13,)
    assert engine.length <= CFG.block_size


def test_reset_and_invalid_input(model: LyraLM) -> None:
    engine = InferenceEngine(CFG, model.state_dict())
    engine.feed([2, 3])
    engine.reset()
    assert engine.length == 0
    with pytest.raises(ValueError, match="at least one"):
        engine.feed([])
    with pytest.raises(ValueError, match="out of range"):
        engine.feed([13])


def test_logits_all_matches_every_position(model: LyraLM) -> None:
    tokens = [2, 9, 4, 4, 11, 3]
    engine = InferenceEngine(CFG, model.state_dict())
    np.testing.assert_allclose(engine.logits_all(tokens), reference(model, tokens), atol=1e-5)
    with pytest.raises(ValueError, match="block_size"):
        engine.logits_all([1] * 9)
