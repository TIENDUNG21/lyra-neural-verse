from __future__ import annotations

import numpy as np
import pytest

from lyra.nn import LyraLM, ModelConfig
from lyra.nn.inference import InferenceEngine
from lyra.text.features import token_features
from lyra.text.phonology import Tone

VOCAB = ["<pad>", "<unk>", "<l6>", "<l8>", "ta", "là", "hoa", "nhau", "dâu", "photo", "người"]
FEATURES = token_features(VOCAB)
CFG = ModelConfig(
    vocab_size=len(VOCAB),
    block_size=8,
    d_model=8,
    n_layers=1,
    n_heads=2,
    dropout=0.0,
    phonology_embeddings=True,
)


def test_token_features_share_ids_by_sound() -> None:
    ta, la, hoa, photo = 4, 5, 6, 9
    assert FEATURES.rhyme[ta] == FEATURES.rhyme[la] == FEATURES.rhyme[hoa] != 0
    assert FEATURES.tone[ta] == 1 + Tone.NGANG.value
    assert FEATURES.tone[la] == 1 + Tone.HUYEN.value
    assert FEATURES.near[7] == FEATURES.near[8]  # nhau ~ dâu
    assert FEATURES.rhyme[7] != FEATURES.rhyme[8]
    assert not FEATURES.is_syllable[:4].any()
    assert not FEATURES.is_syllable[photo]
    assert FEATURES.onset[photo] == FEATURES.rhyme[photo] == FEATURES.tone[photo] == 0


def test_factored_model_requires_features() -> None:
    with pytest.raises(ValueError, match="features"):
        LyraLM(CFG)


def test_factored_parameters_get_gradients() -> None:
    model = LyraLM(CFG, FEATURES)
    ids = np.array([[2, 4, 5, 6, 7, 8, 10, 3]])
    model.loss(ids[:, :-1], ids[:, 1:]).backward()
    for name in ("onset_emb.weight", "rhyme_emb.weight", "tone_emb.weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None
        assert np.abs(grad).sum() > 0


def test_folding_gives_an_equivalent_plain_model() -> None:
    model = LyraLM(CFG, FEATURES)
    model.eval()
    ids = np.array([[2, 4, 5, 6, 7, 8, 10]])
    plain_cfg, state = model.folded_state_dict()
    assert not plain_cfg.phonology_embeddings
    assert "rhyme_emb.weight" not in state
    plain = LyraLM(plain_cfg)
    plain.load_state_dict(state)
    plain.eval()
    np.testing.assert_allclose(plain.forward(ids).data, model.forward(ids).data, atol=1e-6)
    engine = InferenceEngine(plain_cfg, state)
    np.testing.assert_allclose(
        engine.feed(ids[0].tolist()), model.forward(ids).data[0, -1], atol=1e-5
    )
