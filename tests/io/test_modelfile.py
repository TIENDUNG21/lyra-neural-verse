from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lyra.io.modelfile import (
    ModelFileError,
    dequantize_int8,
    quantize_int8,
    read_model_file,
    write_model_file,
)
from lyra.nn import LyraLM, ModelConfig

CFG = ModelConfig(vocab_size=10, block_size=8, d_model=8, n_layers=2, n_heads=2, dropout=0.0)
VOCAB = ["<pad>", "<unk>", "<l6>", "<l8>", "ta", "là", "nhau", "dâu", "đau", "lòng"]


@pytest.mark.parametrize("axis", [0, 1])
def test_quantization_error_is_at_most_half_a_step(axis: int) -> None:
    w = np.random.default_rng(0).normal(size=(6, 5)).astype(np.float32)
    w[2] = 0.0
    q, scales = quantize_int8(w, axis)
    assert q.dtype == np.int8
    assert scales.shape == (w.shape[axis],)
    back = dequantize_int8(q, scales, axis)
    step = np.expand_dims(scales, 1 - axis)
    assert np.all(np.abs(back - w) <= step / 2 + 1e-7)
    assert np.abs(q).max() == 127


def test_quantize_rejects_vectors() -> None:
    with pytest.raises(ValueError, match="matrices"):
        quantize_int8(np.ones(3), 0)


def test_round_trip_preserves_config_vocab_metadata_and_weights(tmp_path: Path) -> None:
    model = LyraLM(CFG)
    path = tmp_path / "m.bin"
    size = write_model_file(
        path, params=model.state_dict(), config=CFG, vocab=VOCAB, metadata={"steps": 3}
    )
    assert size == path.stat().st_size
    assert size % 16 == 0
    loaded = read_model_file(path)
    assert loaded.config == CFG
    assert loaded.vocab == VOCAB
    assert loaded.metadata == {"steps": 3}
    state = model.state_dict()
    assert set(loaded.weights) == set(state)
    for name, value in state.items():
        assert loaded.weights[name].dtype == np.float32
        tolerance = np.abs(value).max() / 127 if value.ndim == 2 else 0.0
        np.testing.assert_allclose(loaded.weights[name], value, atol=tolerance + 1e-7)


def test_vocab_size_mismatch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        write_model_file(tmp_path / "m.bin", params={}, config=CFG, vocab=VOCAB[:3])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: b"NOPE" + raw[4:],
        lambda raw: raw[:4] + (9).to_bytes(4, "little") + raw[8:],
        lambda raw: raw[:20],
        lambda raw: raw[: len(raw) // 2],
    ],
)
def test_corrupt_files_raise_model_file_error(tmp_path: Path, mutate: object) -> None:
    path = tmp_path / "m.bin"
    write_model_file(path, params=LyraLM(CFG).state_dict(), config=CFG, vocab=VOCAB)
    bad = tmp_path / "bad.bin"
    bad.write_bytes(mutate(path.read_bytes()))  # type: ignore[operator]
    with pytest.raises(ModelFileError):
        read_model_file(bad)
