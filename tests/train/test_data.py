from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lyra.text.tokenizer import LINE6, LINE8, UNK, Tokenizer
from lyra.train.corpus import load_corpus
from lyra.train.data import BatchSampler, encode_blocks, eval_windows, split_blocks

ROOT = Path(__file__).resolve().parents[2]
KIEU = load_corpus(ROOT / "data" / "truyen-kieu.txt")


def test_split_is_deterministic_disjoint_and_complete() -> None:
    split = split_blocks(KIEU, block_lines=40, val_every=10)
    assert split.train_lines + split.val_lines == len(KIEU)
    assert split.val[0] == KIEU[360:400]
    assert all(len(block) % 2 == 0 for block in split.train + split.val)
    flat_train = {tuple(b) for b in split.train}
    assert not flat_train & {tuple(b) for b in split.val}
    assert 0.08 < split.val_lines / len(KIEU) < 0.12


@pytest.mark.parametrize(("block_lines", "val_every"), [(0, 10), (7, 10), (40, 1)])
def test_split_rejects_bad_parameters(block_lines: int, val_every: int) -> None:
    with pytest.raises(ValueError, match=r"block_lines|val_every"):
        split_blocks(KIEU, block_lines, val_every)


def test_encode_blocks_restarts_markers_per_block() -> None:
    tok = Tokenizer.build(KIEU)
    ids = encode_blocks([KIEU[0:2], KIEU[2:4]], tok)
    assert ids.dtype == np.int64
    assert [i for i, t in enumerate(ids) if t in (LINE6, LINE8)] == [0, 7, 16, 23]
    assert ids[16] == LINE6


def test_batch_sampler_targets_are_shifted_inputs() -> None:
    tokens = np.arange(100, dtype=np.int64)
    sampler = BatchSampler(tokens, block_size=8, batch_size=4, rng=np.random.default_rng(0))
    x, y = sampler.sample()
    assert x.shape == y.shape == (4, 8)
    np.testing.assert_array_equal(y, x + 1)
    assert int(np.max(y)) <= 99


def test_token_dropout_replaces_inputs_but_not_targets_or_markers() -> None:
    tokens = np.tile(np.array([LINE6, 10, 11, 12, 13, 14, 15], dtype=np.int64), 50)
    sampler = BatchSampler(tokens, 16, 64, np.random.default_rng(0), token_dropout=0.3)
    x, y = sampler.sample()
    clean = BatchSampler(tokens, 16, 64, np.random.default_rng(0)).sample()[0]
    dropped = x != clean
    assert np.all(x[dropped] == UNK)
    assert not np.any(dropped & (clean == LINE6))
    assert 0.2 < dropped[clean != LINE6].mean() < 0.4
    assert not np.any(y == UNK)
    with pytest.raises(ValueError, match="token_dropout"):
        BatchSampler(tokens, 16, 4, np.random.default_rng(0), token_dropout=1.0)


def test_batch_sampler_rejects_short_streams() -> None:
    with pytest.raises(ValueError, match="shorter"):
        BatchSampler(np.arange(5), 8, 2, np.random.default_rng(0))


@pytest.mark.parametrize(("n", "block"), [(2, 8), (9, 8), (50, 8), (51, 7), (200, 16)])
def test_eval_windows_score_each_target_exactly_once(n: int, block: int) -> None:
    tokens = np.arange(10, 10 + n, dtype=np.int64)
    scored: list[int] = []
    for x, y in eval_windows(tokens, block):
        assert x.shape == y.shape
        assert x.shape[1] <= block
        scored.extend(int(t) for t in y[y >= 0])
    assert scored == list(range(11, 10 + n))


def test_eval_windows_can_skip_marker_targets() -> None:
    tokens = np.array([LINE6, 5, 6, 7, LINE8, 8, 9], dtype=np.int64)
    scored = [int(t) for _, y in eval_windows(tokens, 4, skip_markers=True) for t in y[y >= 0]]
    assert scored == [5, 6, 7, 8, 9]
