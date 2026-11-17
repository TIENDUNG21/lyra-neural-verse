"""Train/validation split, token streams, batches and evaluation windows."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lyra.text.tokenizer import LINE6, LINE8, Tokenizer

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Split:
    train: list[list[str]]  # blocks of lines, each block starts with a 6-line
    val: list[list[str]]

    @property
    def train_lines(self) -> int:
        return sum(len(b) for b in self.train)

    @property
    def val_lines(self) -> int:
        return sum(len(b) for b in self.val)


def split_blocks(lines: Sequence[str], block_lines: int = 40, val_every: int = 10) -> Split:
    """Cut the poem into blocks of ``block_lines`` lines; every ``val_every``-th
    block (the 10th, 20th, ...) goes to validation.

    Blocks keep couplets intact (``block_lines`` must be even), and spreading
    the validation blocks over the whole poem makes them representative of it.
    """
    if block_lines <= 0 or block_lines % 2:
        raise ValueError("block_lines must be a positive even number")
    if val_every < 2:
        raise ValueError("val_every must be at least 2")
    blocks = [list(lines[i : i + block_lines]) for i in range(0, len(lines), block_lines)]
    train = [b for i, b in enumerate(blocks) if i % val_every != val_every - 1]
    val = [b for i, b in enumerate(blocks) if i % val_every == val_every - 1]
    return Split(train=train, val=val)


def encode_blocks(blocks: Sequence[Sequence[str]], tokenizer: Tokenizer) -> IntArray:
    """Concatenate the encodings of the blocks (each restarts the 6/8 alternation)."""
    ids: list[int] = []
    for block in blocks:
        ids.extend(tokenizer.encode_poem(block))
    return np.asarray(ids, dtype=np.int64)


class BatchSampler:
    """Random contiguous windows: ``x = s[i : i+T]``, ``y = s[i+1 : i+T+1]``."""

    def __init__(
        self, tokens: IntArray, block_size: int, batch_size: int, rng: np.random.Generator
    ) -> None:
        if len(tokens) < block_size + 1:
            raise ValueError("token stream is shorter than one training window")
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = rng

    def sample(self) -> tuple[IntArray, IntArray]:
        high = len(self.tokens) - self.block_size
        starts = self.rng.integers(0, high, size=self.batch_size)
        offsets = np.arange(self.block_size)
        x = self.tokens[starts[:, None] + offsets]
        y = self.tokens[starts[:, None] + offsets + 1]
        return x, y


def eval_windows(
    tokens: IntArray, block_size: int, *, skip_markers: bool = False
) -> Iterator[tuple[IntArray, IntArray]]:
    """Windows that score every token of ``tokens[1:]`` exactly once.

    Windows advance by half a block; after the first one only the second half
    of each window is scored, so every scored token sees at least ``T/2``
    tokens of context. Unscored targets are ``-1``. With ``skip_markers``
    the line-marker targets are not scored either (syllable-only metrics).
    """
    stride = max(1, block_size // 2)
    n = len(tokens)
    start = 0
    scored_until = 0  # index (in tokens) of the last target already scored
    while scored_until < n - 1:
        end = min(start + block_size, n - 1)
        x = tokens[start:end]
        y = tokens[start + 1 : end + 1].copy()
        first_new = scored_until - start  # targets y[k] predict tokens[start + 1 + k]
        if first_new > 0:
            y[:first_new] = -1
        if skip_markers:
            y[(y == LINE6) | (y == LINE8)] = -1
        yield x[None, :], y[None, :]
        scored_until = end
        start += stride


def count_scored(windows: Sequence[tuple[NDArray[Any], NDArray[Any]]]) -> int:
    return int(sum(int((y >= 0).sum()) for _, y in windows))
