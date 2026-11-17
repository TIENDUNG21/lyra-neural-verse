"""Per-token phonological features over a vocabulary.

For every token id: its onset, rhyme (nucleus + coda), tone, near-rhyme
class and whether it is a Vietnamese syllable at all. These feed

- the factored embeddings of the model (a syllable's vector is the sum of its
  own vector and those of its onset, rhyme and tone), and
- the constrained decoder (tone classes and rhyme classes as masks).

Index 0 of every feature table means "none" (special tokens, unparseable
strings).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lyra.text.phonology import parse_syllable
from lyra.text.rhyme import exact_key, near_key

NONE = "-"


@dataclass(frozen=True)
class TokenFeatures:
    onset: NDArray[np.int64]
    rhyme: NDArray[np.int64]
    near: NDArray[np.int64]
    tone: NDArray[np.int64]  # 0 = none, 1 + Tone.value otherwise
    is_syllable: NDArray[np.bool_]
    onset_names: tuple[str, ...]
    rhyme_names: tuple[str, ...]
    near_names: tuple[str, ...]

    @property
    def vocab_size(self) -> int:
        return len(self.onset)

    @property
    def n_onsets(self) -> int:
        return len(self.onset_names)

    @property
    def n_rhymes(self) -> int:
        return len(self.rhyme_names)

    @property
    def n_tones(self) -> int:
        return 7


def _index(names: list[str], key: str) -> int:
    if key not in names:
        names.append(key)
    return names.index(key)


def token_features(vocab: Sequence[str]) -> TokenFeatures:
    """Features of each vocabulary entry; deterministic in the vocabulary order."""
    onset_names, rhyme_names, near_names = [NONE], [NONE], [NONE]
    onset, rhyme, near, tone, is_syllable = [], [], [], [], []
    for token in vocab:
        syllable = None if token.startswith("<") else parse_syllable(token)
        if syllable is None:
            onset.append(0)
            rhyme.append(0)
            near.append(0)
            tone.append(0)
            is_syllable.append(False)
            continue
        onset.append(_index(onset_names, syllable.onset or "∅"))
        rhyme.append(_index(rhyme_names, exact_key(syllable)))
        near.append(_index(near_names, near_key(syllable)))
        tone.append(1 + syllable.tone.value)
        is_syllable.append(True)
    return TokenFeatures(
        onset=np.asarray(onset, dtype=np.int64),
        rhyme=np.asarray(rhyme, dtype=np.int64),
        near=np.asarray(near, dtype=np.int64),
        tone=np.asarray(tone, dtype=np.int64),
        is_syllable=np.asarray(is_syllable, dtype=bool),
        onset_names=tuple(onset_names),
        rhyme_names=tuple(rhyme_names),
        near_names=tuple(near_names),
    )
