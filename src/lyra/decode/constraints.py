"""The lục bát constraint state machine used during decoding.

At every step it answers "which tokens may come next?" as a boolean mask over
the vocabulary:

- at a line start, only the line marker (``<l6>`` or ``<l8>``) is allowed;
- inside a line, only Vietnamese syllables (no specials, no ``<unk>``);
- at the even positions, only the required tone class (bằng/trắc);
- at position 8 of the 8-line, only the tone opposite to position 6 (ngang
  vs huyền);
- at the rhyme positions, only syllables that rhyme (exactly, or nearly
  unless ``strict_rhyme``) with the syllable they answer.

If the rhyme mask leaves nothing (possible for a rare rhyme typed by the
user), the rhyme requirement is dropped for that step and the step is
flagged as relaxed. Tone masks can never be empty for a real vocabulary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from lyra.text.features import TokenFeatures
from lyra.text.lucbat import TONE_PATTERN
from lyra.text.phonology import Syllable, Tone, parse_syllable
from lyra.text.rhyme import exact_key, near_key
from lyra.text.tokenizer import LINE6, LINE8, UNK, line_marker

BoolArray = NDArray[np.bool_]

# Tone feature ids are 1 + Tone.value (0 means "not a syllable").
_NGANG = 1 + Tone.NGANG.value
_HUYEN = 1 + Tone.HUYEN.value


@dataclass(frozen=True)
class DecodeState:
    """Where the decoder is in the poem. Immutable; ``advance`` returns a new one."""

    lines_done: int = 0
    in_line: bool = False
    position: int = 0  # syllables already emitted in the current line
    sixth_of_six: Syllable | None = None  # answered by the 8-line's 6th syllable
    end_of_eight: Syllable | None = None  # answered by the next 6-line's 6th syllable
    sixth_of_eight: Syllable | None = None  # its tone is contrasted at position 8

    @property
    def line_length(self) -> int:
        return 6 if self.lines_done % 2 == 0 else 8


@dataclass(frozen=True)
class Step:
    mask: BoolArray
    forced: int | None  # the only allowed token, if any
    relaxed: bool  # True if the rhyme constraint had to be dropped


class LucBatConstraints:
    def __init__(
        self, features: TokenFeatures, vocab: Sequence[str], *, strict_rhyme: bool = False
    ) -> None:
        if len(vocab) != features.vocab_size:
            raise ValueError("features and vocabulary disagree in size")
        self.features = features
        self.vocab = list(vocab)
        self.strict_rhyme = strict_rhyme
        size = features.vocab_size
        self.syllables = features.is_syllable.copy()
        self.syllables[UNK] = False
        bang = (features.tone == _NGANG) | (features.tone == _HUYEN)
        self.bang = self.syllables & bang
        self.trac = self.syllables & ~bang
        self.ngang = self.syllables & (features.tone == _NGANG)
        self.huyen = self.syllables & (features.tone == _HUYEN)
        self._exact_ids = {name: i for i, name in enumerate(features.rhyme_names)}
        self._near_ids = {name: i for i, name in enumerate(features.near_names)}
        self._markers = {
            LINE6: np.arange(size) == LINE6,
            LINE8: np.arange(size) == LINE8,
        }

    def rhyme_mask(self, source: Syllable) -> BoolArray:
        """Tokens that rhyme with ``source`` (exactly, or nearly if allowed)."""
        if self.strict_rhyme:
            key_id = self._exact_ids.get(exact_key(source), -1)
            return np.asarray(self.features.rhyme == key_id)
        key_id = self._near_ids.get(near_key(source), -1)
        return np.asarray(self.features.near == key_id)

    def step(self, state: DecodeState) -> Step:
        if not state.in_line:
            marker = line_marker(state.lines_done)
            return Step(self._markers[marker], marker, False)
        length = state.line_length
        position = state.position + 1
        mask = self.syllables.copy()
        need_bang = TONE_PATTERN[length].get(position)
        if need_bang is True:
            mask &= self.bang
        elif need_bang is False:
            mask &= self.trac
        if length == 8 and position == 8 and state.sixth_of_eight is not None:
            sixth = state.sixth_of_eight.tone
            if sixth is Tone.HUYEN:
                mask &= self.ngang
            elif sixth is Tone.NGANG:
                mask &= self.huyen
        relaxed = False
        if position == 6:
            source = state.sixth_of_six if length == 8 else state.end_of_eight
            if source is not None:
                rhymed = mask & self.rhyme_mask(source)
                if rhymed.any():
                    mask = rhymed
                else:
                    relaxed = True
        return Step(mask, None, relaxed)

    def advance(self, state: DecodeState, token: int, text: str | None = None) -> DecodeState:
        """State after emitting ``token``. ``text`` overrides the token's spelling
        (used for prompt syllables that are not in the vocabulary)."""
        if not state.in_line:
            if token not in (LINE6, LINE8):
                raise ValueError("a line must start with its marker token")
            return replace(state, in_line=True, position=0)
        if token in (LINE6, LINE8):
            raise ValueError("marker token inside a line")
        position = state.position + 1
        syllable = parse_syllable(text if text is not None else self.vocab[token])
        length = state.line_length
        new = replace(state, position=position)
        if position == 6 and length == 6:
            new = replace(new, sixth_of_six=syllable)
        elif position == 6 and length == 8:
            new = replace(new, sixth_of_eight=syllable)
        if position == length:
            new = replace(new, lines_done=state.lines_done + 1, in_line=False, position=0)
            if length == 8:
                new = replace(new, end_of_eight=syllable)
        return new
