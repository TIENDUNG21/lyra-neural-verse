"""Line slots: where each token sits inside the lục bát structure.

A slot combines the line type and the position in the line:

- slot 0: unknown (before the first marker, or past the end of a line);
- slots 1..7: a 6-line, position 0 (the ``<l6>`` marker) to position 6;
- slots 8..16: an 8-line, position 0 (the ``<l8>`` marker) to position 8.

Slots are a deterministic function of the token stream, so the model gets
them for free: they become a learned embedding added to the input (like the
format and intra-line position embeddings of SongNet, Li et al. 2020). With
them the model knows when a line ends and which positions carry the tone and
rhyme rules, instead of having to count syllables.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lyra.text.tokenizer import LINE6, LINE8

N_SLOTS = 17
UNKNOWN_SLOT = 0


@dataclass(frozen=True)
class SlotState:
    line_length: int = 0  # 0 = no line seen yet
    position: int = 0


def next_slot(state: SlotState, token: int) -> tuple[int, SlotState]:
    """Slot of ``token`` and the state after it."""
    if token == LINE6:
        return 1, SlotState(6, 0)
    if token == LINE8:
        return 8, SlotState(8, 0)
    if state.line_length == 0:
        return UNKNOWN_SLOT, state
    position = state.position + 1
    new = SlotState(state.line_length, position)
    if position > state.line_length:
        return UNKNOWN_SLOT, new
    base = 1 if state.line_length == 6 else 8
    return base + position, new


def line_slots(tokens: Iterable[int], state: SlotState | None = None) -> NDArray[np.int64]:
    """Slots of a whole token stream."""
    current = state or SlotState()
    out = []
    for token in tokens:
        slot, current = next_slot(current, int(token))
        out.append(slot)
    return np.asarray(out, dtype=np.int64)
