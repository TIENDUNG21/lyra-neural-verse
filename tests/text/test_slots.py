from __future__ import annotations

import numpy as np

from lyra.text.slots import N_SLOTS, UNKNOWN_SLOT, SlotState, line_slots, next_slot
from lyra.text.tokenizer import LINE6, LINE8


def test_slots_of_a_couplet() -> None:
    stream = [LINE6, *range(10, 16), LINE8, *range(20, 28), LINE6, 30]
    assert line_slots(stream).tolist() == [
        1, 2, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2,
    ]  # fmt: skip


def test_tokens_before_any_marker_or_past_the_line_end_are_unknown() -> None:
    assert line_slots([10, 11, LINE6]).tolist() == [UNKNOWN_SLOT, UNKNOWN_SLOT, 1]
    too_long = [LINE6, *range(10, 18)]
    assert line_slots(too_long).tolist()[-1] == UNKNOWN_SLOT


def test_incremental_state_matches_the_batch_function() -> None:
    stream = list(np.random.default_rng(0).choice([LINE6, LINE8, 9, 10, 11], size=200))
    state = SlotState()
    incremental = []
    for token in stream:
        slot, state = next_slot(state, int(token))
        incremental.append(slot)
    batch = line_slots(stream)
    assert incremental == batch.tolist()
    assert int(np.max(batch)) < N_SLOTS
