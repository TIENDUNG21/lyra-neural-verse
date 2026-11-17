from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lyra.decode.rng import Mulberry32
from lyra.decode.sampling import SamplingSettings, ranked_candidates, sample_token

# Reference values from the JavaScript mulberry32 (node, see web/test/rng.test.ts).
REFERENCE = {
    0: [0.26642920868471265, 0.0003297457005828619, 0.2232720274478197],
    1: [0.6270739405881613, 0.002735721180215478, 0.5274470399599522],
    42: [0.6011037519201636, 0.44829055899754167, 0.8524657934904099],
    4294967295: [0.8964226141106337, 0.189478256739676, 0.7156526781618595],
}


@pytest.mark.parametrize(("seed", "expected"), REFERENCE.items())
def test_mulberry32_matches_javascript(seed: int, expected: list[float]) -> None:
    rng = Mulberry32(seed)
    assert [rng.random() for _ in range(3)] == expected


def test_settings_validation() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingSettings(temperature=-1)
    with pytest.raises(ValueError, match="top_k"):
        SamplingSettings(top_k=-1)
    with pytest.raises(ValueError, match="top_p"):
        SamplingSettings(top_p=0.0)


LOGITS = np.array([1.0, 3.0, 2.0, 3.0, -1.0], dtype=np.float32)
ALL = np.ones(5, dtype=bool)


def test_ranking_breaks_ties_by_id() -> None:
    assert ranked_candidates(LOGITS, ALL) == [1, 3, 2, 0, 4]


def test_greedy_takes_the_best_allowed_token() -> None:
    greedy = SamplingSettings(temperature=0.0)
    assert sample_token(LOGITS, ALL, greedy, Mulberry32(0)) == 1
    mask = np.array([True, False, True, False, True])
    assert sample_token(LOGITS, mask, greedy, Mulberry32(0)) == 2


def test_empty_mask_raises() -> None:
    with pytest.raises(ValueError, match="no token"):
        sample_token(LOGITS, np.zeros(5, dtype=bool), SamplingSettings(), Mulberry32(0))


def test_top_k_limits_the_candidates() -> None:
    rng = Mulberry32(3)
    picks = {sample_token(LOGITS, ALL, SamplingSettings(1.0, 2, 1.0), rng) for _ in range(200)}
    assert picks == {1, 3}


def test_top_p_keeps_the_smallest_nucleus() -> None:
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    rng = Mulberry32(9)
    picks = Counter(
        sample_token(logits, np.ones(4, bool), SamplingSettings(1.0, 0, 0.75), rng)
        for _ in range(2000)
    )
    assert set(picks) == {0, 1}
    assert 0.55 < picks[0] / 2000 < 0.70  # 0.5 / 0.8 = 0.625


@settings(max_examples=100, deadline=None)
@given(
    logits=st.lists(st.floats(-20, 20), min_size=2, max_size=30),
    mask_seed=st.integers(0, 2**31),
    seed=st.integers(0, 2**32 - 1),
    temperature=st.floats(0.0, 2.0),
    top_k=st.integers(0, 10),
    top_p=st.floats(0.05, 1.0),
)
def test_sampled_token_is_always_allowed(
    logits: list[float], mask_seed: int, seed: int, temperature: float, top_k: int, top_p: float
) -> None:
    values = np.array(logits, dtype=np.float32)
    mask = np.random.default_rng(mask_seed).random(len(values)) < 0.5
    mask[mask_seed % len(values)] = True
    token = sample_token(
        values, mask, SamplingSettings(temperature, top_k, top_p), Mulberry32(seed)
    )
    assert mask[token]
