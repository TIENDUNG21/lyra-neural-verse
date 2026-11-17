from __future__ import annotations

import itertools

import pytest

from lyra.text.phonology import Syllable, parse_syllable
from lyra.text.rhyme import RhymeMatch, near_key, rhyme_match


def syl(word: str) -> Syllable:
    parsed = parse_syllable(word)
    assert parsed is not None, word
    return parsed


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("ta", "là", RhymeMatch.EXACT),
        ("hoa", "xa", RhymeMatch.EXACT),
        ("phong", "lòng", RhymeMatch.EXACT),
        ("nhau", "dâu", RhymeMatch.NEAR),
        ("phong", "hồng", RhymeMatch.NEAR),
        ("vàng", "vương", RhymeMatch.NEAR),
        ("bay", "tây", RhymeMatch.NEAR),
        ("xanh", "minh", RhymeMatch.NEAR),
        ("lời", "hồi", RhymeMatch.NEAR),
        ("hai", "trời", RhymeMatch.NEAR),
        ("giờ", "thưa", RhymeMatch.NEAR),
        ("lên", "tiền", RhymeMatch.NEAR),
        ("ta", "nhau", RhymeMatch.NONE),
        ("tan", "hoàng", RhymeMatch.NONE),
        ("xanh", "vàng", RhymeMatch.NONE),
        ("đèn", "đơn", RhymeMatch.NONE),
        ("hồng", "vương", RhymeMatch.NONE),
    ],
)
def test_rhyme_match(a: str, b: str, expected: RhymeMatch) -> None:
    assert rhyme_match(syl(a), syl(b)) is expected
    assert rhyme_match(syl(b), syl(a)) is expected


def test_match_ok_flag() -> None:
    assert RhymeMatch.EXACT.ok
    assert RhymeMatch.NEAR.ok
    assert not RhymeMatch.NONE.ok


WORDS = [
    "ta", "là", "nhau", "dâu", "đau", "lòng", "phong", "hồng", "ghen", "đèn",
    "xanh", "minh", "vàng", "vương", "trung", "gia", "nga", "vân", "thần", "mười",
    "vời", "trang", "nang", "da", "mà", "hơn", "sơn", "thành", "hai", "trời",
]  # fmt: skip


def test_near_rhyme_is_an_equivalence_relation() -> None:
    parsed = [syl(w) for w in WORDS]
    for a, b in itertools.product(parsed, repeat=2):
        same_key = near_key(a) == near_key(b)
        assert rhyme_match(a, b).ok == same_key, (a.text, b.text)
