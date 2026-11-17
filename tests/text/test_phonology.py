from __future__ import annotations

import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lyra.text.phonology import (
    Tone,
    canonical_syllable,
    parse_syllable,
    split_tone,
)


@pytest.mark.parametrize(
    ("word", "tone"),
    [
        ("ta", Tone.NGANG),
        ("là", Tone.HUYEN),
        ("sắc", Tone.SAC),
        ("hỏi", Tone.HOI),
        ("ngã", Tone.NGA),
        ("nặng", Tone.NANG),
        ("ộ", Tone.NANG),
        ("ữ", Tone.NGA),
    ],
)
def test_tone_detection(word: str, tone: Tone) -> None:
    syllable = parse_syllable(word)
    assert syllable is not None
    assert syllable.tone is tone


def test_bang_and_trac_classes() -> None:
    assert {t for t in Tone if t.is_bang} == {Tone.NGANG, Tone.HUYEN}


def test_split_tone_returns_bare_syllable() -> None:
    assert split_tone("người") == ("ngươi", Tone.HUYEN)
    assert split_tone(unicodedata.normalize("NFD", "Đường")) == ("Đương", Tone.HUYEN)


def test_two_tone_marks_are_rejected() -> None:
    with pytest.raises(ValueError, match="more than one tone"):
        split_tone("đầụ")
    assert parse_syllable("đầụ") is None


@pytest.mark.parametrize(
    ("word", "onset", "medial", "nucleus", "coda"),
    [
        ("người", "ng", "", "ươ", "i"),
        ("nghiêng", "ngh", "", "iê", "ng"),
        ("giếng", "gi", "", "iê", "ng"),
        ("gì", "gi", "", "i", ""),
        ("gìn", "gi", "", "i", "n"),
        ("già", "gi", "", "a", ""),
        ("giữa", "gi", "", "ươ", ""),
        ("quốc", "qu", "u", "ô", "c"),
        ("qua", "qu", "u", "a", ""),
        ("khuya", "kh", "u", "iê", ""),
        ("khuỷu", "kh", "u", "i", "u"),
        ("thuở", "th", "u", "ơ", ""),
        ("xuân", "x", "u", "â", "n"),
        ("hoa", "h", "o", "a", ""),
        ("ngoài", "ng", "o", "a", "i"),
        ("xoáy", "x", "o", "ă", "i"),
        ("mùa", "m", "", "uô", ""),
        ("mưa", "m", "", "ươ", ""),
        ("kia", "k", "", "iê", ""),
        ("yêu", "", "", "iê", "u"),
        ("tuyết", "t", "u", "iê", "t"),
        ("máy", "m", "", "ă", "i"),
        ("cháu", "ch", "", "ă", "u"),
        ("cao", "c", "", "a", "u"),
        ("sâu", "s", "", "â", "u"),
        ("thủy", "th", "u", "i", ""),
        ("xoong", "x", "", "oo", "ng"),
        ("anh", "", "", "a", "nh"),
        ("đường", "đ", "", "ươ", "ng"),
    ],
)
def test_parse_table(word: str, onset: str, medial: str, nucleus: str, coda: str) -> None:
    syllable = parse_syllable(word)
    assert syllable is not None
    assert (syllable.onset, syllable.medial, syllable.nucleus, syllable.coda) == (
        onset,
        medial,
        nucleus,
        coda,
    )


@pytest.mark.parametrize("word", ["photo", "xyz", "", "123", "nh", "w", "đầụ"])
def test_non_vietnamese_strings_are_rejected(word: str) -> None:
    assert parse_syllable(word) is None


@pytest.mark.parametrize(
    ("variant", "canonical"),
    [
        ("hoà", "hòa"),
        ("thuỷ", "thủy"),
        ("hoè", "hòe"),
        ("Hòa", "hòa"),
        ("quý", "quý"),
        ("ngoài", "ngoài"),
        ("người", "người"),
        ("hoàng", "hoàng"),
    ],
)
def test_tone_mark_placement_is_canonicalized(variant: str, canonical: str) -> None:
    assert canonical_syllable(variant) == canonical
    assert canonical_syllable(unicodedata.normalize("NFD", variant)) == canonical


def test_canonical_syllable_passes_through_unknown_strings() -> None:
    assert canonical_syllable("ABC") == "abc"


@given(
    st.sampled_from(["a", "ă", "â", "e", "ê", "i", "o", "ô", "ơ", "u", "ư", "y"]),
    st.sampled_from(list(Tone)),
    st.sampled_from(["", "n", "ng", "m", "i", "u", "c", "t", "nh", "ch", "p"]),
    st.sampled_from(["", "b", "ch", "ngh", "gi", "qu", "th", "đ"]),
)
def test_canonical_spelling_is_idempotent(vowel: str, tone: Tone, coda: str, onset: str) -> None:
    word = unicodedata.normalize("NFC", onset + vowel + tone.mark + coda)
    once = canonical_syllable(word)
    assert canonical_syllable(once) == once
    parsed = parse_syllable(word)
    if parsed is not None:
        assert parsed.tone is tone
