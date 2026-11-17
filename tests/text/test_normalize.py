from __future__ import annotations

import unicodedata

from hypothesis import given
from hypothesis import strategies as st

from lyra.text.normalize import normalize_text, split_syllables

LINE = "Trăm năm trong cõi người ta,"


def test_split_drops_punctuation_and_lowercases() -> None:
    assert split_syllables(LINE) == ["trăm", "năm", "trong", "cõi", "người", "ta"]
    assert split_syllables("Rằng: «Sao trong tiết Thanh minh?»") == [
        "rằng",
        "sao",
        "trong",
        "tiết",
        "thanh",
        "minh",
    ]


def test_nfd_input_gives_identical_syllables() -> None:
    decomposed = unicodedata.normalize("NFD", LINE)
    assert decomposed != LINE
    assert split_syllables(decomposed) == split_syllables(LINE)


def test_zero_width_characters_do_not_split_syllables() -> None:
    assert split_syllables("ng\u200bư\u200dời t\ufeffa") == ["người", "ta"]


def test_icelandic_eth_is_read_as_d_with_stroke() -> None:
    assert normalize_text("\u00d0ường") == "Đường"
    assert split_syllables("\u00f0ường") == ["đường"]


def test_emoji_and_astral_characters_do_not_crash() -> None:
    assert split_syllables("ta \U0001f600 \U00020000 là") == ["ta", "\U00020000", "là"]
    assert split_syllables("") == []
    assert split_syllables(" \t\n ") == []


@given(st.text())
def test_split_is_stable_under_nfc_nfd(text: str) -> None:
    nfd = unicodedata.normalize("NFD", text)
    assert split_syllables(nfd) == split_syllables(unicodedata.normalize("NFC", text))
