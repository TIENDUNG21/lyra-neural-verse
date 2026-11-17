"""Unicode normalization and syllable splitting for Vietnamese text.

Vietnamese can arrive precomposed (NFC) or decomposed (NFD), may contain
invisible characters copied from the web, and sometimes uses the Icelandic
eth (U+00D0/U+00F0) instead of the Vietnamese D with stroke (U+0110/U+0111).
Everything downstream sees one canonical form.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width and other invisible format characters that should never split or
# join syllables.
_INVISIBLE = dict.fromkeys(
    map(ord, "​‌‍⁠﻿­᠎"),
    None,
)
_LOOKALIKES = str.maketrans({"Ð": "Đ", "ð": "đ"})
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC-normalize, drop invisible characters and fix D-with-stroke lookalikes."""
    cleaned = text.translate(_INVISIBLE).translate(_LOOKALIKES)
    return unicodedata.normalize("NFC", cleaned)


def is_word_char(char: str) -> bool:
    """Letters and combining marks belong to syllables; everything else separates."""
    category = unicodedata.category(char)
    return category[0] == "L" or category[0] == "M"


def split_syllables(line: str) -> list[str]:
    """Split a line into lowercase syllables, dropping punctuation and symbols.

    >>> split_syllables("Trăm năm trong cõi người ta,")
    ['trăm', 'năm', 'trong', 'cõi', 'người', 'ta']
    """
    text = unicodedata.normalize("NFC", normalize_text(line).lower())
    separated = "".join(c if is_word_char(c) else " " for c in text)
    return [s for s in _SPACES.split(separated) if s]
