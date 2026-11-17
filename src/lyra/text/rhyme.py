"""Rhyme (vần) matching for lục bát.

Two syllables rhyme when their rhymes (nucleus + coda) agree; onset and medial
do not matter (``hoa`` rhymes with ``ta``). Lục bát accepts two strengths:

- EXACT (vần chính): same nucleus and same coda — ``ta / là``.
- NEAR (vần thông): same coda and a nucleus from the same vowel class —
  ``nhau / dâu`` (ă ~ â), ``phong / hồng`` (o ~ ô), ``vàng / vương`` (a ~ ươ).

Vowel classes follow the place/rounding of the nucleus:

- FRONT: i, ê, e, iê (and the a of ``anh``/``ach``, which is pronounced front)
- ROUNDED (back rounded): u, ô, o, uô, oo
- CENTRAL (non-front unrounded): a, ă, â, ơ, ư, ươ

Before the glide coda ``-i`` (``ai, ay, ây, oi, ôi, ơi, ui, ưi, uôi, ươi``)
every nucleus rhymes: front vowels cannot precede ``-i``, and Truyện Kiều freely
pairs rounded and unrounded vowels there (``lời / hồi``, ``hai / trời``).

Near-rhyme is an equivalence relation, so each syllable has a ``near_key``
and two syllables near-rhyme exactly when their keys are equal. The decoder
uses these keys to precompute masks.
"""

from __future__ import annotations

from enum import Enum

from lyra.text.phonology import Syllable


class RhymeMatch(Enum):
    NONE = 0
    NEAR = 1
    EXACT = 2

    @property
    def ok(self) -> bool:
        return self is not RhymeMatch.NONE


class VowelClass(Enum):
    FRONT = "front"
    ROUNDED = "rounded"
    CENTRAL = "central"


_VOWEL_CLASS = {
    "i": VowelClass.FRONT,
    "ê": VowelClass.FRONT,
    "e": VowelClass.FRONT,
    "iê": VowelClass.FRONT,
    "u": VowelClass.ROUNDED,
    "ô": VowelClass.ROUNDED,
    "o": VowelClass.ROUNDED,
    "uô": VowelClass.ROUNDED,
    "oo": VowelClass.ROUNDED,
    "ôô": VowelClass.ROUNDED,
    "a": VowelClass.CENTRAL,
    "ă": VowelClass.CENTRAL,
    "â": VowelClass.CENTRAL,
    "ơ": VowelClass.CENTRAL,
    "ư": VowelClass.CENTRAL,
    "ươ": VowelClass.CENTRAL,
}


def vowel_class(syllable: Syllable) -> VowelClass:
    if syllable.nucleus == "a" and syllable.coda in ("nh", "ch"):
        return VowelClass.FRONT
    return _VOWEL_CLASS[syllable.nucleus]


def exact_key(syllable: Syllable) -> str:
    """Key shared by exactly rhyming syllables, e.g. ``"a|"`` or ``"ươ|ng"``."""
    return f"{syllable.nucleus}|{syllable.coda}"


def near_key(syllable: Syllable) -> str:
    """Key shared by syllables that rhyme at least nearly, e.g. ``"central|ng"``."""
    if syllable.coda == "i":
        return "*|i"
    return f"{vowel_class(syllable).value}|{syllable.coda}"


def rhyme_match(a: Syllable, b: Syllable) -> RhymeMatch:
    """How well ``a`` and ``b`` rhyme."""
    if a.rhyme == b.rhyme:
        return RhymeMatch.EXACT
    if near_key(a) == near_key(b):
        return RhymeMatch.NEAR
    return RhymeMatch.NONE
