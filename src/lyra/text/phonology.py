"""Vietnamese syllable phonology from orthography.

A written syllable is decomposed into

    onset + medial + nucleus + coda + tone

e.g. ``thuyền`` = ``th`` + ``u`` + ``iê`` + ``n`` + huyền. Only the parts that
matter for lục bát are modelled: the tone (and its bằng/trắc class) and the
rhyme, which is the nucleus plus the coda (onset and medial do not take part
in rhyming: ``hoa`` rhymes with ``ta``).

The parser works on the written form and returns ``None`` for strings that
are not Vietnamese syllables, so callers can treat them as unknown.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from lyra.text.normalize import normalize_text


class Tone(Enum):
    """The six tones of standard Vietnamese, by their traditional names."""

    NGANG = 0  # level, unmarked
    HUYEN = 1  # falling, grave accent
    SAC = 2  # rising, acute accent
    HOI = 3  # dipping, hook above
    NGA = 4  # broken, tilde
    NANG = 5  # heavy, dot below

    @property
    def is_bang(self) -> bool:
        """Bằng (flat) tones: ngang and huyền. All others are trắc (sharp)."""
        return self in (Tone.NGANG, Tone.HUYEN)

    @property
    def mark(self) -> str:
        return _TONE_TO_MARK.get(self, "")


_MARK_TO_TONE = {
    "̀": Tone.HUYEN,
    "́": Tone.SAC,
    "̉": Tone.HOI,
    "̃": Tone.NGA,
    "̣": Tone.NANG,
}
_TONE_TO_MARK = {tone: mark for mark, tone in _MARK_TO_TONE.items()}

_ONSETS = (
    "ngh",
    "ng",
    "nh",
    "ch",
    "tr",
    "th",
    "ph",
    "kh",
    "gh",
    "gi",
    "qu",
    "b",
    "c",
    "d",
    "đ",
    "g",
    "h",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "x",
)
_CONSONANT_CODAS = ("ng", "nh", "ch", "c", "m", "n", "p", "t")
_VOWELS = frozenset("aăâeêioôơuưy")

# Vowel spellings (after removing onset, medial and coda) -> nucleus phoneme.
_NUCLEI = {
    "a": "a",
    "ă": "ă",
    "â": "â",
    "e": "e",
    "ê": "ê",
    "i": "i",
    "y": "i",
    "o": "o",
    "ô": "ô",
    "ơ": "ơ",
    "u": "u",
    "ư": "ư",
    "iê": "iê",
    "yê": "iê",
    "ia": "iê",
    "ya": "iê",
    "uô": "uô",
    "ua": "uô",
    "ươ": "ươ",
    "ưa": "ươ",
    "oo": "oo",
    "ôô": "ôô",
}
# Medial (glide) spellings: o before a/ă/e; u before â/ê/y/ơ.
_MEDIAL_O = frozenset({"a", "ă", "e"})
_MEDIAL_U = frozenset({"â", "ê", "y", "ơ", "yê", "ya"})


@dataclass(frozen=True)
class Syllable:
    """A parsed syllable. ``text`` is the canonical lowercase NFC spelling."""

    text: str
    onset: str
    medial: str
    nucleus: str
    coda: str
    tone: Tone

    @property
    def rhyme(self) -> tuple[str, str]:
        """The rhyme as (nucleus, coda): what has to match for a rhyme."""
        return (self.nucleus, self.coda)

    @property
    def rhyme_text(self) -> str:
        return self.nucleus + self.coda

    @property
    def is_bang(self) -> bool:
        return self.tone.is_bang


def split_tone(text: str) -> tuple[str, Tone]:
    """Return the syllable without its tone mark (NFC) and the tone.

    Raises ``ValueError`` if the syllable carries two different tone marks.
    """
    decomposed = unicodedata.normalize("NFD", normalize_text(text))
    tones = {_MARK_TO_TONE[c] for c in decomposed if c in _MARK_TO_TONE}
    if len(tones) > 1:
        raise ValueError(f"syllable {text!r} has more than one tone mark")
    bare = "".join(c for c in decomposed if c not in _MARK_TO_TONE)
    return unicodedata.normalize("NFC", bare), (tones.pop() if tones else Tone.NGANG)


def _split_onset(bare: str) -> tuple[str, str]:
    if bare == "gi":  # "gì": onset gi, the vowel i is supplied by the caller
        return "gi", ""
    for onset in _ONSETS:
        if bare.startswith(onset) and len(bare) > len(onset):
            return onset, bare[len(onset) :]
    return "", bare


def _split_vowels(vowels: str, onset: str) -> tuple[str, str] | None:
    """Split a vowel spelling into (medial, nucleus spelling)."""
    if onset == "qu":
        return ("u", vowels) if vowels in _NUCLEI else None
    if vowels in ("ua", "uô") or vowels in _NUCLEI:
        return "", vowels
    if len(vowels) >= 2 and vowels[0] == "o" and vowels[1:] in _MEDIAL_O:
        return "o", vowels[1:]
    if len(vowels) >= 2 and vowels[0] == "u" and vowels[1:] in _MEDIAL_U:
        return "u", vowels[1:]
    return None


@lru_cache(maxsize=65536)
def parse_syllable(text: str) -> Syllable | None:
    """Parse one written syllable; ``None`` if it is not Vietnamese."""
    try:
        bare, tone = split_tone(text.lower())
    except ValueError:
        return None
    if not bare or any(not ("a" <= c <= "z" or c in _VOWELS or c == "đ") for c in bare):
        return None

    onset, rest = _split_onset(bare)
    if onset == "gi" and (rest.startswith("ê") or not any(c in _VOWELS for c in rest)):
        rest = "i" + rest  # "gìn" = gi + in, "giếng" = gi + iêng
    coda = ""
    for candidate in _CONSONANT_CODAS:
        if rest.endswith(candidate) and len(rest) > len(candidate):
            coda, rest = candidate, rest[: -len(candidate)]
            break
    if not rest or any(c not in _VOWELS for c in rest):
        return None

    # Semivowel codas: final i/y/o/u after another vowel ("ai", "ao", "uôi"...),
    # except "uy" where y is the nucleus.
    if not coda and len(rest) >= 2 and rest[-1] in "iyou" and rest not in ("uy", "oo"):
        semivowel = rest[-1]
        body = rest[:-1]
        coda = "i" if semivowel in "iy" else "u"
        if body == "a" and semivowel in "yu":
            body = "ă"  # "ay", "au" are written with a but have the short vowel ă
        elif body.endswith("a") and semivowel == "y":
            body = body[:-1] + "ă"  # "oay" -> o + ă + i
        rest = body

    split = _split_vowels(rest, onset)
    if split is None:
        return None
    medial, nucleus_spelling = split
    nucleus = _NUCLEI.get(nucleus_spelling)
    if nucleus is None:
        return None
    canonical = canonical_spelling(bare, tone)
    return Syllable(canonical, onset, medial, nucleus, coda, tone)


def _tone_vowel_index(bare: str) -> int:
    """Index of the vowel that carries the tone mark (traditional placement).

    Rules: a vowel with its own diacritic (ă â ê ô ơ ư) wins (the second one
    in ươ); otherwise, with a final consonant the last vowel takes it; in an
    open syllable with two vowels the first one does (``hòa``, ``thủy``,
    ``mùa``), with three the middle one (``ngoài``). The u of ``qu`` and the
    i of ``gi`` + vowel belong to the onset.
    """
    start = 2 if bare.startswith(("qu", "gi")) and len(bare) > 2 else 0
    positions = [i for i in range(start, len(bare)) if bare[i] in _VOWELS]
    if not positions and bare.startswith("gi"):
        positions = [1]
    if not positions:
        return -1
    marked = [i for i in positions if bare[i] in "ăâêôơư"]
    if marked:
        return marked[-1]
    # Vowels form one contiguous run; is anything consonantal after it?
    last = positions[-1]
    if last < len(bare) - 1 or len(positions) == 1:
        return last
    return positions[1] if len(positions) >= 3 else positions[0]


def canonical_spelling(bare: str, tone: Tone) -> str:
    """Place ``tone`` on the conventional vowel of the toneless spelling ``bare``."""
    if tone is Tone.NGANG:
        return bare
    index = _tone_vowel_index(bare)
    if index < 0:
        return bare
    decomposed = bare[: index + 1] + tone.mark + bare[index + 1 :]
    return unicodedata.normalize("NFC", decomposed)


def canonical_syllable(text: str) -> str:
    """Canonical lowercase spelling of ``text`` (tone mark position normalized).

    Non-Vietnamese strings are returned lowercased and NFC-normalized.
    """
    parsed = parse_syllable(text)
    if parsed is None:
        return unicodedata.normalize("NFC", normalize_text(text).lower())
    return parsed.text
