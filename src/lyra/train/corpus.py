"""The training corpus: Truyện Kiều as published on Vietnamese Wikisource.

``extract_poem_lines`` turns the page's wikitext into one verse per line. The
cleaned text keeps the original capitalization and punctuation (it is meant to
be readable); the tokenizer drops punctuation later.
"""

from __future__ import annotations

import re
from pathlib import Path

from lyra.text.normalize import normalize_text

SOURCE_PAGE = "Truyện Kiều"
SOURCE_REVISION = 206607
SOURCE_URL = "https://vi.wikisource.org/w/index.php?title=Truy%E1%BB%87n_Ki%E1%BB%81u&oldid=206607"
EXPECTED_LINES = 3254

_POEM = re.compile(r"<poem>(.*?)</poem>", re.DOTALL)
_LINE_NUMBER = re.compile(r"\{\{\s*số\s*\|\s*\d+\s*\}\}")
_TEMPLATE_OR_TAG = re.compile(r"\{\{.*?\}\}|<[^>]+>|\[\[|\]\]|'''|''")


# Digitization errors in the source revision, fixed by line number. In seven
# lines the final punctuation was merged into the last vowel as a tone mark
# ("." became a dot below, "?" a hook above); one line has a wrong vowel mark.
CORRECTIONS: dict[int, tuple[str, str]] = {
    506: ("làm đầụ", "làm đầu."),
    676: ("nưóc non", "nước non"),
    1054: ("ghế ngồị", "ghế ngồi."),
    1174: ("là aỉ", "là ai?"),
    1488: ("được nàọ", "được nào."),
    2920: ("muôn ngườị", "muôn người."),
    2946: ("đổi dờỉ", "đổi dời?"),
    3042: ("cũng vừạ", "cũng vừa."),
}


class CorpusError(ValueError):
    """Raised when the source text does not have the expected structure."""


def extract_poem_lines(
    wikitext: str, corrections: dict[int, tuple[str, str]] | None = None
) -> list[str]:
    """Return the verse lines inside every ``<poem>`` block, cleaned.

    ``corrections`` maps a 1-based line number to an ``(old, new)`` substring
    replacement; a correction whose ``old`` text is missing raises, so a
    changed source cannot be patched silently.
    """
    blocks = _POEM.findall(wikitext)
    if not blocks:
        raise CorpusError("no <poem> block found in the wikitext")
    lines: list[str] = []
    for block in blocks:
        for raw in block.splitlines():
            text = _LINE_NUMBER.sub("", raw)
            text = _TEMPLATE_OR_TAG.sub("", text)
            text = normalize_text(" ".join(text.split()))
            if text:
                lines.append(text)
    for number, (old, new) in (corrections or {}).items():
        if number > len(lines) or old not in lines[number - 1]:
            raise CorpusError(f"correction for line {number} does not match: {old!r}")
        lines[number - 1] = lines[number - 1].replace(old, new)
    return lines


def load_corpus(path: str | Path) -> list[str]:
    """Read a cleaned corpus file: one verse per line, blank lines ignored."""
    text = Path(path).read_text(encoding="utf-8")
    return [normalize_text(line.strip()) for line in text.splitlines() if line.strip()]
