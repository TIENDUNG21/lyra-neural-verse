from __future__ import annotations

from pathlib import Path

import pytest

from lyra.text.normalize import split_syllables
from lyra.text.phonology import parse_syllable
from lyra.train.corpus import (
    CORRECTIONS,
    EXPECTED_LINES,
    CorpusError,
    extract_poem_lines,
    load_corpus,
)

ROOT = Path(__file__).resolve().parents[2]
KIEU = ROOT / "data" / "truyen-kieu.txt"

WIKITEXT = """{{đầu đề | tựa đề = X}}
<poem>
\tTrăm năm trong cõi người ta,
\tChữ tài chữ mệnh khéo là ghét nhau.
{{số|3}}\tTrải qua một cuộc '''bể''' dâu,

\tNhững điều trông thấy mà đau đớn lòng.
</poem>
{{PD-old}}
"""


def test_extract_strips_templates_markup_and_blank_lines() -> None:
    lines = extract_poem_lines(WIKITEXT)
    assert lines == [
        "Trăm năm trong cõi người ta,",
        "Chữ tài chữ mệnh khéo là ghét nhau.",
        "Trải qua một cuộc bể dâu,",
        "Những điều trông thấy mà đau đớn lòng.",
    ]


def test_corrections_apply_by_line_number() -> None:
    lines = extract_poem_lines(WIKITEXT, {2: ("ghét nhau.", "ghét nhau!")})
    assert lines[1].endswith("ghét nhau!")


def test_correction_that_does_not_match_raises() -> None:
    with pytest.raises(CorpusError, match="line 2"):
        extract_poem_lines(WIKITEXT, {2: ("không có", "x")})


def test_missing_poem_block_raises() -> None:
    with pytest.raises(CorpusError, match="poem"):
        extract_poem_lines("no poem here")


def test_committed_corpus_shape() -> None:
    lines = load_corpus(KIEU)
    assert len(lines) == EXPECTED_LINES
    counts = [len(split_syllables(line)) for line in lines]
    assert counts == [6, 8] * (EXPECTED_LINES // 2)
    assert lines[0] == "Trăm năm trong cõi người ta,"
    assert lines[-1] == "Mua vui cũng được một vài trống canh."


def test_committed_corpus_has_corrections_applied_and_every_syllable_parses() -> None:
    lines = load_corpus(KIEU)
    for number, (_old, new) in CORRECTIONS.items():
        assert new in lines[number - 1]
    unparsed = {s for line in lines for s in split_syllables(line) if parse_syllable(s) is None}
    assert unparsed == set()
