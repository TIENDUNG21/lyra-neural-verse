from __future__ import annotations

import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from lyra.text.lucbat import (
    Rule,
    Violation,
    Waiver,
    check_lines,
    check_poem,
    load_waivers,
    split_poem,
)
from lyra.text.rhyme import RhymeMatch
from lyra.train.corpus import load_corpus

ROOT = Path(__file__).resolve().parents[2]
KIEU = load_corpus(ROOT / "data" / "truyen-kieu.txt")
WAIVERS = load_waivers(ROOT / "data" / "kieu-exceptions.tsv")

OPENING = KIEU[:4]


def rules(violations: list[Violation]) -> list[tuple[int, str, int | None]]:
    return [v.key for v in violations]


def test_opening_of_kieu_passes_with_exact_and_near_rhymes() -> None:
    report = check_lines(OPENING)
    assert report.ok
    assert [(r.first, r.second, r.match) for r in report.rhymes] == [
        ((1, 6), (2, 6), RhymeMatch.EXACT),  # ta ~ là
        ((2, 8), (3, 6), RhymeMatch.NEAR),  # nhau ~ dâu
        ((3, 6), (4, 6), RhymeMatch.NEAR),  # dâu ~ đau
    ]


def test_wrong_tone_is_reported_with_line_and_position() -> None:
    report = check_poem("Trăm năm trong cõi người ta\nChữ tài chữ mệnh khéo là ghét nhau")
    assert report.ok
    bad = check_poem("Trăm năm trong ngày người ta\nChữ tài chữ mệnh khéo là ghét nhau")
    assert rules(bad.violations) == [(1, "tone", 4)]
    assert bad.violations[0].message == "'ngày' (huyền) should be trắc"


def test_huyen_ngang_contrast_on_the_eight_line() -> None:
    report = check_poem("Trăm năm trong cõi người ta / Chữ tài chữ mệnh khéo là ghét nhà")
    assert rules(report.violations) == [(2, "contrast", 8)]


def test_missing_rhyme_is_attributed_to_the_later_line() -> None:
    report = check_poem("Trăm năm trong cõi người ta / Chữ tài chữ mệnh khéo lòng ghét nhau")
    assert rules(report.violations) == [(2, "rhyme", 6)]
    assert "does not rhyme with 'ta'" in report.violations[0].message


def test_rhyme_from_eight_line_to_next_six_line() -> None:
    poem = [*OPENING[:2], "Trải qua một cuộc bể sông"]
    report = check_lines(poem)
    assert rules(report.violations) == [(3, "rhyme", 6)]


def test_wrong_length_skips_position_rules_for_that_line() -> None:
    report = check_poem("Trăm năm trong cõi người ta ơi / Chữ tài chữ mệnh khéo là ghét nhau")
    assert rules(report.violations) == [(1, "length", None)]
    assert report.rhymes == []


def test_unknown_syllable_at_a_constrained_position() -> None:
    report = check_poem("Trăm năm trong cõi người photo")
    assert rules(report.violations) == [(1, "syllable", 6)]


def test_nfd_and_zero_width_input_give_the_same_report() -> None:
    text = "\n".join(KIEU[:20])
    noisy = unicodedata.normalize("NFD", text).replace("người", "ng\u200bười")
    a, b = check_poem(text), check_poem(noisy)
    assert rules(a.violations) == rules(b.violations)
    assert [r.match for r in a.rhymes] == [r.match for r in b.rhymes]


def test_split_poem_accepts_newlines_and_slashes() -> None:
    assert split_poem("a b / c d\n\n e |f") == ["a b", "c d", "e", "f"]


def test_empty_poem_is_an_error() -> None:
    with pytest.raises(ValueError, match="no lines"):
        check_poem(" \n / ")


def test_emoji_poem_does_not_crash() -> None:
    with pytest.raises(ValueError, match="no lines"):
        check_poem("\U0001f600 \U0001f600")
    assert check_poem("ta \U0001f600").violations[0].rule is Rule.LENGTH


def test_waivers_move_violations_to_waived_and_report_unused() -> None:
    bad = "Trăm năm trong ngày người ta\nChữ tài chữ mệnh khéo là ghét nhau"
    used = Waiver(1, Rule.TONE, 4, "test")
    unused = Waiver(2, Rule.RHYME, 6, "not needed")
    report = check_poem(bad, waivers=[used, unused])
    assert report.ok
    assert report.waived[0][1] is used
    assert report.unused_waivers == [unused]


def test_all_of_kieu_passes_with_its_documented_exceptions() -> None:
    report = check_lines(KIEU, waivers=WAIVERS)
    assert report.violations == []
    assert report.unused_waivers == []
    assert len(report.waived) == len(WAIVERS)


def test_kieu_exceptions_are_exactly_the_strict_violations() -> None:
    strict = check_lines(KIEU)
    assert sorted(v.key for v in strict.violations) == sorted(w.key for w in WAIVERS)
    counts = Counter(w.rule for w in WAIVERS)
    assert set(counts) == {Rule.TONE, Rule.RHYME}
    # Every tone exception is a 6-line (the 3/3-rhythm variant); 8-lines have none.
    assert all(w.line % 2 == 1 for w in WAIVERS if w.rule is Rule.TONE)
    # No 8-line in the poem breaks the ngang/huyền contrast.
    assert Rule.CONTRAST not in counts


def test_kieu_rhymes_are_mostly_exact() -> None:
    strict = check_lines(KIEU)
    matches = Counter(r.match for r in strict.rhymes)
    assert len(strict.rhymes) == len(KIEU) - 1
    assert matches[RhymeMatch.EXACT] > matches[RhymeMatch.NEAR] > matches[RhymeMatch.NONE]
