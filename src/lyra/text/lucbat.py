"""Lục bát (6-8) form checker.

A lục bát poem alternates a 6-syllable line (câu lục) and an 8-syllable line
(câu bát). The rules checked here:

- LENGTH: 6 and 8 syllables, alternating, starting with 6.
- TONE: at the even positions, the 6-line needs bằng-trắc-bằng (2, 4, 6) and
  the 8-line bằng-trắc-bằng-bằng (2, 4, 6, 8). Odd positions are free.
- CONTRAST: syllables 6 and 8 of the 8-line are both bằng but must differ:
  one ngang and one huyền.
- RHYME: syllable 6 of the 6-line rhymes with syllable 6 of the following
  8-line; syllable 8 of the 8-line rhymes with syllable 6 of the next 6-line.
  Exact and near rhymes (see ``lyra.text.rhyme``) are accepted.
- SYLLABLE: a constrained position must hold a Vietnamese syllable.

The checker is strict. Known deviations in real poems are not absorbed into
the rules; they are declared as ``Waiver`` records (see
``data/kieu-exceptions.tsv``) and reported separately.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from lyra.text.normalize import split_syllables
from lyra.text.phonology import Syllable, Tone, parse_syllable
from lyra.text.rhyme import RhymeMatch, rhyme_match

# Required class at each constrained position: True = bằng, False = trắc.
TONE_PATTERN: dict[int, dict[int, bool]] = {
    6: {2: True, 4: False, 6: True},
    8: {2: True, 4: False, 6: True, 8: True},
}


class Rule(Enum):
    LENGTH = "length"
    TONE = "tone"
    CONTRAST = "contrast"
    RHYME = "rhyme"
    SYLLABLE = "syllable"


@dataclass(frozen=True)
class Violation:
    line: int  # 1-based
    rule: Rule
    position: int | None  # 1-based syllable position, None for whole-line rules
    message: str

    @property
    def key(self) -> tuple[int, str, int | None]:
        return (self.line, self.rule.value, self.position)


@dataclass(frozen=True)
class Waiver:
    """A declared, explained exception for one rule at one place."""

    line: int
    rule: Rule
    position: int | None
    note: str

    @property
    def key(self) -> tuple[int, str, int | None]:
        return (self.line, self.rule.value, self.position)


@dataclass(frozen=True)
class RhymeLink:
    """A rhyme pair: (line, position) of both syllables and how well they match."""

    first: tuple[int, int]
    second: tuple[int, int]
    match: RhymeMatch


@dataclass(frozen=True)
class LineReport:
    number: int
    expected_length: int
    syllables: tuple[str, ...]
    parsed: tuple[Syllable | None, ...]

    @property
    def length_ok(self) -> bool:
        return len(self.syllables) == self.expected_length

    def at(self, position: int) -> Syllable | None:
        return self.parsed[position - 1] if position <= len(self.parsed) else None


@dataclass
class CheckReport:
    lines: list[LineReport]
    violations: list[Violation] = field(default_factory=list)
    waived: list[tuple[Violation, Waiver]] = field(default_factory=list)
    unused_waivers: list[Waiver] = field(default_factory=list)
    rhymes: list[RhymeLink] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


_LINE_SEPARATORS = re.compile(r"[\n/|]+")


def split_poem(text: str) -> list[str]:
    """Split a poem into lines on newlines, ``/`` or ``|``; empty lines are dropped."""
    return [line.strip() for line in _LINE_SEPARATORS.split(text) if split_syllables(line)]


def check_poem(text: str, *, waivers: Iterable[Waiver] = ()) -> CheckReport:
    """Check a poem given as one string (see ``split_poem``)."""
    return check_lines(split_poem(text), waivers=waivers)


def check_lines(lines: Sequence[str], *, waivers: Iterable[Waiver] = ()) -> CheckReport:
    """Check verse lines (the first one is a 6-line) against the lục bát rules."""
    if not lines:
        raise ValueError("nothing to check: the poem has no lines")
    reports = []
    for index, line in enumerate(lines):
        syllables = tuple(split_syllables(line))
        reports.append(
            LineReport(
                number=index + 1,
                expected_length=6 if index % 2 == 0 else 8,
                syllables=syllables,
                parsed=tuple(parse_syllable(s) for s in syllables),
            )
        )
    found: list[Violation] = []
    rhymes: list[RhymeLink] = []
    for index, report in enumerate(reports):
        found.extend(_check_line(report))
        if index > 0:
            link, violation = _check_rhyme(reports[index - 1], report)
            if link is not None:
                rhymes.append(link)
            if violation is not None:
                found.append(violation)

    by_key = {w.key: w for w in waivers}
    result = CheckReport(lines=reports, rhymes=rhymes)
    used: set[tuple[int, str, int | None]] = set()
    for violation in found:
        waiver = by_key.get(violation.key)
        if waiver is None:
            result.violations.append(violation)
        else:
            result.waived.append((violation, waiver))
            used.add(waiver.key)
    result.unused_waivers = [w for w in by_key.values() if w.key not in used]
    return result


def _check_line(report: LineReport) -> list[Violation]:
    if not report.length_ok:
        return [
            Violation(
                report.number,
                Rule.LENGTH,
                None,
                f"expected {report.expected_length} syllables, found {len(report.syllables)}",
            )
        ]
    out: list[Violation] = []
    for position, need_bang in TONE_PATTERN[report.expected_length].items():
        syllable = report.at(position)
        word = report.syllables[position - 1]
        if syllable is None:
            out.append(
                Violation(report.number, Rule.SYLLABLE, position, f"'{word}' is not Vietnamese")
            )
        elif syllable.is_bang != need_bang:
            want = "bằng" if need_bang else "trắc"
            out.append(
                Violation(
                    report.number,
                    Rule.TONE,
                    position,
                    f"'{word}' ({syllable.tone.name.lower()}) should be {want}",
                )
            )
    if report.expected_length == 8:
        sixth, eighth = report.at(6), report.at(8)
        if sixth and eighth and sixth.is_bang and eighth.is_bang and sixth.tone == eighth.tone:
            kind = "huyền" if sixth.tone is Tone.HUYEN else "ngang"
            out.append(
                Violation(
                    report.number,
                    Rule.CONTRAST,
                    8,
                    f"syllables 6 and 8 are both {kind}; one must be ngang, the other huyền",
                )
            )
    return out


def _check_rhyme(
    previous: LineReport, current: LineReport
) -> tuple[RhymeLink | None, Violation | None]:
    """Rhyme between the end of ``previous`` and syllable 6 of ``current``."""
    if not (previous.length_ok and current.length_ok):
        return None, None
    source_pos = previous.expected_length  # 6 of a 6-line, 8 of an 8-line
    source, target = previous.at(source_pos), current.at(6)
    if source is None or target is None:
        return None, None
    match = rhyme_match(source, target)
    link = RhymeLink((previous.number, source_pos), (current.number, 6), match)
    if match.ok:
        return link, None
    message = (
        f"'{target.text}' does not rhyme with '{source.text}' "
        f"(line {previous.number}, syllable {source_pos})"
    )
    return link, Violation(current.number, Rule.RHYME, 6, message)


def load_waivers(path: str | Path) -> list[Waiver]:
    """Read waivers from a TSV file with columns line, rule, position, note."""
    waivers = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            position = row["position"].strip()
            waivers.append(
                Waiver(
                    line=int(row["line"]),
                    rule=Rule(row["rule"].strip()),
                    position=int(position) if position else None,
                    note=row["note"].strip(),
                )
            )
    return waivers
