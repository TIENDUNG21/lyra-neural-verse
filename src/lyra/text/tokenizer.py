"""Syllable-level tokenizer with line-type marker tokens.

Vietnamese writes one syllable per space-separated word, and lục bát rules are
stated per syllable, so syllables are the natural token. Every line is
prefixed by a marker that tells the model which line type follows:

    <l6> trăm năm trong cõi người ta <l8> chữ tài chữ mệnh khéo là ghét nhau

Tokens are canonical syllables: lowercase, NFC, tone mark on the conventional
vowel (so ``hoà`` and ``hòa`` are one token).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from lyra.text.normalize import split_syllables
from lyra.text.phonology import canonical_syllable

PAD, UNK, LINE6, LINE8 = 0, 1, 2, 3
SPECIALS = ("<pad>", "<unk>", "<l6>", "<l8>")


def line_marker(index: int) -> int:
    """Marker token for the ``index``-th line (0-based) of a lục bát poem."""
    return LINE6 if index % 2 == 0 else LINE8


class Tokenizer:
    """Maps canonical syllables to integer ids and back."""

    def __init__(self, vocab: Sequence[str]) -> None:
        if tuple(vocab[: len(SPECIALS)]) != SPECIALS:
            raise ValueError(f"vocabulary must start with the special tokens {SPECIALS}")
        if len(set(vocab)) != len(vocab):
            raise ValueError("vocabulary contains duplicates")
        self.vocab: tuple[str, ...] = tuple(vocab)
        self._ids = {token: index for index, token in enumerate(self.vocab)}

    @classmethod
    def build(cls, lines: Iterable[str], min_count: int = 1) -> Tokenizer:
        """Vocabulary of syllables seen at least ``min_count`` times.

        Ordered by descending frequency, ties alphabetically, so the result is
        deterministic.
        """
        counts = Counter(canonical_syllable(s) for line in lines for s in split_syllables(line))
        ranked = sorted(
            (t for t, c in counts.items() if c >= min_count), key=lambda t: (-counts[t], t)
        )
        return cls([*SPECIALS, *ranked])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, syllable: str) -> bool:
        return canonical_syllable(syllable) in self._ids

    def token_id(self, syllable: str) -> int:
        return self._ids.get(canonical_syllable(syllable), UNK)

    def token(self, token_id: int) -> str:
        return self.vocab[token_id]

    def is_syllable(self, token_id: int) -> bool:
        return token_id >= len(SPECIALS)

    def encode_line(self, line: str, marker: int | None = None) -> list[int]:
        ids = [self.token_id(s) for s in split_syllables(line)]
        return ids if marker is None else [marker, *ids]

    def encode_poem(self, lines: Sequence[str]) -> list[int]:
        """Encode lines alternating 6/8 markers, starting with ``<l6>``."""
        out: list[int] = []
        for index, line in enumerate(lines):
            out.extend(self.encode_line(line, line_marker(index)))
        return out

    def decode_lines(self, ids: Iterable[int]) -> list[list[str]]:
        """Split a token stream into lines (on markers) of syllable strings."""
        lines: list[list[str]] = []
        for token_id in ids:
            if token_id in (LINE6, LINE8):
                lines.append([])
            elif token_id != PAD:
                if not lines:
                    lines.append([])
                lines[-1].append(self.vocab[token_id])
        return lines

    def to_json(self) -> str:
        return json.dumps(
            {"format": "lyra-tokenizer/1", "vocab": list(self.vocab)}, ensure_ascii=False
        )

    @classmethod
    def from_json(cls, text: str) -> Tokenizer:
        data = json.loads(text)
        if data.get("format") != "lyra-tokenizer/1":
            raise ValueError("not a lyra tokenizer file")
        return cls(data["vocab"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
