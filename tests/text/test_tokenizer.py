from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lyra.text.tokenizer import LINE6, LINE8, PAD, SPECIALS, UNK, Tokenizer
from lyra.train.corpus import load_corpus

ROOT = Path(__file__).resolve().parents[2]
KIEU = load_corpus(ROOT / "data" / "truyen-kieu.txt")
TOKENIZER = Tokenizer.build(KIEU)


def test_specials_come_first_with_fixed_ids() -> None:
    assert TOKENIZER.vocab[:4] == SPECIALS
    assert (PAD, UNK, LINE6, LINE8) == (0, 1, 2, 3)


def test_build_is_deterministic_and_frequency_ordered() -> None:
    again = Tokenizer.build(reversed(KIEU))
    assert again.vocab == TOKENIZER.vocab
    assert TOKENIZER.vocab_size == 4 + len(
        {t for line in KIEU for t in TOKENIZER.encode_line(line)}
    )


def test_kieu_vocabulary_has_no_unknowns() -> None:
    assert all(UNK not in TOKENIZER.encode_line(line) for line in KIEU)


def test_encode_poem_alternates_markers() -> None:
    ids = TOKENIZER.encode_poem(KIEU[:3])
    assert ids[0] == LINE6
    assert ids[7] == LINE8
    assert ids[16] == LINE6
    assert len(ids) == 3 + 6 + 8 + 6


def test_decode_lines_inverts_encode() -> None:
    ids = TOKENIZER.encode_poem(KIEU[:2])
    assert TOKENIZER.decode_lines(ids) == [
        ["trăm", "năm", "trong", "cõi", "người", "ta"],
        ["chữ", "tài", "chữ", "mệnh", "khéo", "là", "ghét", "nhau"],
    ]


def test_unknown_syllables_map_to_unk() -> None:
    assert TOKENIZER.token_id("photo") == UNK
    assert TOKENIZER.token_id("\U0001f600") == UNK


def test_tone_placement_variants_share_a_token() -> None:
    assert TOKENIZER.token_id("hoà") == TOKENIZER.token_id("hòa") != UNK
    assert "hoà" in TOKENIZER


def test_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    TOKENIZER.save(path)
    assert Tokenizer.load(path).vocab == TOKENIZER.vocab


def test_invalid_vocabularies_are_rejected() -> None:
    with pytest.raises(ValueError, match="special"):
        Tokenizer(["a", "b"])
    with pytest.raises(ValueError, match="duplicates"):
        Tokenizer([*SPECIALS, "ta", "ta"])
    with pytest.raises(ValueError, match="tokenizer"):
        Tokenizer.from_json('{"vocab": []}')


@settings(max_examples=50, deadline=None)
@given(st.sampled_from(KIEU), st.sampled_from(["NFC", "NFD"]), st.booleans())
def test_encoding_ignores_normalization_form_and_case(line: str, form: str, upper: bool) -> None:
    variant = unicodedata.normalize(form, line.upper() if upper else line)  # type: ignore[arg-type]
    assert TOKENIZER.encode_line(variant) == TOKENIZER.encode_line(line)
