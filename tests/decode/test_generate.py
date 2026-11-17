from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lyra.decode.constraints import DecodeState, LucBatConstraints
from lyra.decode.generate import Poet
from lyra.decode.sampling import SamplingSettings
from lyra.nn import LyraLM, ModelConfig
from lyra.nn.inference import InferenceEngine
from lyra.text.features import token_features
from lyra.text.lucbat import check_lines
from lyra.text.phonology import Tone, parse_syllable
from lyra.text.rhyme import RhymeMatch
from lyra.text.tokenizer import LINE6, LINE8, UNK, Tokenizer
from lyra.train.corpus import load_corpus

ROOT = Path(__file__).resolve().parents[2]
KIEU = load_corpus(ROOT / "data" / "truyen-kieu.txt")
TOKENIZER = Tokenizer.build(KIEU)
FEATURES = token_features(TOKENIZER.vocab)


@pytest.fixture(scope="module")
def poet() -> Poet:
    cfg = ModelConfig(
        vocab_size=TOKENIZER.vocab_size, block_size=32, d_model=16, n_layers=1, n_heads=2
    )
    model = LyraLM(cfg)
    return Poet(InferenceEngine(cfg, model.state_dict()), TOKENIZER.vocab)


def tone_of(token: int) -> Tone:
    syllable = parse_syllable(TOKENIZER.token(token))
    assert syllable is not None
    return syllable.tone


def test_line_start_forces_the_marker() -> None:
    c = LucBatConstraints(FEATURES, TOKENIZER.vocab)
    step = c.step(DecodeState())
    assert step.forced == LINE6
    assert step.mask.sum() == 1
    state = DecodeState(lines_done=1)
    assert c.step(state).forced == LINE8


def test_tone_masks_follow_the_pattern() -> None:
    c = LucBatConstraints(FEATURES, TOKENIZER.vocab)
    state = c.advance(DecodeState(), LINE6)
    first = c.step(state).mask
    assert not first[UNK]
    assert not first[LINE6]
    assert first.sum() == FEATURES.is_syllable.sum()
    state = c.advance(state, TOKENIZER.token_id("trăm"))
    second = np.flatnonzero(c.step(state).mask)
    assert all(tone_of(int(t)).is_bang for t in second)
    state = c.advance(state, TOKENIZER.token_id("năm"))
    state = c.advance(state, TOKENIZER.token_id("trong"))
    fourth = np.flatnonzero(c.step(state).mask)
    assert all(not tone_of(int(t)).is_bang for t in fourth)


def test_eight_line_contrast_and_rhymes() -> None:
    c = LucBatConstraints(FEATURES, TOKENIZER.vocab)
    state = DecodeState()
    for line in KIEU[:1]:
        state = c.advance(state, LINE6)
        for s in TOKENIZER.encode_line(line):
            state = c.advance(state, s)
    state = c.advance(state, LINE8)
    for word in ["chữ", "tài", "chữ", "mệnh", "khéo"]:
        state = c.advance(state, TOKENIZER.token_id(word))
    sixth = np.flatnonzero(c.step(state).mask)
    assert TOKENIZER.token_id("là") in sixth
    assert TOKENIZER.token_id("nhau") not in sixth
    state = c.advance(state, TOKENIZER.token_id("là"))  # huyền at 6
    state = c.advance(state, TOKENIZER.token_id("ghét"))
    eighth = np.flatnonzero(c.step(state).mask)
    assert {tone_of(int(t)) for t in eighth} == {Tone.NGANG}


def test_advance_rejects_out_of_place_markers() -> None:
    c = LucBatConstraints(FEATURES, TOKENIZER.vocab)
    with pytest.raises(ValueError, match="marker"):
        c.advance(DecodeState(), TOKENIZER.token_id("ta"))
    with pytest.raises(ValueError, match="inside"):
        c.advance(DecodeState(in_line=True), LINE8)


@pytest.mark.parametrize("seed", range(6))
def test_generated_poems_pass_the_strict_checker(poet: Poet, seed: int) -> None:
    poem = poet.generate(lines=6, seed=seed, settings=SamplingSettings(1.0, 0, 1.0))
    assert len(poem.lines) == 6
    report = check_lines([line.text() for line in poem.lines])
    assert report.violations == []
    assert not any(line.relaxed for line in poem.lines)


def test_generation_is_deterministic_per_seed(poet: Poet) -> None:
    a = poet.generate(lines=4, seed=11)
    b = poet.generate(lines=4, seed=11)
    c = poet.generate(lines=4, seed=12)
    assert a == b
    assert a.text() != c.text()


def test_prompt_is_continued_and_answered(poet: Poet) -> None:
    poem = poet.generate(lines=2, prompt=[KIEU[0]], seed=3)
    assert poem.lines[0].from_prompt
    assert poem.lines[0].text() == "Trăm năm trong cõi người ta"
    report = check_lines([line.text() for line in poem.lines])
    assert report.ok
    assert report.rhymes[0].match is not RhymeMatch.NONE


def test_strict_rhyme_only_accepts_exact_rhymes(poet: Poet) -> None:
    for seed in range(4):
        poem = poet.generate(lines=4, seed=seed, strict_rhyme=True)
        report = check_lines([line.text() for line in poem.lines])
        assert {r.match for r in report.rhymes} == {RhymeMatch.EXACT}


def test_unrhymable_prompt_relaxes_and_flags(poet: Poet) -> None:
    poem = poet.generate(lines=2, prompt=["Đêm nay ngồi ngắm con boong"], seed=1, strict_rhyme=True)
    assert poem.lines[1].relaxed


def test_bad_prompts_and_arguments(poet: Poet) -> None:
    with pytest.raises(ValueError, match="expected 6"):
        poet.generate(lines=2, prompt=["quá ngắn"])
    with pytest.raises(ValueError, match="more lines"):
        poet.generate(lines=1, prompt=[KIEU[0], KIEU[1]])
    with pytest.raises(ValueError, match="lines"):
        poet.generate(lines=0)


def test_greedy_decoding_is_the_argmax_path(poet: Poet) -> None:
    a = poet.generate(lines=2, seed=1, settings=SamplingSettings(temperature=0.0))
    b = poet.generate(lines=2, seed=99, settings=SamplingSettings(temperature=0.0))
    assert a == b
