from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from lyra.io.modelfile import write_model_file
from lyra.nn import LyraLM, ModelConfig
from lyra.nn.inference import InferenceEngine
from lyra.text.tokenizer import Tokenizer
from lyra.train.corpus import load_corpus
from lyra.train.data import encode_blocks, split_blocks
from lyra.train.evaluate import engine_perplexity, evaluate_file, ngram_baselines
from lyra.train.loop import evaluate

ROOT = Path(__file__).resolve().parents[2]
LINES = load_corpus(ROOT / "data" / "truyen-kieu.txt")[:400]
TOKENIZER = Tokenizer.build(LINES)
SPLIT = split_blocks(LINES, 40, 5)
TRAIN = encode_blocks(SPLIT.train, TOKENIZER)
VAL = encode_blocks(SPLIT.val, TOKENIZER)
CFG = ModelConfig(vocab_size=TOKENIZER.vocab_size, block_size=16, d_model=16, n_layers=1, n_heads=2)


def test_engine_perplexity_matches_training_time_evaluation() -> None:
    model = LyraLM(CFG)
    float_eval = evaluate(model, VAL)
    engine = engine_perplexity(InferenceEngine(CFG, model.state_dict()), VAL)
    assert math.isclose(engine.all_tokens, float_eval.ppl, rel_tol=1e-4)
    assert math.isclose(engine.syllables, float_eval.syllable_ppl, rel_tol=1e-4)


def test_ngram_baselines_are_ordered_sensibly() -> None:
    result = ngram_baselines(TRAIN, VAL, TOKENIZER.vocab_size)
    assert 1.0 < result["bigram"] < result["unigram"] < TOKENIZER.vocab_size * 2
    assert 0.0 < result["bigram_lambda"] < 1.0


def test_evaluate_file_reports_model_and_baselines(tmp_path: Path) -> None:
    model = LyraLM(CFG)
    path = tmp_path / "m.bin"
    write_model_file(path, params=model.state_dict(), config=CFG, vocab=list(TOKENIZER.vocab))
    result = evaluate_file(path, "unused", block_lines=40, val_every=5, lines=LINES)
    assert set(result) == {"val_ppl", "val_syllable_ppl", "unigram", "bigram", "bigram_lambda"}
    # An untrained model is close to uniform over the vocabulary.
    assert np.isclose(result["val_syllable_ppl"], TOKENIZER.vocab_size, rtol=0.2)
