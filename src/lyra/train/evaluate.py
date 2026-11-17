"""Held-out evaluation of a model file, plus n-gram baselines for context.

The numbers in the README come from here (``lyra evaluate``): validation
perplexity of the released int8 file on the validation blocks, and two
baselines computed on the same split so the model's number has a reference:

- unigram: add-0.5 smoothed syllable frequencies;
- bigram: linear interpolation of bigram and unigram estimates, with the
  interpolation weight chosen on a slice of the *training* blocks.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lyra.io.modelfile import read_model_file
from lyra.nn.inference import InferenceEngine
from lyra.text.tokenizer import LINE6, LINE8, Tokenizer
from lyra.train.corpus import load_corpus
from lyra.train.data import IntArray, encode_blocks, eval_windows, split_blocks


@dataclass(frozen=True)
class Perplexities:
    all_tokens: float
    syllables: float


def _log_softmax_rows(logits: NDArray[Any]) -> NDArray[np.float64]:
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    return np.asarray(shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True)))


def engine_perplexity(engine: InferenceEngine, tokens: IntArray) -> Perplexities:
    """Perplexity with the same half-overlapping windows as training-time evaluation."""
    results = []
    for skip in (False, True):
        nll, count = 0.0, 0
        for x, y in eval_windows(tokens, engine.config.block_size, skip_markers=skip):
            logp = _log_softmax_rows(engine.logits_all([int(t) for t in x[0]]))
            for position, target in enumerate(y[0]):
                if target >= 0:
                    nll -= float(logp[position, target])
                    count += 1
        results.append(math.exp(nll / max(count, 1)))
    return Perplexities(all_tokens=results[0], syllables=results[1])


def _syllable_pairs(tokens: IntArray) -> list[tuple[int, int]]:
    return [
        (int(a), int(b)) for a, b in itertools.pairwise(tokens.tolist()) if b not in (LINE6, LINE8)
    ]


def ngram_baselines(train: IntArray, val: IntArray, vocab_size: int) -> dict[str, float]:
    """Syllable perplexity of unigram and interpolated bigram models."""
    held = len(train) // 10
    fit, tune = train[:-held], train[-held:]

    def unigram_logp(counts: Counter[int], total: int, token: int) -> float:
        return math.log((counts[token] + 0.5) / (total + 0.5 * vocab_size))

    def bigram_nll(data: IntArray, source: IntArray, lam: float) -> float:
        uni = Counter(int(t) for t in source)
        bi = Counter(zip(source[:-1].tolist(), source[1:].tolist(), strict=True))
        total = len(source)
        nll = 0.0
        pairs = _syllable_pairs(data)
        for a, b in pairs:
            p_uni = math.exp(unigram_logp(uni, total, b))
            p_bi = bi[(a, b)] / uni[a] if uni[a] else 0.0
            nll -= math.log(lam * p_bi + (1.0 - lam) * p_uni)
        return nll / len(pairs)

    best_lam = min((i / 20 for i in range(1, 20)), key=lambda lam: bigram_nll(tune, fit, lam))
    counts = Counter(int(t) for t in train)
    pairs = _syllable_pairs(val)
    unigram = -sum(unigram_logp(counts, len(train), b) for _, b in pairs) / len(pairs)
    return {
        "unigram": math.exp(unigram),
        "bigram": math.exp(bigram_nll(val, train, best_lam)),
        "bigram_lambda": best_lam,
    }


def evaluate_file(
    model_path: str | Path,
    corpus_path: str | Path,
    *,
    block_lines: int = 40,
    val_every: int = 10,
    lines: Sequence[str] | None = None,
) -> dict[str, float]:
    model = read_model_file(model_path)
    corpus = list(lines) if lines is not None else load_corpus(corpus_path)
    tokenizer = Tokenizer(model.vocab)
    split = split_blocks(corpus, block_lines, val_every)
    if not split.val or not split.train:
        raise ValueError("the corpus is too short for a train/validation split")
    train = encode_blocks(split.train, tokenizer)
    val = encode_blocks(split.val, tokenizer)
    ppl = engine_perplexity(InferenceEngine(model.config, model.weights), val)
    return {
        "val_ppl": ppl.all_tokens,
        "val_syllable_ppl": ppl.syllables,
        **ngram_baselines(train, val, tokenizer.vocab_size),
    }
