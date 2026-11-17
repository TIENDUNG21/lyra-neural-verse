"""Write the Python reference outputs that the TypeScript port must reproduce.

Usage::

    python scripts/make_parity_fixture.py            # writes web/test/fixtures/parity.json

The fixture holds, for the committed model file:

- logits after feeding fixed token sequences (one of them longer than the
  context, so the re-prefill path is exercised);
- token sequences produced by the constrained decoder for fixed seeds and
  settings (greedy, sampled, strict rhyme, with a prompt, long enough to
  overflow the context);
- the phonological parse of every vocabulary entry plus edge cases;
- the strict checker's verdict on all of Truyện Kiều.

``tests/test_parity_fixture.py`` rebuilds it and fails if the committed file
is stale, so the browser tests always compare against the current model.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyra.decode.generate import Poet  # noqa: E402
from lyra.decode.sampling import SamplingSettings  # noqa: E402
from lyra.text.lucbat import check_lines  # noqa: E402
from lyra.text.phonology import parse_syllable  # noqa: E402
from lyra.train.corpus import load_corpus  # noqa: E402

MODEL = ROOT / "models" / "lyra-kieu-q8.bin"
CORPUS = ROOT / "data" / "truyen-kieu.txt"
OUT = ROOT / "web" / "test" / "fixtures" / "parity.json"

GENERATIONS: list[dict[str, Any]] = [
    {"lines": 4, "seed": 1, "temperature": 0.9, "top_k": 40, "top_p": 0.95},
    {"lines": 4, "seed": 2, "temperature": 0.0, "top_k": 0, "top_p": 1.0},
    {"lines": 6, "seed": 3, "temperature": 1.0, "top_k": 0, "top_p": 1.0, "strict_rhyme": True},
    {"lines": 4, "seed": 4, "temperature": 0.8, "top_k": 20, "top_p": 0.9, "prompt": 1},
    {"lines": 20, "seed": 5, "temperature": 0.9, "top_k": 40, "top_p": 0.95},
]
EXTRA_WORDS = [
    "hoà",
    "thuỷ",
    unicodedata.normalize("NFD", "người"),
    "GIẾNG",
    "gì",
    "quốc",
    "khuỷu",
    "photo",
    "đầụ",
    "ðường",
    "ng​ười",
]


def _round(values: Any) -> list[float]:
    return [round(float(v), 5) for v in values]


def parse_record(word: str) -> list[Any] | None:
    s = parse_syllable(word)
    if s is None:
        return None
    return [s.text, s.onset, s.medial, s.nucleus, s.coda, s.tone.value]


def build_fixture() -> dict[str, Any]:
    poet = Poet.from_file(MODEL)
    tokenizer = poet.tokenizer
    kieu = load_corpus(CORPUS)
    sequences = {
        "couplet": tokenizer.encode_poem(kieu[:2]),
        "prefix": tokenizer.encode_poem(kieu[:3])[:-3],
        "overflow": tokenizer.encode_poem(kieu[:20]),
    }
    logits = {}
    for name, tokens in sequences.items():
        poet.engine.reset()
        logits[name] = {"tokens": tokens, "logits": _round(poet.engine.feed(tokens))}

    generations = []
    for case in GENERATIONS:
        prompt = kieu[: case.get("prompt", 0)]
        poem = poet.generate(
            lines=case["lines"],
            prompt=prompt,
            settings=SamplingSettings(case["temperature"], case["top_k"], case["top_p"]),
            seed=case["seed"],
            strict_rhyme=case.get("strict_rhyme", False),
        )
        generations.append({**case, "prompt_lines": prompt, "tokens": list(poem.tokens)})

    strict = check_lines(kieu)
    return {
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "logits": logits,
        "generations": generations,
        "vocab_parses": [
            parse_record(t) if not t.startswith("<") else None for t in tokenizer.vocab
        ],
        "extra_words": [[w, parse_record(w)] for w in EXTRA_WORDS],
        "kieu": {
            "violations": [[v.line, v.rule.value, v.position] for v in strict.violations],
            "rhymes": [r.match.value for r in strict.rhymes],
        },
    }


def render(fixture: dict[str, Any]) -> str:
    return json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(build_fixture()), encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
