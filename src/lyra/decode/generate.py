"""Generate lục bát poems: model logits + constraint masks + sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from lyra.decode.constraints import DecodeState, LucBatConstraints
from lyra.decode.rng import Mulberry32
from lyra.decode.sampling import SamplingSettings, sample_token
from lyra.io.modelfile import read_model_file
from lyra.nn.inference import InferenceEngine
from lyra.text.features import token_features
from lyra.text.normalize import split_syllables
from lyra.text.tokenizer import Tokenizer, line_marker


@dataclass(frozen=True)
class PoemLine:
    syllables: tuple[str, ...]
    from_prompt: bool
    relaxed: bool  # the rhyme constraint had to be dropped in this line

    def text(self) -> str:
        words = " ".join(self.syllables)
        return words[:1].upper() + words[1:]


@dataclass(frozen=True)
class Poem:
    lines: tuple[PoemLine, ...]
    tokens: tuple[int, ...] = field(default=())

    def text(self) -> str:
        return "\n".join(line.text() for line in self.lines)


class Poet:
    """A loaded model plus everything needed to write constrained verse."""

    def __init__(self, engine: InferenceEngine, vocab: Sequence[str]) -> None:
        self.engine = engine
        self.tokenizer = Tokenizer(vocab)
        self.features = token_features(vocab)

    @classmethod
    def from_file(cls, path: str | Path) -> Poet:
        model = read_model_file(path)
        return cls(InferenceEngine(model.config, model.weights), model.vocab)

    def generate(
        self,
        *,
        lines: int = 4,
        prompt: Sequence[str] = (),
        settings: SamplingSettings | None = None,
        seed: int = 0,
        strict_rhyme: bool = False,
    ) -> Poem:
        """Write a poem of ``lines`` lines, continuing the ``prompt`` lines if given."""
        if lines < 1:
            raise ValueError("lines must be >= 1")
        if len(prompt) > lines:
            raise ValueError("the prompt has more lines than requested")
        settings = settings or SamplingSettings()
        constraints = LucBatConstraints(
            self.features, self.tokenizer.vocab, strict_rhyme=strict_rhyme
        )
        rng = Mulberry32(seed)
        engine = self.engine
        engine.reset()
        state = DecodeState()
        tokens: list[int] = []
        out: list[PoemLine] = []

        for index, line in enumerate(prompt):
            syllables = split_syllables(line)
            expected = 6 if index % 2 == 0 else 8
            if len(syllables) != expected:
                raise ValueError(
                    f"prompt line {index + 1} has {len(syllables)} syllables; expected {expected}"
                )
            marker = line_marker(index)
            ids = [marker] + [self.tokenizer.token_id(s) for s in syllables]
            state = constraints.advance(state, marker)
            for s, token in zip(syllables, ids[1:], strict=True):
                state = constraints.advance(state, token, s)
            tokens.extend(ids)
            out.append(PoemLine(tuple(syllables), from_prompt=True, relaxed=False))

        logits = engine.feed(tokens) if tokens else None
        current: list[str] = []
        relaxed = False
        while state.lines_done < lines:
            step = constraints.step(state)
            if step.forced is not None:
                token = step.forced
            else:
                assert logits is not None
                token = sample_token(logits, step.mask, settings, rng)
                relaxed = relaxed or step.relaxed
                current.append(self.tokenizer.token(token))
            tokens.append(token)
            state = constraints.advance(state, token)
            if not state.in_line and step.forced is None:
                out.append(PoemLine(tuple(current), from_prompt=False, relaxed=relaxed))
                current, relaxed = [], False
            if state.lines_done < lines:
                logits = engine.feed([token])
        return Poem(tuple(out), tuple(tokens))
