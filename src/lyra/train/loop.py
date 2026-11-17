"""The training loop: batches, AdamW updates, evaluation, logs and checkpoints.

Outputs in ``out_dir``:

- ``log.csv``: one row every ``log_every`` steps (mean train loss, lr, grad
  norm, throughput) plus validation metrics on evaluation steps;
- ``tokenizer.json``: the vocabulary;
- ``best.npz``: the checkpoint with the lowest validation loss;
- ``last.npz``: the latest checkpoint, with optimizer and RNG state, used by
  ``resume=True``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lyra.autograd import no_grad
from lyra.nn import LyraLM, ModelConfig
from lyra.optim import AdamW, clip_grad_norm, load_checkpoint, save_checkpoint, warmup_cosine
from lyra.text.features import token_features
from lyra.text.tokenizer import Tokenizer
from lyra.train.corpus import load_corpus
from lyra.train.csvlog import CsvLogger
from lyra.train.data import BatchSampler, IntArray, encode_blocks, eval_windows, split_blocks

LOG_FIELDS = (
    "step",
    "elapsed_s",
    "lr",
    "train_loss",
    "grad_norm",
    "tokens_per_s",
    "val_loss",
    "val_ppl",
    "val_syllable_ppl",
)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    batch_size: int = 16
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    token_dropout: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 100
    log_every: int = 10
    seed: int = 1234
    block_lines: int = 40
    val_every: int = 10
    max_minutes: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown train config keys: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class EvalResult:
    loss: float  # mean NLL over every token (syllables and line markers)
    syllable_loss: float  # mean NLL over syllable tokens only

    @property
    def ppl(self) -> float:
        return math.exp(self.loss)

    @property
    def syllable_ppl(self) -> float:
        return math.exp(self.syllable_loss)


@dataclass(frozen=True)
class TrainResult:
    steps: int
    best_step: int
    best_val_loss: float
    best_val_syllable_ppl: float
    elapsed_s: float


def evaluate(model: LyraLM, tokens: IntArray) -> EvalResult:
    """Validation NLL with the half-overlapping windows of ``eval_windows``."""
    was_training = model.training
    model.eval()
    totals = []
    with no_grad():
        for skip in (False, True):
            nll, count = 0.0, 0
            for x, y in eval_windows(tokens, model.config.block_size, skip_markers=skip):
                scored = int((y >= 0).sum())
                if scored:
                    nll += model.loss(x, y).item() * scored
                    count += scored
            totals.append(nll / max(count, 1))
    model.train(was_training)
    return EvalResult(loss=totals[0], syllable_loss=totals[1])


def load_run_config(path: str | Path) -> tuple[dict[str, Any], TrainConfig]:
    """Read ``{"model": {...}, "train": {...}}``; ``vocab_size`` comes from the data."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    model = dict(data.get("model", {}))
    if "vocab_size" in model:
        raise ValueError("vocab_size is derived from the corpus; remove it from the config")
    return model, TrainConfig.from_dict(dict(data.get("train", {})))


def train(
    *,
    corpus_path: str | Path,
    model_options: dict[str, Any],
    config: TrainConfig,
    out_dir: str | Path,
    resume: bool = False,
    report: Callable[[str], None] = print,
) -> TrainResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = load_corpus(corpus_path)
    tokenizer = Tokenizer.build(lines)
    tokenizer.save(out / "tokenizer.json")
    split = split_blocks(lines, config.block_lines, config.val_every)
    train_tokens = encode_blocks(split.train, tokenizer)
    val_tokens = encode_blocks(split.val, tokenizer)
    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size, **model_options)
    model = LyraLM(model_cfg, token_features(tokenizer.vocab))
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)
    sampler = BatchSampler(
        train_tokens, model_cfg.block_size, config.batch_size, rng, config.token_dropout
    )

    start_step, best_loss, best_step, best_syllable_ppl, elapsed_before = (
        0,
        math.inf,
        -1,
        math.inf,
        0.0,
    )
    last_path = out / "last.npz"
    if resume and last_path.exists():
        ckpt = load_checkpoint(last_path)
        if ckpt.config != model_cfg.to_dict():
            raise ValueError("checkpoint config differs from the requested model config")
        model.load_state_dict(ckpt.params)
        if ckpt.optimizer is not None:
            optimizer.load_state_dict(ckpt.optimizer)
        rng.bit_generator.state = ckpt.extra["data_rng"]
        model.drop.rng.bit_generator.state = ckpt.extra["dropout_rng"]
        start_step = ckpt.step
        best_loss = float(ckpt.extra["best_val_loss"])
        best_step = int(ckpt.extra["best_step"])
        best_syllable_ppl = float(ckpt.extra["best_val_syllable_ppl"])
        elapsed_before = float(ckpt.extra["elapsed_s"])
        report(f"resumed from step {start_step}")

    report(
        f"model: {model.num_parameters():,} parameters, vocab {tokenizer.vocab_size}; "
        f"train {len(train_tokens):,} tokens ({split.train_lines} lines), "
        f"val {len(val_tokens):,} tokens ({split.val_lines} lines)"
    )
    log = CsvLogger(out / "log.csv", LOG_FIELDS, append=resume and start_step > 0)
    t0 = time.perf_counter() - elapsed_before
    window_loss, window_norm, window_steps, window_t = 0.0, 0.0, 0, time.perf_counter()
    step = start_step
    model.train()
    try:
        while step < config.steps:
            lr = warmup_cosine(
                step,
                base_lr=config.lr,
                min_lr=config.min_lr,
                warmup_steps=config.warmup_steps,
                total_steps=config.steps,
            )
            x, y = sampler.sample()
            model.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            norm = clip_grad_norm(model.parameters(), config.grad_clip)
            optimizer.step(lr)
            step += 1
            window_loss += loss.item()
            window_norm += norm
            window_steps += 1

            is_eval = step % config.eval_every == 0 or step == config.steps
            out_of_time = (
                config.max_minutes is not None
                and time.perf_counter() - t0 > config.max_minutes * 60.0
            )
            if step % config.log_every == 0 or is_eval or out_of_time:
                now = time.perf_counter()
                row: dict[str, object] = {
                    "step": step,
                    "elapsed_s": round(now - t0, 2),
                    "lr": lr,
                    "train_loss": window_loss / window_steps,
                    "grad_norm": window_norm / window_steps,
                    "tokens_per_s": round(
                        window_steps * config.batch_size * model_cfg.block_size / (now - window_t)
                    ),
                }
                if is_eval or out_of_time:
                    result = evaluate(model, val_tokens)
                    row.update(
                        val_loss=result.loss,
                        val_ppl=result.ppl,
                        val_syllable_ppl=result.syllable_ppl,
                    )
                    if result.loss < best_loss:
                        best_loss, best_step = result.loss, step
                        best_syllable_ppl = result.syllable_ppl
                        save_checkpoint(out / "best.npz", model=model, step=step)
                    _save_last(
                        last_path, model, optimizer, rng, step, now - t0,
                        best_loss, best_step, best_syllable_ppl,
                    )  # fmt: skip
                    report(
                        f"step {step}: train {row['train_loss']:.4f} val {result.loss:.4f} "
                        f"(syllable ppl {result.syllable_ppl:.1f}) {now - t0:.0f}s"
                    )
                log.log(row)
                window_loss, window_norm, window_steps, window_t = 0.0, 0.0, 0, now
            if out_of_time:
                report(f"stopping at step {step}: time budget of {config.max_minutes} min reached")
                break
    finally:
        log.close()
    return TrainResult(
        steps=step,
        best_step=best_step,
        best_val_loss=best_loss,
        best_val_syllable_ppl=best_syllable_ppl,
        elapsed_s=time.perf_counter() - t0,
    )


def _save_last(
    path: Path,
    model: LyraLM,
    optimizer: AdamW,
    rng: np.random.Generator,
    step: int,
    elapsed: float,
    best_loss: float,
    best_step: int,
    best_syllable_ppl: float,
) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=step,
        extra={
            "data_rng": rng.bit_generator.state,
            "dropout_rng": model.drop.rng.bit_generator.state,
            "elapsed_s": elapsed,
            "best_val_loss": best_loss,
            "best_step": best_step,
            "best_val_syllable_ppl": best_syllable_ppl,
        },
    )
