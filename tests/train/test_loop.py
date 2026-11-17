from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyra.optim import load_checkpoint
from lyra.train.csvlog import CsvLogger, read_log
from lyra.train.loop import LOG_FIELDS, TrainConfig, load_run_config, train

ROOT = Path(__file__).resolve().parents[2]
TINY_MODEL = {"block_size": 16, "d_model": 16, "n_layers": 1, "n_heads": 2, "dropout": 0.1}


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    lines = (ROOT / "data" / "truyen-kieu.txt").read_text(encoding="utf-8").splitlines()[:120]
    path = tmp_path / "tiny.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def tiny_config(**overrides: object) -> TrainConfig:
    base: dict[str, object] = {
        "steps": 30,
        "batch_size": 4,
        "lr": 3e-3,
        "warmup_steps": 5,
        "eval_every": 10,
        "log_every": 5,
        "block_lines": 20,
        "val_every": 3,
    }
    base.update(overrides)
    return TrainConfig.from_dict(base)


def test_training_lowers_loss_and_writes_outputs(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    messages: list[str] = []
    result = train(
        corpus_path=tiny_corpus,
        model_options=TINY_MODEL,
        config=tiny_config(),
        out_dir=out,
        report=messages.append,
    )
    rows = read_log(out / "log.csv")
    assert list(rows[0]) == list(LOG_FIELDS)
    assert [int(r["step"]) for r in rows] == [5, 10, 15, 20, 25, 30]
    assert float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"])
    evals = [r for r in rows if r["val_loss"]]
    assert [int(r["step"]) for r in evals] == [10, 20, 30]
    assert result.steps == 30
    assert result.best_step in (10, 20, 30)
    saved = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert saved["train"]["steps"] == 30
    assert saved["model"]["vocab_size"] > 100
    for name in ("best.npz", "last.npz", "tokenizer.json"):
        assert (out / name).exists()
    assert load_checkpoint(out / "best.npz").step == result.best_step
    assert any("parameters" in m for m in messages)


def test_resume_continues_the_step_count_and_log(tiny_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    train(
        corpus_path=tiny_corpus,
        model_options=TINY_MODEL,
        config=tiny_config(steps=20),
        out_dir=out,
        report=lambda _: None,
    )
    result = train(
        corpus_path=tiny_corpus,
        model_options=TINY_MODEL,
        config=tiny_config(steps=30),
        out_dir=out,
        resume=True,
        report=lambda _: None,
    )
    assert result.steps == 30
    steps = [int(r["step"]) for r in read_log(out / "log.csv")]
    assert steps == [5, 10, 15, 20, 25, 30]


def test_time_budget_stops_early(tiny_corpus: Path, tmp_path: Path) -> None:
    result = train(
        corpus_path=tiny_corpus,
        model_options=TINY_MODEL,
        config=tiny_config(steps=10_000, max_minutes=0.0),
        out_dir=tmp_path / "run",
        report=lambda _: None,
    )
    assert result.steps == 1
    assert result.best_step == 1


def test_run_config_rejects_vocab_size_and_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"model": {"vocab_size": 3}}), encoding="utf-8")
    with pytest.raises(ValueError, match="vocab_size"):
        load_run_config(path)
    path.write_text(json.dumps({"train": {"stepz": 3}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_run_config(path)


def test_csv_logger_appends_and_validates_columns(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    with CsvLogger(path, ["a", "b"]) as log:
        log.log({"a": 1, "b": 0.5})
        with pytest.raises(KeyError, match="unknown"):
            log.log({"c": 1})
    with CsvLogger(path, ["a", "b"], append=True) as log:
        log.log({"a": 2})
    assert read_log(path) == [{"a": "1", "b": "0.5"}, {"a": "2", "b": ""}]
    with pytest.raises(ValueError, match="columns"):
        CsvLogger(path, ["x"], append=True)
