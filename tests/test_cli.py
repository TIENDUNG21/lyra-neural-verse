from __future__ import annotations

import json
from pathlib import Path

import pytest

from lyra.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_check_valid_couplet_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["check", "Trăm năm trong cõi người ta / Chữ tài chữ mệnh khéo là ghét nhau"])
    out = capsys.readouterr().out
    assert code == 0
    assert "2 lines, 0 violations" in out
    assert "BBBTBB" in out


def test_check_reports_violations_and_exits_one(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["check", "Trăm năm trong ngày người ta", "Chữ tài chữ mệnh khéo lòng ghét nhau"])
    out = capsys.readouterr().out
    assert code == 1
    assert "line 1, syllable 4: tone" in out
    assert "line 2, syllable 6: rhyme" in out


def test_check_full_kieu_with_waivers(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "check",
            "--quiet",
            "--file",
            str(ROOT / "data" / "truyen-kieu.txt"),
            "--waivers",
            str(ROOT / "data" / "kieu-exceptions.tsv"),
        ]
    )
    assert code == 0
    assert "3254 lines, 0 violations, 60 waived" in capsys.readouterr().out


def test_check_empty_input_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", " / "]) == 2
    assert "no lines" in capsys.readouterr().err


def test_train_command_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = tmp_path / "c.txt"
    lines = (ROOT / "data" / "truyen-kieu.txt").read_text(encoding="utf-8").splitlines()[:80]
    corpus.write_text("\n".join(lines), encoding="utf-8")
    config = tmp_path / "cfg.json"
    config.write_text(
        json.dumps(
            {
                "model": {"block_size": 8, "d_model": 8, "n_layers": 1, "n_heads": 2},
                "train": {"steps": 4, "batch_size": 2, "warmup_steps": 1, "eval_every": 2,
                          "log_every": 2, "block_lines": 20, "val_every": 2},
            }
        ),
        encoding="utf-8",
    )  # fmt: skip
    code = main(["--threads", "1", "train", "--config", str(config), "--corpus", str(corpus),
                 "--out", str(tmp_path / "run")])  # fmt: skip
    assert code == 0
    assert "done: 4 steps" in capsys.readouterr().out
    assert (tmp_path / "run" / "best.npz").exists()


def test_unknown_command_exits_with_usage() -> None:
    with pytest.raises(SystemExit):
        main(["fly"])
