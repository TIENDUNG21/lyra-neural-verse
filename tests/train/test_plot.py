from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from lyra.train.plot import nice_ticks, render_loss_svg

NS = "{http://www.w3.org/2000/svg}"


def rows() -> list[dict[str, str]]:
    out = []
    for step in range(10, 210, 10):
        row = {"step": str(step), "train_loss": f"{8.0 - step / 40:.3f}", "val_loss": ""}
        if step % 50 == 0:
            row["val_loss"] = f"{7.0 - step / 60 + (step / 100) ** 2:.3f}"
        out.append(row)
    return out


def test_nice_ticks_are_round_and_cover_the_range() -> None:
    assert nice_ticks(0, 3000, 6) == [0, 500, 1000, 1500, 2000, 2500, 3000]
    ticks = nice_ticks(3.37, 7.81)
    assert ticks[0] <= 3.37
    assert ticks[-1] >= 7.81
    assert ticks == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert nice_ticks(2.0, 2.0) == [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]


def test_nice_ticks_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        nice_ticks(0.0, float("nan"))


def test_svg_is_well_formed_with_two_series_legend_and_best_marker() -> None:
    svg = render_loss_svg(rows(), title="Loss & more", subtitle="test run")
    root = ET.fromstring(svg)
    lines = root.findall(f"{NS}polyline")
    assert [ln.get("data-series") for ln in lines] == ["train", "validation"]
    assert len(lines[0].get("points", "").split()) == 20
    assert len(lines[1].get("points", "").split()) == 4
    best = [c for c in root.findall(f"{NS}circle") if c.get("data-best")]
    assert len(best) == 1
    texts = [t.text or "" for t in root.iter(f"{NS}text")]
    assert "Loss & more" in texts
    assert any(t.startswith("best 6.33 at step 100") for t in texts)
    assert root.find(f"{NS}title") is not None


def test_empty_or_train_only_log_is_rejected() -> None:
    with pytest.raises(ValueError, match="val_loss"):
        render_loss_svg(
            [{"step": "1", "train_loss": "2.0", "val_loss": ""}], title="t", subtitle="s"
        )
