"""The committed browser parity fixture must match the committed model and code."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_parity_fixture.py"
FIXTURE = ROOT / "web" / "test" / "fixtures" / "parity.json"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not generated yet")
def test_parity_fixture_is_up_to_date() -> None:
    spec = importlib.util.spec_from_file_location("make_parity_fixture", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads(module.render(module.build_fixture()))
    committed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert committed["model_sha256"] == expected["model_sha256"], "re-run make_parity_fixture.py"
    assert committed == expected
