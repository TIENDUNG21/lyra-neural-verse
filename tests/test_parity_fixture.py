"""The committed browser parity fixture must match the committed model and code.

Logits are compared with a tolerance: they are rounded to 5 decimals, and BLAS
builds on different platforms can disagree in the last digit. Everything
discrete (decoded tokens, parses, checker verdicts) must match exactly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_parity_fixture.py"
FIXTURE = ROOT / "web" / "test" / "fixtures" / "parity.json"


def first_difference(a: list[Any], b: list[Any]) -> str:
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return f"index {index}: committed {x!r}, rebuilt {y!r}"
    return f"lengths differ: committed {len(a)}, rebuilt {len(b)}"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not generated yet")
def test_parity_fixture_is_up_to_date() -> None:
    spec = importlib.util.spec_from_file_location("make_parity_fixture", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rebuilt = json.loads(module.render(module.build_fixture()))
    committed = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert committed["model_sha256"] == rebuilt["model_sha256"], "run make_parity_fixture.py"
    for name, case in rebuilt["logits"].items():
        assert committed["logits"][name]["tokens"] == case["tokens"], name
        np.testing.assert_allclose(
            committed["logits"][name]["logits"], case["logits"], atol=1e-4, err_msg=name
        )
    for i, (old, new) in enumerate(
        zip(committed["generations"], rebuilt["generations"], strict=True)
    ):
        assert old["tokens"] == new["tokens"], (
            f"generation {i}: {first_difference(old['tokens'], new['tokens'])}"
        )
    for key in ("vocab_parses", "extra_words"):
        assert committed[key] == rebuilt[key], (
            f"{key}: {first_difference(committed[key], rebuilt[key])}"
        )
    for key in ("violations", "rhymes"):
        old, new = committed["kieu"][key], rebuilt["kieu"][key]
        assert old == new, f"kieu {key}: {first_difference(old, new)}"
