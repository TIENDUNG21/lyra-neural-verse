"""Print a Markdown table of training runs from their committed logs.

Usage::

    python scripts/summarize_runs.py runs/kieu-v1 runs/ablation-*

For each run directory it reads ``config.json`` and ``log.csv`` (written by
``lyra train``) and reports the evaluation step with the lowest validation
loss, which is the checkpoint ``lyra train`` keeps as ``best.npz``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyra.train.csvlog import read_log  # noqa: E402


def summarize(run: Path) -> dict[str, str]:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    rows = read_log(run / "log.csv")
    evals = [r for r in rows if r["val_loss"]]
    if not evals:
        raise ValueError(f"{run} has no evaluation rows")
    best = min(evals, key=lambda r: float(r["val_loss"]))
    model = config["model"]
    return {
        "run": run.name,
        "phonology": "yes" if model.get("phonology_embeddings") else "no",
        "slots": "yes" if model.get("slot_embeddings") else "no",
        "best step": best["step"],
        "val loss": f"{float(best['val_loss']):.3f}",
        "val ppl": f"{math.exp(float(best['val_loss'])):.0f}",
        "syllable ppl": f"{float(best['val_syllable_ppl']):.0f}",
        "train loss": f"{float(best['train_loss']):.3f}",
        "minutes": f"{float(rows[-1]['elapsed_s']) / 60:.1f}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    table = [summarize(run) for run in args.runs]
    headers = list(table[0])
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in table:
        print("| " + " | ".join(row[h] for h in headers) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
