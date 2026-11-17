"""Append-only CSV metrics log."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path


class CsvLogger:
    """Writes one row per call and flushes, so a running job can be watched."""

    def __init__(self, path: str | Path, fields: Sequence[str], *, append: bool = False) -> None:
        self.path = Path(path)
        self.fields = list(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        resume = append and self.path.exists() and self.path.stat().st_size > 0
        if resume:
            with self.path.open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), [])
            if header != self.fields:
                raise ValueError(f"existing log {self.path} has different columns")
        self._handle = self.path.open("a" if resume else "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fields, lineterminator="\n")
        if not resume:
            self._writer.writeheader()
            self._handle.flush()

    def log(self, row: Mapping[str, object]) -> None:
        unknown = set(row) - set(self.fields)
        if unknown:
            raise KeyError(f"unknown log columns: {sorted(unknown)}")
        self._writer.writerow({k: _fmt(row.get(k)) for k in self.fields})
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> CsvLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def read_log(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
