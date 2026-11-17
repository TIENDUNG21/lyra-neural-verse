"""Training checkpoints stored as ``.npz`` archives (no pickle).

Layout: ``param/<name>`` arrays, ``optim/<key>`` arrays and a ``meta`` JSON
string holding the step, the model config and free-form extras.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from lyra.nn.transformer import LyraLM
from lyra.optim.adamw import AdamW

FORMAT = "lyra-checkpoint/1"


@dataclass
class Checkpoint:
    step: int
    config: dict[str, Any]
    params: dict[str, NDArray[Any]]
    optimizer: dict[str, NDArray[Any]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(
    path: str | Path,
    *,
    model: LyraLM,
    step: int,
    optimizer: AdamW | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write model (and optionally optimizer) state atomically to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": FORMAT,
        "step": step,
        "config": model.config.to_dict(),
        "extra": extra or {},
    }
    arrays: dict[str, NDArray[Any]] = {"meta": np.array(json.dumps(meta))}
    for name, value in model.state_dict().items():
        arrays[f"param/{name}"] = value
    if optimizer is not None:
        for key, value in optimizer.state_dict().items():
            arrays[f"optim/{key}"] = value
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as handle:
        cast(Any, np.savez)(handle, **arrays)
    tmp.replace(target)


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Read a checkpoint written by ``save_checkpoint``."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if "meta" not in archive.files:
            raise ValueError(f"{path} is not a lyra checkpoint (no meta entry)")
        meta = json.loads(str(archive["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not a lyra checkpoint (format {meta.get('format')!r})")
        params = {k.removeprefix("param/"): archive[k] for k in archive.files if k[:6] == "param/"}
        optim = {k.removeprefix("optim/"): archive[k] for k in archive.files if k[:6] == "optim/"}
    return Checkpoint(
        step=int(meta["step"]),
        config=dict(meta["config"]),
        params=params,
        optimizer=optim or None,
        extra=dict(meta.get("extra", {})),
    )
