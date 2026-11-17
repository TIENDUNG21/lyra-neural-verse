"""The release model file: one binary read by both the Python CLI and the browser.

Layout (little-endian)::

    b"LYRA" | u32 version | u32 header length | header JSON (UTF-8) | pad to 16
    tensor data, each tensor starting on a 16-byte boundary

The header holds the model config, the vocabulary, training metadata and a
tensor table. Matrices are stored as int8 with one float32 scale per output
channel (symmetric quantization: ``w = q * scale``); vectors (biases,
layer-norm parameters) stay float32. The format is simple enough to parse in
a few lines of TypeScript.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lyra.nn.transformer import ModelConfig

MAGIC = b"LYRA"
VERSION = 1
ALIGN = 16

FloatArray = NDArray[np.float32]


class ModelFileError(ValueError):
    """Raised for files that are not valid Lyra model files."""


@dataclass
class ModelFile:
    config: ModelConfig
    vocab: list[str]
    weights: dict[str, FloatArray]  # dequantized float32, keyed like LyraLM.state_dict()
    metadata: dict[str, Any]


def channel_axis(name: str) -> int:
    """Axis that indexes output channels: rows for embeddings, columns for Linear."""
    return 0 if name.endswith("emb.weight") else 1


def quantize_int8(weight: NDArray[Any], axis: int) -> tuple[NDArray[np.int8], FloatArray]:
    """Symmetric per-channel int8 quantization along ``axis`` of a 2-D matrix."""
    if weight.ndim != 2:
        raise ValueError("only matrices are quantized")
    reduce_axis = 1 - axis
    peak = np.abs(weight).max(axis=reduce_axis)
    scales = np.where(peak > 0, peak / 127.0, 1.0).astype(np.float32)
    expanded = np.expand_dims(scales, reduce_axis)
    q = np.clip(np.rint(weight / expanded), -127, 127).astype(np.int8)
    return q, scales


def dequantize_int8(q: NDArray[np.int8], scales: FloatArray, axis: int) -> FloatArray:
    return (q.astype(np.float32) * np.expand_dims(scales, 1 - axis)).astype(np.float32)


def _pad(buffer: bytearray) -> None:
    buffer.extend(b"\0" * (-len(buffer) % ALIGN))


def write_model_file(
    path: str | Path,
    *,
    params: dict[str, NDArray[Any]],
    config: ModelConfig,
    vocab: list[str],
    metadata: dict[str, Any] | None = None,
) -> int:
    """Quantize ``params`` and write the model file; returns its size in bytes."""
    if len(vocab) != config.vocab_size:
        raise ValueError("vocabulary size does not match the config")
    data = bytearray()
    table: list[dict[str, Any]] = []
    for name in sorted(params):
        array = np.asarray(params[name], dtype=np.float32)
        entry: dict[str, Any] = {"name": name, "shape": list(array.shape)}
        if array.ndim == 2:
            axis = channel_axis(name)
            q, scales = quantize_int8(array, axis)
            entry.update(dtype="int8", axis=axis, offset=len(data))
            data.extend(q.tobytes())
            _pad(data)
            entry["scale_offset"] = len(data)
            data.extend(scales.astype("<f4").tobytes())
        else:
            entry.update(dtype="float32", offset=len(data))
            data.extend(array.astype("<f4").tobytes())
        _pad(data)
        table.append(entry)
    header = {
        "format": "lyra-model/1",
        "config": config.to_dict(),
        "vocab": vocab,
        "tensors": table,
        "metadata": metadata or {},
    }
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    prefix = bytearray(MAGIC + struct.pack("<II", VERSION, len(header_bytes)) + header_bytes)
    _pad(prefix)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(prefix + data))
    return len(prefix) + len(data)


def read_model_file(path: str | Path) -> ModelFile:
    """Load and dequantize a model file written by ``write_model_file``."""
    raw = Path(path).read_bytes()
    if len(raw) < 12 or raw[:4] != MAGIC:
        raise ModelFileError(f"{path} is not a Lyra model file")
    version, header_len = struct.unpack("<II", raw[4:12])
    if version != VERSION:
        raise ModelFileError(f"unsupported model file version {version}")
    if 12 + header_len > len(raw):
        raise ModelFileError("truncated header")
    header = json.loads(raw[12 : 12 + header_len].decode("utf-8"))
    base = 12 + header_len
    base += -base % ALIGN
    weights: dict[str, FloatArray] = {}
    for entry in header["tensors"]:
        shape = tuple(entry["shape"])
        count = int(np.prod(shape))
        start = base + entry["offset"]
        if entry["dtype"] == "int8":
            if start + count > len(raw):
                raise ModelFileError(f"truncated tensor {entry['name']}")
            q = np.frombuffer(raw, dtype=np.int8, count=count, offset=start).reshape(shape)
            axis = int(entry["axis"])
            channels = shape[axis]
            scales = np.frombuffer(
                raw, dtype="<f4", count=channels, offset=base + entry["scale_offset"]
            ).astype(np.float32)
            weights[entry["name"]] = dequantize_int8(q, scales, axis)
        elif entry["dtype"] == "float32":
            if start + 4 * count > len(raw):
                raise ModelFileError(f"truncated tensor {entry['name']}")
            values = np.frombuffer(raw, dtype="<f4", count=count, offset=start)
            weights[entry["name"]] = values.astype(np.float32).reshape(shape)
        else:
            raise ModelFileError(f"unknown dtype {entry['dtype']!r}")
    return ModelFile(
        config=ModelConfig.from_dict(header["config"]),
        vocab=list(header["vocab"]),
        weights=weights,
        metadata=dict(header.get("metadata", {})),
    )
