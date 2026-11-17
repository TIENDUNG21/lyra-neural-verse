"""Minimal module system: parameters, nesting, train/eval mode and state dicts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Self

import numpy as np
from numpy.typing import DTypeLike, NDArray

from lyra.autograd import Tensor


class Parameter(Tensor):
    """A tensor that is a learnable leaf of the model."""

    __slots__ = ()

    def __init__(self, data: NDArray[Any]) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    """Base class: parameters and sub-modules are discovered from attributes.

    Attribute order (definition order in ``__init__``) fixes parameter order,
    so state dicts and optimizer moments line up across runs.
    """

    def __init__(self) -> None:
        self.training = True

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, value in vars(self).items():
            full = f"{prefix}{name}"
            if isinstance(value, Parameter):
                yield full, value
            elif isinstance(value, Module):
                yield from value.named_parameters(f"{full}.")

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.named_parameters()]

    def children(self) -> Iterator[Module]:
        for value in vars(self).values():
            if isinstance(value, Module):
                yield value

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def train(self, mode: bool = True) -> Self:
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> Self:
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def to(self, dtype: DTypeLike) -> Self:
        for p in self.parameters():
            p.data = p.data.astype(dtype)
        return self

    def state_dict(self) -> dict[str, NDArray[Any]]:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict[str, NDArray[Any]]) -> None:
        params = dict(self.named_parameters())
        missing = sorted(set(params) - set(state))
        unexpected = sorted(set(state) - set(params))
        if missing:
            raise KeyError(f"missing keys in state dict: {missing}")
        if unexpected:
            raise KeyError(f"unexpected keys in state dict: {unexpected}")
        for name, p in params.items():
            value = np.asarray(state[name])
            if value.shape != p.shape:
                raise ValueError(f"shape mismatch for {name}: {value.shape} vs {p.shape}")
            p.data = value.astype(p.dtype, copy=True)


class ModuleList(Module):
    """An indexable list of modules whose parameters are named ``<index>.<name>``."""

    def __init__(self, modules: Iterable[Module]) -> None:
        super().__init__()
        self._items = list(modules)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for index, module in enumerate(self._items):
            yield from module.named_parameters(f"{prefix}{index}.")

    def children(self) -> Iterator[Module]:
        yield from self._items

    def __iter__(self) -> Iterator[Module]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Module:
        return self._items[index]
