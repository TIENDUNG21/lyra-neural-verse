"""mulberry32: a tiny 32-bit PRNG with a bit-exact TypeScript twin.

NumPy's generators cannot be reproduced in the browser, so sampling uses this
generator in both implementations: the same seed gives the same poem in the
CLI and in the playground.
"""

from __future__ import annotations

_MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """32-bit multiplication modulo 2**32 (JavaScript ``Math.imul`` bit pattern)."""
    return (a * b) & _MASK


class Mulberry32:
    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK

    def next_uint32(self) -> int:
        self.state = (self.state + 0x6D2B79F5) & _MASK
        t = self.state
        t = _imul(t ^ (t >> 15), t | 1)
        t = ((t + _imul(t ^ (t >> 7), t | 61)) & _MASK) ^ t
        return (t ^ (t >> 14)) & _MASK

    def random(self) -> float:
        """Uniform float in [0, 1) with 32 bits of randomness."""
        return self.next_uint32() / 4294967296.0
