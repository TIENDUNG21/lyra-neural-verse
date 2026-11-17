"""End-to-end sanity check: the whole stack can memorize a short sequence."""

from __future__ import annotations

import numpy as np

from lyra.nn import LyraLM, ModelConfig
from lyra.optim import AdamW, clip_grad_norm, warmup_cosine


def test_tiny_model_memorizes_a_sequence() -> None:
    cfg = ModelConfig(vocab_size=12, block_size=16, d_model=32, n_layers=2, n_heads=4, dropout=0.0)
    model = LyraLM(cfg)
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    seq = np.array([[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2]])
    x, y = seq[:, :-1], seq[:, 1:]
    first = model.loss(x, y).item()
    for step in range(200):
        model.zero_grad()
        loss = model.loss(x, y)
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step(warmup_cosine(step, base_lr=3e-3, min_lr=3e-4, warmup_steps=10, total_steps=200))
    model.eval()
    final = model.loss(x, y).item()
    assert first > 2.0
    assert final < 0.1
    predictions = model.forward(x).data.argmax(axis=-1)
    np.testing.assert_array_equal(predictions, y)
