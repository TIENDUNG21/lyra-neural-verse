from __future__ import annotations

import functools
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from lyra.nn import LyraLM, ModelConfig, Parameter
from lyra.optim import AdamW, clip_grad_norm, load_checkpoint, save_checkpoint, warmup_cosine


def test_adamw_first_step_matches_hand_computation() -> None:
    w = Parameter(np.array([[1.0, -2.0]], dtype=np.float32))
    b = Parameter(np.array([0.5], dtype=np.float32))
    opt = AdamW([w, b], lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.5)
    w.grad = np.array([[0.2, -0.4]], dtype=np.float32)
    b.grad = np.array([0.3], dtype=np.float32)
    opt.step()
    # Step 1: m_hat = g, v_hat = g^2, so the Adam update is lr * sign(g).
    expected_w = np.array([[1.0, -2.0]]) * (1 - 0.1 * 0.5) - 0.1 * np.sign([[0.2, -0.4]])
    np.testing.assert_allclose(w.data, expected_w, rtol=1e-6)
    # 1-D parameters are not decayed.
    np.testing.assert_allclose(b.data, [0.5 - 0.1], rtol=1e-6)
    assert opt.step_count == 1


def test_adamw_skips_parameters_without_grad_and_zero_grad_clears() -> None:
    w = Parameter(np.ones((2, 2), dtype=np.float32))
    opt = AdamW([w], lr=0.1)
    opt.step()
    np.testing.assert_array_equal(w.data, 1.0)
    w.grad = np.ones((2, 2), dtype=np.float32)
    opt.zero_grad()
    assert w.grad is None


def test_adamw_minimizes_a_quadratic() -> None:
    w = Parameter(np.array([[3.0, -4.0]], dtype=np.float32))
    opt = AdamW([w], lr=0.1, weight_decay=0.0)
    for _ in range(300):
        w.grad = 2.0 * w.data
        opt.step()
    assert np.abs(w.data).max() < 0.05


def test_adamw_state_dict_round_trip() -> None:
    w = Parameter(np.ones((2, 3), dtype=np.float32))
    opt = AdamW([w], lr=0.01)
    w.grad = np.full((2, 3), 0.5, dtype=np.float32)
    opt.step()
    other = AdamW([Parameter(w.data.copy())], lr=0.01)
    other.load_state_dict(opt.state_dict())
    assert other.step_count == 1
    np.testing.assert_array_equal(other.state_dict()["m.0"], opt.state_dict()["m.0"])


def test_warmup_cosine_schedule_shape() -> None:
    lr = functools.partial(warmup_cosine, base_lr=1.0, min_lr=0.1, warmup_steps=10, total_steps=110)
    assert lr(0) == pytest.approx(0.1)
    assert lr(9) == pytest.approx(1.0)
    assert lr(10) == pytest.approx(1.0)
    assert lr(60) == pytest.approx(0.55)
    assert lr(110) == pytest.approx(0.1)
    assert lr(500) == pytest.approx(0.1)
    decay = [lr(s) for s in range(10, 111)]
    assert all(a >= b for a, b in itertools.pairwise(decay))


def test_warmup_cosine_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="total_steps"):
        warmup_cosine(0, base_lr=1.0, min_lr=0.0, warmup_steps=10, total_steps=5)


def test_clip_grad_norm_scales_to_max_norm() -> None:
    a = Parameter(np.zeros(3, dtype=np.float32))
    b = Parameter(np.zeros((2, 2), dtype=np.float32))
    a.grad = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    b.grad = np.array([[0.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    norm = clip_grad_norm([a, b], max_norm=1.0)
    assert norm == pytest.approx(5.0)
    total = math.sqrt(float((a.grad**2).sum() + (b.grad**2).sum()))
    assert total == pytest.approx(1.0, rel=1e-5)


def test_clip_grad_norm_leaves_small_gradients_alone() -> None:
    a = Parameter(np.zeros(2, dtype=np.float32))
    a.grad = np.array([0.3, 0.4], dtype=np.float32)
    assert clip_grad_norm([a], max_norm=1.0) == pytest.approx(0.5)
    np.testing.assert_allclose(a.grad, [0.3, 0.4])


def _train_steps(model: LyraLM, opt: AdamW, data: np.ndarray, steps: int) -> None:
    for _ in range(steps):
        model.zero_grad()
        model.loss(data[:, :-1], data[:, 1:]).backward()
        opt.step()


def test_checkpoint_resume_is_bit_identical(tmp_path: Path) -> None:
    cfg = ModelConfig(vocab_size=9, block_size=6, d_model=8, n_layers=1, n_heads=2, dropout=0.0)
    data = np.random.default_rng(0).integers(0, 9, size=(2, 7))

    straight = LyraLM(cfg)
    opt = AdamW(straight.parameters(), lr=0.01)
    _train_steps(straight, opt, data, 6)

    first = LyraLM(cfg)
    opt_first = AdamW(first.parameters(), lr=0.01)
    _train_steps(first, opt_first, data, 3)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model=first, optimizer=opt_first, step=3, extra={"note": "x"})

    ckpt = load_checkpoint(path)
    assert ckpt.step == 3
    assert ckpt.extra == {"note": "x"}
    assert ModelConfig.from_dict(ckpt.config) == cfg
    resumed = LyraLM(cfg)
    resumed.load_state_dict(ckpt.params)
    opt_resumed = AdamW(resumed.parameters(), lr=0.01)
    assert ckpt.optimizer is not None
    opt_resumed.load_state_dict(ckpt.optimizer)
    _train_steps(resumed, opt_resumed, data, 3)

    for (name, p), q in zip(straight.named_parameters(), resumed.parameters(), strict=True):
        np.testing.assert_array_equal(p.data, q.data, err_msg=name)


def test_checkpoint_without_optimizer_and_bad_file(tmp_path: Path) -> None:
    model = LyraLM(ModelConfig(vocab_size=5, block_size=4, d_model=4, n_layers=1, n_heads=1))
    path = tmp_path / "weights.npz"
    save_checkpoint(path, model=model, step=0)
    assert load_checkpoint(path).optimizer is None
    bad = tmp_path / "bad.npz"
    np.savez(bad, something=np.zeros(1))
    with pytest.raises(ValueError, match="not a lyra checkpoint"):
        load_checkpoint(bad)
