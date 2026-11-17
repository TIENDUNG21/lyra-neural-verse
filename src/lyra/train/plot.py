"""Render the training log as a self-contained SVG line chart.

No plotting library: the chart is computed here (scales, ticks, paths) and
written as plain SVG, so the README figure is produced by the project itself
from ``log.csv``.

Design: dark surface, hairline solid grid, 2 px lines, a legend plus direct
end labels for the two series, the best validation point marked, and native
``<title>`` tooltips on every evaluation point.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from xml.sax.saxutils import escape

SURFACE = "#0a0a0a"
GRID = "#262626"
TEXT_PRIMARY = "#ededed"
TEXT_SECONDARY = "#a3a3a3"
TRAIN_COLOR = "#3987e5"
VAL_COLOR = "#bf8a22"  # the project accent stepped into the dark-surface band

WIDTH, HEIGHT = 840, 440
LEFT, RIGHT, TOP, BOTTOM = 64, 120, 84, 52


@dataclass(frozen=True)
class Curve:
    name: str
    color: str
    points: tuple[tuple[float, float], ...]


def nice_ticks(low: float, high: float, target: int = 5) -> list[float]:
    """Round tick values covering ``[low, high]`` (steps of 1, 2 or 5 x 10^k)."""
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("tick range must be finite")
    if high <= low:
        high = low + 1.0
    raw = (high - low) / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)
    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    if ticks[-1] < high:
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def _fmt_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:g}"


def curves_from_log(rows: Sequence[Mapping[str, str]]) -> tuple[Curve, Curve]:
    train = tuple((float(r["step"]), float(r["train_loss"])) for r in rows if r.get("train_loss"))
    val = tuple((float(r["step"]), float(r["val_loss"])) for r in rows if r.get("val_loss"))
    if not train or not val:
        raise ValueError("the log needs train_loss and val_loss values to plot")
    return Curve("train", TRAIN_COLOR, train), Curve("validation", VAL_COLOR, val)


def render_loss_svg(rows: Sequence[Mapping[str, str]], *, title: str, subtitle: str) -> str:
    """Return SVG markup for the train/validation loss curves in ``rows``."""
    train, val = curves_from_log(rows)
    all_points = train.points + val.points
    x_max = max(p[0] for p in all_points)
    y_ticks = nice_ticks(min(p[1] for p in all_points), max(p[1] for p in all_points))
    x_ticks = nice_ticks(0.0, x_max, 6)
    y_lo, y_hi = y_ticks[0], y_ticks[-1]
    x_hi = x_ticks[-1]
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def sx(x: float) -> float:
        return LEFT + plot_w * x / x_hi

    def sy(y: float) -> float:
        return TOP + plot_h * (1.0 - (y - y_lo) / (y_hi - y_lo))

    out: list[str] = []
    add = out.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="t d" '
        'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">'
    )
    best = min(val.points, key=lambda p: p[1])
    add(f'<title id="t">{escape(title)}</title>')
    add(
        f'<desc id="d">{escape(subtitle)} Final train loss {train.points[-1][1]:.3f}; '
        f"best validation loss {best[1]:.3f} at step {best[0]:.0f}.</desc>"
    )
    add(f'<rect width="{WIDTH}" height="{HEIGHT}" rx="12" fill="{SURFACE}"/>')
    add(
        f'<text x="{LEFT}" y="34" fill="{TEXT_PRIMARY}" font-size="18" '
        f'font-weight="600">{escape(title)}</text>'
    )
    add(f'<text x="{LEFT}" y="56" fill="{TEXT_SECONDARY}" font-size="13">{escape(subtitle)}</text>')

    # Grid and axes: hairline, solid, recessive.
    for tick in y_ticks:
        y = sy(tick)
        add(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + plot_w}" y2="{y:.1f}" stroke="{GRID}"/>')
        add(
            f'<text x="{LEFT - 10}" y="{y + 4:.1f}" fill="{TEXT_SECONDARY}" font-size="12" '
            f'text-anchor="end">{_fmt_tick(tick)}</text>'
        )
    for tick in x_ticks:
        x = sx(tick)
        add(
            f'<text x="{x:.1f}" y="{TOP + plot_h + 22}" fill="{TEXT_SECONDARY}" font-size="12" '
            f'text-anchor="middle">{_fmt_tick(tick)}</text>'
        )
    add(
        f'<text x="{LEFT + plot_w / 2:.1f}" y="{HEIGHT - 10}" fill="{TEXT_SECONDARY}" '
        'font-size="12" text-anchor="middle">training step</text>'
    )
    add(
        f'<text transform="translate(18 {TOP + plot_h / 2:.1f}) rotate(-90)" '
        f'fill="{TEXT_SECONDARY}" font-size="12" text-anchor="middle">loss (nats / token)</text>'
    )

    for curve in (train, val):
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in curve.points)
        add(
            f'<polyline data-series="{curve.name}" points="{path}" fill="none" '
            f'stroke="{curve.color}" stroke-width="2" stroke-linejoin="round" '
            'stroke-linecap="round"/>'
        )
        end_x, end_y = curve.points[-1]
        add(
            f'<text x="{sx(end_x) + 10:.1f}" y="{sy(end_y) + 4:.1f}" fill="{TEXT_PRIMARY}" '
            f'font-size="12">{curve.name} {end_y:.2f}</text>'
        )

    # Native tooltips on every evaluation point (larger, invisible hit targets).
    for x, y in val.points:
        add(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="7" fill="transparent">'
            f"<title>step {x:.0f}: validation loss {y:.3f}</title></circle>"
        )
    bx, by = sx(best[0]), sy(best[1])
    add(
        f'<circle data-best="1" cx="{bx:.1f}" cy="{by:.1f}" r="5" fill="{VAL_COLOR}" '
        f'stroke="{SURFACE}" stroke-width="2"/>'
    )
    add(
        f'<text x="{bx:.1f}" y="{by + 22:.1f}" fill="{TEXT_PRIMARY}" font-size="12" '
        f'text-anchor="middle">best {best[1]:.2f} at step {best[0]:.0f}</text>'
    )

    # Legend, top right, text in text tokens beside a colored key.
    legend_x = WIDTH - RIGHT - 190
    for i, curve in enumerate((train, val)):
        lx = legend_x + i * 100
        add(
            f'<line x1="{lx}" y1="{TOP - 14}" x2="{lx + 18}" y2="{TOP - 14}" '
            f'stroke="{curve.color}" stroke-width="2" stroke-linecap="round"/>'
        )
        add(
            f'<text x="{lx + 24}" y="{TOP - 10}" fill="{TEXT_SECONDARY}" '
            f'font-size="12">{curve.name}</text>'
        )
    add("</svg>")
    return "\n".join(out) + "\n"
