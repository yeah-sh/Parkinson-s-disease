"""The page's figures: the gauges (SVG), three Altair charts and the drivers (HTML bars).

Every figure draws values it is given; nothing here computes a feature or a prediction. The EEG
traces are thinned for the screen only (min-max per pixel column), which keeps every peak.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from neurolab.charts import thin
from pdeeg.config import Config
from restwave import plain
from restwave import theme as t
from restwave.blocks import image

alt.data_transformers.disable_max_rows()

GROUPS = ("Healthy group", "Parkinson's group")
GROUP_COLOURS = (t.HC, t.PD)
SHAPES = ("circle", "diamond")  # recording A, recording B
DASHES = ([1, 0], [7, 4])

# --- the gauge --------------------------------------------------------------------------------

_CX, _CY, _R, _WIDTH = 150.0, 148.0, 112.0, 22.0
_GAP = 0.012  # of the half turn, between zones: the surface gap


def _point(value: float, radius: float) -> tuple[float, float]:
    angle = math.pi * (1.0 - value)
    return _CX + radius * math.cos(angle), _CY - radius * math.sin(angle)


def _arc(start: float, stop: float) -> str:
    (x1, y1), (x2, y2) = _point(start, _R), _point(stop, _R)
    return f"M {x1:.2f} {y1:.2f} A {_R} {_R} 0 0 1 {x2:.2f} {y2:.2f}"


def gauge(
    value: float,
    threshold: float,
    band: Sequence[float],
    ends: tuple[str, str],
    score_of: str,
    tints: tuple[str, str] = (t.PAPER_2, t.PAPER_2),
) -> str:
    """A semicircle from 0 to 1 with the cut-off, the inconclusive zone and the score."""
    low, high = float(band[0]), float(band[1])
    zones = [(0.0, low, tints[0]), (low, high, "url(#hatch)"), (high, 1.0, tints[1])]
    arcs = []
    for start, stop, paint in zones:
        if stop - start <= 1e-9:
            continue
        a = start + (_GAP / 2 if start > 0 else 0.0)
        b = stop - (_GAP / 2 if stop < 1 else 0.0)
        arcs.append(
            f'<path d="{_arc(a, b)}" stroke="{paint}" stroke-width="{_WIDTH}" fill="none"/>'
        )
    tick_in, tick_out = (
        _point(threshold, _R - _WIDTH / 2 - 5),
        _point(threshold, _R + _WIDTH / 2 + 5),
    )
    tip = _point(value, _R - _WIDTH / 2 - 8)
    dot = _point(value, _R)
    band_words = (
        "every score is inconclusive"
        if low <= 0 and high >= 1
        else ("" if high - low <= 1e-9 else f"scores from {low:.2f} to {high:.2f} are inconclusive")
    )
    label = (
        f"{score_of} {value:.2f}, on a scale from 0 ({ends[0].lower()}) to 1 "
        f"({ends[1].lower()}). Cut-off {threshold:.2f}"
        + (f"; {band_words}" if band_words else "")
        + "."
    )
    caption = f"Cut-off {threshold:.2f}" + (f" · hatched: {band_words}" if band_words else "")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 160">
  <defs>
    <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse"
      patternTransform="rotate(45)">
      <rect width="7" height="7" fill="{t.SLATE_WASH}"/>
      <line x1="0" y1="0" x2="0" y2="7" stroke="{t.SLATE}" stroke-width="2.2"/>
    </pattern>
  </defs>
  {"".join(arcs)}
  <line x1="{tick_in[0]:.2f}" y1="{tick_in[1]:.2f}" x2="{tick_out[0]:.2f}" y2="{tick_out[1]:.2f}"
    stroke="{t.TEXT}" stroke-width="2.5"/>
  <line x1="{_CX}" y1="{_CY}" x2="{tip[0]:.2f}" y2="{tip[1]:.2f}" stroke="{t.TEXT}"
    stroke-width="3.5" stroke-linecap="round"/>
  <circle cx="{dot[0]:.2f}" cy="{dot[1]:.2f}" r="9" fill="{t.TEXT}" stroke="{t.PAPER}"
    stroke-width="3"/>
  <circle cx="{_CX}" cy="{_CY}" r="7" fill="{t.TEXT}"/>
</svg>"""
    return (
        f'<figure class="rw-gauge" role="img" aria-label="{escape(label)}">{image(svg)}'
        f'<div class="scale" aria-hidden="true"><span>{escape(ends[0])}</span>'
        f'<span class="value">{value:.2f}<small>{escape(score_of)}</small></span>'
        f"<span>{escape(ends[1])}</span></div>"
        f'<figcaption aria-hidden="true">{escape(caption)}</figcaption></figure>'
    )


# --- 1. the brain signal ----------------------------------------------------------------------

_SHORT = {
    "frontal": "front",
    "central": "top",
    "temporal": "side",
    "parietal": "upper back",
    "occipital": "back",
}


def channel_label(channel: str, config: Config) -> str:
    for region, channels in config.data.regions.items():
        if channel in channels:
            return f"{channel} · {_SHORT.get(region, region)}"
    return channel


def spacing(signal_uv: np.ndarray) -> float:
    """Microvolts between stacked traces: four typical amplitudes, rounded to a clean number."""
    typical = float(np.median(signal_uv.std(axis=1))) or 10.0
    raw = 4.0 * typical
    step = 10 ** math.floor(math.log10(raw))
    return float(min((m * step for m in (1, 2, 2.5, 5, 10) if m * step >= raw), default=raw))


def signal(
    signal_uv: np.ndarray, channels: Sequence[str], sfreq: float, start_s: float, config: Config
) -> alt.TopLevelMixin:
    """A few seconds of cleaned EEG, one trace per channel, stacked front of head to back."""
    gap = spacing(signal_uv)
    index, values = thin(signal_uv - signal_uv.mean(axis=1, keepdims=True), 640)
    n = len(channels)
    offsets = [(n - 1 - row) * gap for row in range(n)]
    frames = [
        pd.DataFrame(
            {
                "time": start_s + index[row] / sfreq,
                "y": values[row] + offsets[row],
                "channel": channels[row],
            }
        )
        for row in range(n)
    ]
    labels = {
        f"{offset:g}": channel_label(c, config) for offset, c in zip(offsets, channels, strict=True)
    }
    expression = f"{json.dumps(labels)}[format(datum.value, '')] || ''"
    chart = (
        alt.Chart(pd.concat(frames, ignore_index=True))
        .mark_line(strokeWidth=1.3, color=t.TEAL_DEEP, strokeJoin="round")
        .encode(
            x=alt.X(
                "time:Q",
                title="Seconds",
                scale=alt.Scale(nice=False, zero=False),
                axis=alt.Axis(grid=False, tickCount=6),
            ),
            y=alt.Y(
                "y:Q",
                title=None,
                scale=alt.Scale(domain=[-gap * 0.9, offsets[0] + gap * 0.9], nice=False),
                axis=alt.Axis(values=offsets, labelExpr=expression, grid=True, domain=False,
                              ticks=False, labelPadding=8, labelFontSize=12, minExtent=112),
            ),
            detail="channel:N",
        )
    )  # fmt: skip
    return t.finish(chart, 46 * n + 30)


# --- 2. brain-wave balance -------------------------------------------------------------------


def _percent(log_relative: float) -> float:
    return 100.0 * 10.0**log_relative


def balance(
    recordings: Sequence[tuple[str, Mapping[str, float]]],
    groups: pd.DataFrame | None,
    config: Config,
) -> alt.TopLevelMixin:
    """Each band's share of the power: this recording's dots over the groups' ranges.

    ``recordings`` is (label, band -> log10 relative power); ``groups`` has a row per group and
    band with ``mean``, ``low`` and ``high`` in the same unit.
    """
    bands = list(config.psd.bands)
    labels = {band: plain.band_label(band, config) for band in bands}
    order = [labels[b] for b in bands]
    scale = alt.Scale(type="log", nice=False, padding=14)
    axis = alt.Axis(values=[0.5, 1, 2, 5, 10, 20, 50, 100], labelExpr="datum.value + ' %'")

    def x(field: str) -> alt.X:
        return alt.X(field, title="Share of power (log scale)", scale=scale, axis=axis)

    y = alt.Y(
        "band:N",
        sort=order,
        title=None,
        axis=alt.Axis(
            labelFontSize=12.5,
            labelPadding=8,
            ticks=False,
            domain=False,
            labelLimit=170,
            minExtent=128,
        ),
    )
    layers = []
    if groups is not None:
        frame = groups.assign(
            band=groups["band"].map(labels),
            mean=groups["mean"].map(_percent),
            low=groups["low"].map(_percent),
            high=groups["high"].map(_percent),
        )
        frame["range"] = [
            f"{low:.1f}–{high:.1f} % (mean {mean:.1f} %)"
            for low, high, mean in zip(frame["low"], frame["high"], frame["mean"], strict=True)
        ]
        colour = alt.Color(
            "group:N", scale=alt.Scale(domain=list(GROUPS), range=list(GROUP_COLOURS)), legend=None
        )
        offset = alt.YOffset("group:N", sort=list(GROUPS), scale=alt.Scale(paddingInner=0.3))
        tooltip = [alt.Tooltip("group:N", title="Group"), alt.Tooltip("band:N", title="Band"),
                   alt.Tooltip("range:N", title="Typical range")]  # fmt: skip
        base = alt.Chart(frame).encode(y=y, yOffset=offset, color=colour, tooltip=tooltip)
        layers.append(
            base.mark_bar(height=9, cornerRadius=4.5, opacity=0.55).encode(
                x=x("low:Q"), x2="high:Q"
            )
        )
        layers.append(base.mark_tick(thickness=2.5, size=15).encode(x=x("mean:Q")))
    rows = [
        {"recording": label, "band": labels[band], "share": _percent(value),
         "value": f"{_percent(value):.1f} % of the power"}
        for label, values in recordings
        for band, value in values.items()
        if band in labels
    ]  # fmt: skip
    names = [label for label, _ in recordings]
    layers.append(
        alt.Chart(pd.DataFrame(rows))
        .mark_point(filled=True, size=150, color=t.TEXT, stroke=t.PAPER, strokeWidth=2, opacity=1)
        .encode(
            x=x("share:Q"),
            y=y,
            shape=alt.Shape(
                "recording:N",
                scale=alt.Scale(domain=names, range=list(SHAPES[: len(names)])),
                legend=None,
            ),
            tooltip=[alt.Tooltip("recording:N", title="Recording"),
                     alt.Tooltip("band:N", title="Band"), alt.Tooltip("value:N", title="Value")],
        )
    )  # fmt: skip
    return t.finish(alt.layer(*layers), 54 * len(bands) + 34)


# --- 3. signal complexity --------------------------------------------------------------------


def complexity(
    recordings: Sequence[tuple[str, pd.DataFrame]], groups: pd.DataFrame | None
) -> alt.TopLevelMixin:
    """f(alpha) of this recording (bold) over the groups' mean curves and alpha ranges.

    ``recordings`` is (label, frame with q, alpha, f_alpha); ``groups`` has, per group and q,
    ``alpha``, ``low``, ``high`` (the alpha range) and ``f_alpha``.
    """
    x_scale = alt.Scale(zero=False, nice=False, padding=10)

    def x(field: str) -> alt.X:
        return alt.X(field, title="α · local fluctuation strength", scale=x_scale, stack=None)

    y = alt.Y(
        "f_alpha:Q",
        title="f(α) · how common",
        scale=alt.Scale(zero=False, nice=False),
        axis=alt.Axis(minExtent=40),
    )
    colour = alt.Color(
        "group:N", scale=alt.Scale(domain=list(GROUPS), range=list(GROUP_COLOURS)), legend=None
    )
    layers = []
    if groups is not None:
        curves = groups.dropna(subset=["alpha", "f_alpha"])
        # The two branches of each curve (q above and below 0), each a horizontal band; the
        # point nearest q = 0, the top of the curve, belongs to both. No order channel here: on
        # an area it sets the stacking, and Vega-Lite then draws nothing.
        bands = curves.assign(branch=np.where(curves["q"] >= 0, "left", "right"))
        nearest = curves["q"].abs() == curves.groupby("group")["q"].transform(
            lambda q: q.abs().min()
        )
        top = curves[nearest].assign(
            branch=np.where(curves.loc[nearest, "q"] >= 0, "right", "left")
        )
        bands = pd.concat([bands, top], ignore_index=True)
        bands["key"] = bands["group"] + " " + bands["branch"]
        layers.append(
            alt.Chart(bands)
            .mark_area(orient="horizontal", opacity=0.14)
            .encode(y=y, x=x("low:Q"), x2="high:Q", color=colour, detail="key:N")
        )
        layers.append(
            alt.Chart(curves)
            .mark_line(strokeWidth=2)
            .encode(
                x=x("alpha:Q"),
                y=y,
                color=colour,
                detail="group:N",
                order=alt.Order("q:Q"),
                tooltip=[alt.Tooltip("group:N", title="Mean curve of the")],
            )
        )
    rows = pd.concat(
        [frame.assign(recording=label) for label, frame in recordings], ignore_index=True
    ).dropna(subset=["alpha", "f_alpha"])
    names = [label for label, _ in recordings]
    layers.append(
        alt.Chart(rows)
        .mark_line(strokeWidth=3.5, color=t.TEXT, strokeCap="round", strokeJoin="round")
        .encode(
            x=x("alpha:Q"),
            y=y,
            strokeDash=alt.StrokeDash(
                "recording:N",
                scale=alt.Scale(domain=names, range=list(DASHES[: len(names)])),
                legend=None,
            ),
            order=alt.Order("q:Q"),
            tooltip=[alt.Tooltip("recording:N", title="Recording")],
        )
    )
    return t.finish(alt.layer(*layers), 290)


# --- 4. what drove the decision -------------------------------------------------------------


def drivers(top: Mapping[str, Any], ends: tuple[str, str], config: Config) -> str:
    """The stage's top features as diverging bars: ``ends`` are (negative, positive) words."""
    features = top["features"]
    if not features:
        return '<p class="rw-missing">The model gave no features for this recording.</p>'
    if "contribution" not in features[0]:
        biggest = max(f["importance"] for f in features) or 1.0
        rows = "".join(
            f'<li><div class="name">{escape(plain.feature(f["feature"], config))}</div>'
            f'<div class="row"><div></div><div class="right"><div class="bar" '
            f'style="width:{100 * f["importance"] / biggest:.0f}%;background:{t.TEXT_3}"></div>'
            "</div></div></li>"
            for f in features
        )
        return (
            f'<ol class="rw-drivers">{rows}</ol><p class="rw-note">This model only says which '
            "measurements matter most overall, not which way they pushed this recording.</p>"
        )
    biggest = max(abs(f["contribution"]) for f in features) or 1.0
    rows = []
    for f in features:
        width = f"{100 * abs(f['contribution']) / biggest:.0f}%"
        towards_positive = f["contribution"] > 0
        bar = f'<div class="bar" style="width:{width}"></div>'
        left, right = ("", bar) if towards_positive else (bar, "")
        words = f"towards {ends[1] if towards_positive else ends[0]}"
        rows.append(
            f'<li><div class="name">{escape(plain.feature(f["feature"], config))}</div>'
            f'<div class="row" aria-hidden="true"><div class="left">{left}</div>'
            f'<div class="right">{right}</div></div>'
            f'<div class="dir{" towards-right" if towards_positive else ""}">{escape(words)}</div>'
            "</li>"
        )
    return (
        f'<ol class="rw-drivers">{"".join(rows)}</ol>'
        f'<div class="rw-axis" aria-hidden="true"><span>← towards {escape(ends[0])}</span>'
        f"<span>towards {escape(ends[1])} →</span></div>"
    )
