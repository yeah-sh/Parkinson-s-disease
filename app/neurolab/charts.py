"""Interactive charts of NeuroLab (Altair, which Streamlit ships with).

Every chart draws values it is given; nothing here computes a feature or a prediction. The
EEG traces are thinned for the screen only (min-max per pixel column), which keeps every peak.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import altair as alt
import numpy as np
import pandas as pd

from neurolab import theme as t

alt.data_transformers.disable_max_rows()

_FONT = "Source Sans 3, Source Sans Pro, Segoe UI, sans-serif"
_POINTS_PER_TRACE = 700  # min-max pairs: about one per pixel column of a wide screen


def finish(chart: alt.TopLevelMixin, height: int | None = None) -> alt.TopLevelMixin:
    """The NeuroLab look: transparent on the card, hairline grid, quiet axes."""
    if height is not None:
        chart = chart.properties(height=height)
    return (
        chart.configure(
            background="transparent",
            font=_FONT,
            padding={"left": 2, "right": 10, "top": 6, "bottom": 2},
        )  # fmt: skip
        .configure_view(stroke=None)
        .configure_axis(
            domainColor=t.AXIS,
            gridColor=t.GRID,
            tickColor=t.AXIS,
            labelColor=t.TEXT_2,
            titleColor=t.TEXT_3,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight=500,
            titlePadding=8,
            gridWidth=1,
            domainWidth=1,
            tickSize=4,
            labelFont=_FONT,
        )
        .configure_legend(
            labelColor=t.TEXT_2,
            titleColor=t.TEXT_3,
            labelFontSize=11,
            titleFontSize=11,
            orient="top",
            direction="horizontal",
            symbolStrokeWidth=2,
        )
        .configure_title(color=t.TEXT, fontSize=13, fontWeight=600, anchor="start")
    )


def _tooltip_format() -> dict[str, Any]:
    return {}


# --- the EEG ---------------------------------------------------------------------------------


def thin(x: np.ndarray, points: int = _POINTS_PER_TRACE) -> tuple[np.ndarray, np.ndarray]:
    """Sample indices and values of ``x`` (channels x samples) thinned to about ``points``
    per channel, keeping each stretch's minimum and maximum in time order."""
    n = x.shape[1]
    if n <= points:
        index = np.broadcast_to(np.arange(n), x.shape)
        return index, x
    width = int(np.ceil(n / (points // 2)))
    n_buckets = n // width
    body = x[:, : n_buckets * width].reshape(x.shape[0], n_buckets, width)
    low, high = body.argmin(axis=2), body.argmax(axis=2)
    first, second = np.minimum(low, high), np.maximum(low, high)
    offsets = np.arange(n_buckets) * width
    index = np.stack([first + offsets, second + offsets], axis=2).reshape(x.shape[0], -1)
    return index, np.take_along_axis(x, index, axis=1)


def waveform(
    traces: Mapping[str, np.ndarray],
    channels: Sequence[str],
    sfreq: float,
    start: float,
    spacing: float,
    bad: Sequence[Mapping[str, Any]],
    colours: Mapping[str, str],
) -> alt.TopLevelMixin:
    """Stacked EEG traces of ``channels`` from ``start``, one row per channel.

    ``traces`` maps a label ("Cleaned", "Before cleaning") to channels x samples in µV over
    the window shown; every row is ``spacing`` µV tall. ``bad`` are the bad stretches
    (onset_s, duration_s, description) of the whole window, drawn where they overlap.
    """
    n = len(channels)
    rows = []
    for label, data in traces.items():
        index, values = thin(data)
        for i, channel in enumerate(channels):
            row = n - 1 - i
            rows.append(
                pd.DataFrame(
                    {
                        "channel": channel,
                        "t": start + index[i] / sfreq,
                        "uv": values[i],
                        "y": row + values[i] / spacing,
                        "trace": label,
                    }
                )
            )
    frame = pd.concat(rows, ignore_index=True)
    end = start + next(iter(traces.values())).shape[1] / sfreq
    names = json.dumps(list(reversed(channels)))
    x = alt.X(
        "t:Q",
        title="Time in the analysed window (s)",
        scale=alt.Scale(domain=[start, end], nice=False, zero=False),  # fmt: skip
    )
    y_scale = alt.Scale(domain=[-0.9, n - 0.1], nice=False, zero=False)
    layers = []
    shaded = [
        {
            "from": max(s["onset_s"], start),
            "to": min(s["onset_s"] + s["duration_s"], end),
            "kind": _BAD_LABELS.get(s["description"], s["description"]),
        }
        for s in bad
        if s["onset_s"] < end and s["onset_s"] + s["duration_s"] > start
    ]
    if shaded:
        layers.append(
            alt.Chart(pd.DataFrame(shaded))
            .mark_rect(color=t.AMBER, opacity=0.13)
            .encode(
                x=alt.X("from:Q", scale=alt.Scale(domain=[start, end], nice=False, zero=False)),
                x2="to:Q",
                tooltip=[alt.Tooltip("kind:N", title="Marked bad")],
            )
        )
    domain = list(traces)
    layers.append(
        alt.Chart(frame)
        .mark_line(strokeWidth=1, strokeJoin="round", clip=True)
        .encode(
            x=x,
            y=alt.Y(
                "y:Q",
                scale=y_scale,
                title=None,
                axis=alt.Axis(
                    values=list(range(n)),
                    labelExpr=f"{names}[datum.value]",
                    grid=False,
                    domain=False,
                    ticks=False,
                    labelPadding=6,
                    labelFontSize=11 if n <= 16 else 10,
                ),
            ),
            color=alt.Color(
                "trace:N",
                scale=alt.Scale(domain=domain, range=[colours[d] for d in domain]),
                legend=None,
            ),
            detail="channel:N",
        )
    )
    main = domain[-1]
    layers.append(
        alt.Chart(frame[frame["trace"] == main])
        .mark_point(opacity=0, size=40)
        .encode(
            x=x,
            y=alt.Y("y:Q", scale=y_scale),
            tooltip=[
                alt.Tooltip("channel:N", title="Channel"),
                alt.Tooltip("t:Q", title="Time (s)", format=".3f"),
                alt.Tooltip("uv:Q", title=f"{main} (µV)", format=".1f"),
            ],
        )
    )
    zoomable = alt.layer(*layers).interactive(bind_y=False)
    return finish(zoomable, height=max(220, min(26 * n + 40, 760)))


_BAD_LABELS = {
    "BAD_peak": "Peak-to-peak amplitude too large",
    "BAD_flat": "Flat (amplitude too small)",
}


def bad_track(
    bad: Sequence[Mapping[str, Any]], total: float, window: tuple[float, float]
) -> alt.TopLevelMixin:
    """The whole analysed window as a strip: bad stretches, and the part on screen."""
    base = pd.DataFrame({"from": [0.0], "to": [total]})
    x = alt.X(
        "from:Q", title="Time in the analysed window (s)",
        scale=alt.Scale(domain=[0, total], nice=False, zero=True),
    )  # fmt: skip
    layers = [alt.Chart(base).mark_rect(color=t.SURFACE_3, cornerRadius=3).encode(x=x, x2="to:Q")]
    if bad:
        frame = pd.DataFrame(
            {
                "from": [s["onset_s"] for s in bad],
                "to": [s["onset_s"] + s["duration_s"] for s in bad],
                "kind": [_BAD_LABELS.get(s["description"], s["description"]) for s in bad],
                "seconds": [s["duration_s"] for s in bad],
            }
        )
        layers.append(
            alt.Chart(frame)
            .mark_rect(color=t.AMBER, opacity=0.85)
            .encode(
                x=x,
                x2="to:Q",
                tooltip=[
                    alt.Tooltip("kind:N", title="Marked bad"),
                    alt.Tooltip("from:Q", title="From (s)", format=".1f"),
                    alt.Tooltip("seconds:Q", title="Length (s)", format=".1f"),
                ],
            )
        )
    shown = pd.DataFrame({"from": [window[0]], "to": [window[1]]})
    layers.append(
        alt.Chart(shown)
        .mark_rect(fill=t.CYAN, fillOpacity=0.1, stroke=t.CYAN, strokeWidth=1.5, cornerRadius=2)
        .encode(x=x, x2="to:Q", tooltip=[alt.Tooltip("from:Q", title="On screen from (s)")])
    )
    return finish(alt.layer(*layers), height=34)


def spectrum_before_after(
    freqs: np.ndarray,
    before_db: np.ndarray,
    after_db: np.ndarray,
    band: tuple[float, float],
    mains: float | None,
) -> alt.TopLevelMixin:
    """Median spectrum over channels before and after cleaning, from the cleaning log."""
    keep = freqs >= band[0]
    wide = pd.DataFrame(
        {"freq": freqs[keep], "Before cleaning": before_db[keep], "Cleaned": after_db[keep]}
    )
    long = wide.melt("freq", var_name="trace", value_name="db")
    low = float(before_db[keep].min()) - 15
    high = float(max(before_db[keep].max(), after_db[keep].max())) + 5
    x = alt.X("freq:Q", title="Frequency (Hz)", scale=alt.Scale(domain=[0, float(freqs.max())]))
    y = alt.Y("db:Q", title="Power (dB re 1 µV²/Hz)", scale=alt.Scale(domain=[low, high]))
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["freq"], empty=False)
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2, clip=True)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                "trace:N",
                scale=alt.Scale(
                    domain=["Before cleaning", "Cleaned"], range=[t.CONTEXT, t.SERIES[0]]
                ),
                legend=alt.Legend(title=None),
            ),
        )
    )
    marks = pd.DataFrame(
        [
            {"freq": band[0], "label": f"{band[0]:g} Hz"},
            {"freq": band[1], "label": f"{band[1]:g} Hz"},
        ]
        + ([{"freq": mains, "label": f"mains {mains:g} Hz"}] if mains else [])
    )
    rules = (
        alt.Chart(marks).mark_rule(color=t.TEXT_3, strokeWidth=1, opacity=0.6).encode(x="freq:Q")
    )
    labels = (
        alt.Chart(marks)
        .mark_text(align="left", dx=4, dy=-4, baseline="top", color=t.TEXT_3, fontSize=10)
        .encode(x="freq:Q", y=alt.value(4), text="label:N")
    )
    points = (
        alt.Chart(wide)
        .mark_rule(color=t.TEXT_2, strokeWidth=1)
        .encode(
            x="freq:Q",
            opacity=alt.condition(hover, alt.value(0.5), alt.value(0)),
            tooltip=[
                alt.Tooltip("freq:Q", title="Frequency (Hz)", format=".2f"),
                alt.Tooltip("Before cleaning:Q", format=".1f", title="Before (dB)"),
                alt.Tooltip("Cleaned:Q", format=".1f", title="Cleaned (dB)"),
            ],
        )
        .add_params(hover)
    )
    return finish(alt.layer(rules, labels, lines, points), height=300)


def channel_columns(
    frame: pd.DataFrame, value: str, title: str, fmt: str, colour: str = t.SERIES[0]
) -> alt.TopLevelMixin:
    """One column per channel, in montage order (``frame`` has ``channel`` and ``value``)."""
    order = list(frame["channel"])
    return finish(
        alt.Chart(frame)
        .mark_bar(color=colour, cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=12)
        .encode(
            x=alt.X("channel:N", sort=order, title=None, axis=alt.Axis(labelAngle=-60)),
            y=alt.Y(f"{value}:Q", title=title),
            tooltip=[
                alt.Tooltip("channel:N", title="Channel"),
                alt.Tooltip(f"{value}:Q", title=title, format=fmt),
            ],
        ),
        height=210,
    )


# --- MFDFA -----------------------------------------------------------------------------------


def q_colours(qs: Sequence[float]) -> list[str]:
    n = len(qs)
    ramp = t.Q_RAMP
    if n <= len(ramp):
        return list(ramp[len(ramp) - n :])
    return [ramp[round(i * (len(ramp) - 1) / (n - 1))] for i in range(n)]


def fluctuations(
    scales: np.ndarray,
    qs: np.ndarray,
    fq: np.ndarray,
    ranges: Mapping[str, tuple[float, float]],
) -> alt.TopLevelMixin:
    """log Fq(s) against log s, one line per q, with the configured fit ranges shaded."""
    frame = pd.DataFrame(
        [
            {"q": f"q = {q:g}", "order": float(q), "s": float(s), "fq": float(value)}
            for q, row in zip(qs, fq, strict=True)
            for s, value in zip(scales, row, strict=True)
            if np.isfinite(value) and value > 0
        ]
    )
    labels = [f"q = {q:g}" for q in qs]
    x = alt.X("s:Q", title="Window size s (seconds, log)", scale=alt.Scale(type="log"))
    y = alt.Y("fq:Q", title="Fq(s) (log)", scale=alt.Scale(type="log"))
    fit = pd.DataFrame(
        [{"range": name, "low": low, "high": high} for name, (low, high) in ranges.items()]
    )
    shading = (
        alt.Chart(fit)
        .mark_rect(color=t.TEXT, opacity=0.05)
        .encode(
            x=alt.X("low:Q", scale=alt.Scale(type="log")),
            x2="high:Q",
            tooltip=[
                alt.Tooltip("range:N", title="Fit range"),
                alt.Tooltip("low:Q", title="From (s)"),
                alt.Tooltip("high:Q", title="To (s)"),
            ],
        )
    )
    names = (
        alt.Chart(fit)
        .transform_calculate(mid="sqrt(datum.low * datum.high)")
        .mark_text(baseline="top", dy=4, color=t.TEXT_3, fontSize=11, fontWeight=600)
        .encode(x=alt.X("mid:Q", scale=alt.Scale(type="log")), y=alt.value(0), text="range:N")
    )
    lines = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                "q:N",
                sort=labels,
                scale=alt.Scale(domain=labels, range=q_colours(qs)),
                legend=alt.Legend(title="Moment order"),
            ),
            order="s:Q",
        )
    )
    hover = (
        alt.Chart(frame)
        .mark_point(opacity=0, size=60)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("q:N", title="Order"),
                alt.Tooltip("s:Q", title="Window (s)", format=".3g"),
                alt.Tooltip("fq:Q", title="Fq(s)", format=".4g"),
            ],
        )
    )
    return finish(alt.layer(shading, names, lines, hover), height=360)


def h_of_q(curves: pd.DataFrame, greyed: bool = False) -> alt.TopLevelMixin:
    colour = t.CONTEXT if greyed else t.SERIES[0]
    base = alt.Chart(curves.dropna(subset=["h"])).encode(
        x=alt.X("q:Q", title="Moment order q"),
        y=alt.Y("h:Q", title="h(q)", scale=alt.Scale(zero=False)),
    )
    return finish(
        alt.layer(
            base.mark_line(color=colour, strokeWidth=2),
            base.mark_point(
                filled=True, color=colour, size=36, stroke=t.SURFACE, strokeWidth=1.5
            ).encode(tooltip=[alt.Tooltip("q:Q"), alt.Tooltip("h:Q", title="h(q)", format=".3f")]),
        ),
        height=280,
    )


def singularity(curves: pd.DataFrame, alpha0: float | None) -> alt.TopLevelMixin:
    """f(alpha) with its peak alpha0 marked."""
    frame = curves.dropna(subset=["alpha", "f_alpha"])
    base = alt.Chart(frame).encode(
        x=alt.X("alpha:Q", title="Singularity exponent α", scale=alt.Scale(zero=False)),
        y=alt.Y("f_alpha:Q", title="f(α)", scale=alt.Scale(zero=False)),
        order="q:Q",
    )
    layers = [
        base.mark_line(color=t.SERIES[0], strokeWidth=2),
        base.mark_point(
            filled=True, color=t.SERIES[0], size=40, stroke=t.SURFACE, strokeWidth=1.5
        ).encode(
            tooltip=[
                alt.Tooltip("q:Q"),
                alt.Tooltip("alpha:Q", title="α", format=".3f"),
                alt.Tooltip("f_alpha:Q", title="f(α)", format=".3f"),
            ]
        ),
    ]
    if alpha0 is not None and np.isfinite(alpha0):
        mark = pd.DataFrame({"alpha": [alpha0], "label": [f"α₀ = {alpha0:.3f}"]})
        layers.append(alt.Chart(mark).mark_rule(color=t.TEXT_3, strokeWidth=1).encode(x="alpha:Q"))
        layers.append(
            alt.Chart(mark)
            .mark_text(align="left", dx=5, baseline="top", color=t.TEXT_2, fontSize=11)
            .encode(x="alpha:Q", y=alt.value(4), text="label:N")
        )
    return finish(alt.layer(*layers), height=280)


def ranking(frame: pd.DataFrame, label: str, signed: bool, fmt: str = ".3f") -> alt.TopLevelMixin:
    """Channels sorted by value (``frame``: ``channel``, ``value``); signed values in two hues."""
    data = frame.dropna(subset=["value"]).copy()
    data["side"] = np.where(data["value"] >= 0, "above 0", "below 0")
    colour = (
        alt.Color(
            "side:N",
            scale=alt.Scale(domain=["above 0", "below 0"], range=[t.SERIES[0], t.SERIES[1]]),
            legend=alt.Legend(title=None),
        )
        if signed
        else alt.value(t.SERIES[0])
    )
    bars = (
        alt.Chart(data)
        .mark_bar(size=10, cornerRadiusEnd=3)
        .encode(
            y=alt.Y("channel:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=10)),
            x=alt.X(
                "value:Q",
                title=label,
                scale=alt.Scale(zero=signed or bool((data["value"] >= 0).all())),
            ),
            color=colour,
            tooltip=[
                alt.Tooltip("channel:N", title="Channel"),
                alt.Tooltip("value:Q", title=label, format=fmt),
            ],
        )
    )
    return finish(bars, height=max(240, 15 * len(data)))


# --- the models ---------------------------------------------------------------------------


def probability(
    oof: pd.DataFrame,
    classes: tuple[str, str],
    threshold: float,
    band: Sequence[float],
    this: float | None,
    *,
    greyed: bool = False,
    this_label: str = "This recording",
) -> alt.TopLevelMixin:
    """Out-of-fold probabilities of the training recordings, one row per true class, with the
    threshold, the uncertain band, and this recording's probability.

    ``oof`` has ``probability`` and ``label`` (1 for ``classes[1]``) and ``fold``.
    """
    negative, positive = classes
    rng = np.random.default_rng(0)  # the same jitter every time
    frame = pd.DataFrame(
        {
            "probability": oof["probability"].to_numpy(dtype=float),
            "row": oof["label"].to_numpy(dtype=int),
            "fold": oof["fold"].to_numpy(dtype=int),
        }
    )
    frame["y"] = frame["row"] + rng.uniform(-0.26, 0.26, len(frame))
    frame["truth"] = np.where(frame["row"] == 1, positive, negative)
    rows = json.dumps([negative, positive])
    x = alt.X(
        "probability:Q",
        title=f"Model score: P({positive})",
        scale=alt.Scale(domain=[0, 1], nice=False),
        axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".2f"),
    )
    y = alt.Y(
        "y:Q",
        title=None,
        scale=alt.Scale(domain=[-0.6, 1.95], nice=False),
        axis=alt.Axis(
            values=[0, 1],
            labelExpr=f"{rows}[datum.value]",
            grid=False,
            ticks=False,
            domain=False,
            labelPadding=6,
        ),  # fmt: skip
    )
    low, high = band
    layers = []
    if high > low:
        span = pd.DataFrame({"from": [low], "to": [high], "label": ["no label (uncertain)"]})
        layers.append(
            alt.Chart(span)
            .mark_rect(color=t.AMBER, opacity=0.1)
            .encode(
                x=alt.X("from:Q", scale=alt.Scale(domain=[0, 1], nice=False)),
                x2="to:Q",
                tooltip=[
                    alt.Tooltip("label:N", title="Band"),
                    alt.Tooltip("from:Q", title="From"),
                    alt.Tooltip("to:Q", title="To"),
                ],
            )
        )
        layers.append(
            alt.Chart(span)
            .transform_calculate(mid="(datum.from + datum.to) / 2")
            .mark_text(color=t.AMBER, fontSize=10, baseline="top", dy=2, opacity=0.9)
            .encode(x="mid:Q", y=alt.value(0), text="label:N")
        )
    rule = pd.DataFrame({"x": [threshold], "label": [f"threshold {threshold:g}"]})
    layers.append(alt.Chart(rule).mark_rule(color=t.TEXT_3, strokeWidth=1).encode(x="x:Q"))
    colours = [t.CONTEXT, t.CONTEXT] if greyed else [t.SERIES[1], t.SERIES[0]]
    layers.append(
        alt.Chart(frame)
        .mark_circle(size=58, opacity=0.85, stroke=t.SURFACE, strokeWidth=1)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                "truth:N", scale=alt.Scale(domain=[negative, positive], range=colours), legend=None
            ),
            tooltip=[
                alt.Tooltip("truth:N", title="True class"),
                alt.Tooltip("probability:Q", title="Out-of-fold score", format=".3f"),
                alt.Tooltip("fold:Q", title="Fold"),
            ],
        )
    )
    if this is not None:
        mark = pd.DataFrame({"x": [this], "label": [f"{this_label} · {this:.2f}"]})
        colour = t.TEXT_2 if greyed else t.CYAN
        layers.append(alt.Chart(mark).mark_rule(color=colour, strokeWidth=2).encode(x="x:Q"))
        align = "right" if this > 0.7 else "left"
        layers.append(
            alt.Chart(mark)
            .mark_text(
                color=colour,
                fontSize=11,
                fontWeight=600,
                align=align,
                dx=-6 if align == "right" else 6,
                baseline="bottom",
                dy=-2,
            )  # fmt: skip
            .encode(x="x:Q", y=alt.value(196), text="label:N")
        )
    return finish(alt.layer(*layers), height=200)


METRIC_NAMES = {
    "balanced_accuracy": "Balanced accuracy",
    "roc_auc": "ROC AUC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
}


def forest(metrics: Mapping[str, Mapping[str, float]], chance: float, greyed: bool = False):
    """Each metric with its 95 % interval, against chance."""
    frame = pd.DataFrame(
        [
            {
                "metric": METRIC_NAMES[name],
                "value": m["value"],
                "low": m["low"],
                "high": m["high"],
                "p": m.get("p"),
            }
            for name, m in metrics.items()
        ]
    )
    order = list(frame["metric"])
    colour = t.CONTEXT if greyed else t.SERIES[0]
    x = alt.X("value:Q", title="Cross-validated estimate and 95 % interval",
              scale=alt.Scale(domain=[0, 1], nice=False))  # fmt: skip
    y = alt.Y("metric:N", sort=order, title=None, axis=alt.Axis(ticks=False, domain=False))
    chance_rule = (
        alt.Chart(pd.DataFrame({"x": [chance], "label": [f"chance {chance:g}"]}))
        .mark_rule(color=t.AMBER, strokeWidth=1, opacity=0.8)
        .encode(x="x:Q")
    )
    chance_text = (
        alt.Chart(pd.DataFrame({"x": [chance], "label": [f"chance {chance:g}"]}))
        .mark_text(color=t.AMBER, fontSize=10, align="left", dx=4, baseline="top")
        .encode(x="x:Q", y=alt.value(0), text="label:N")
    )
    tooltip = [
        alt.Tooltip("metric:N", title="Metric"),
        alt.Tooltip("value:Q", title="Estimate", format=".3f"),
        alt.Tooltip("low:Q", title="95 % low", format=".3f"),
        alt.Tooltip("high:Q", title="95 % high", format=".3f"),
        alt.Tooltip("p:Q", title="Permutation p", format=".3f"),
    ]
    spans = alt.Chart(frame).mark_rule(color=colour, strokeWidth=2.5, opacity=0.55).encode(
        x=alt.X("low:Q", scale=alt.Scale(domain=[0, 1], nice=False)), x2="high:Q", y=y,
        tooltip=tooltip,
    )  # fmt: skip
    dots = (
        alt.Chart(frame)
        .mark_circle(color=colour, size=90, opacity=1, stroke=t.SURFACE, strokeWidth=2)
        .encode(x=x, y=y, tooltip=tooltip)
    )
    return finish(alt.layer(chance_rule, chance_text, spans, dots), height=36 * len(frame) + 50)


def contributions(
    top: Mapping[str, Any], ends: tuple[str, str], greyed: bool = False
) -> alt.TopLevelMixin:
    """The features that moved one prediction most (``top_features`` of a stage's result)."""
    features = top["features"]
    signed = "contribution" in features[0]
    frame = pd.DataFrame(
        {
            "feature": [f["feature"].replace("_", " ") for f in features],
            "value": [f["contribution"] if signed else f["importance"] for f in features],
            "standardised": [f.get("standardised_value", np.nan) for f in features],
        }
    )
    order = list(frame["feature"])
    if signed:
        frame["towards"] = np.where(frame["value"] > 0, ends[1], ends[0])
        colour = alt.Color(
            "towards:N",
            scale=alt.Scale(
                domain=[ends[1], ends[0]],
                range=[t.CONTEXT, t.CONTEXT] if greyed else [t.SERIES[0], t.SERIES[1]],
            ),
            legend=alt.Legend(title="Pushes towards"),
        )
        limit = float(frame["value"].abs().max()) * 1.15 or 1.0
        x = alt.X(
            "value:Q",
            title="Contribution to the score (coefficient × standardised value)",
            scale=alt.Scale(domain=[-limit, limit]),
        )
        tooltip = [
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("value:Q", title="Contribution", format="+.3f"),
            alt.Tooltip("standardised:Q", title="Standardised value", format="+.2f"),
            alt.Tooltip("towards:N", title="Towards"),
        ]
    else:
        colour = alt.value(t.CONTEXT if greyed else t.SERIES[0])
        x = alt.X("value:Q", title="Impurity importance (the same for every recording)")
        tooltip = [
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("value:Q", title="Importance", format=".3f"),
        ]
    bars = (
        alt.Chart(frame)
        .mark_bar(size=14, cornerRadiusEnd=3)
        .encode(
            y=alt.Y("feature:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
            x=x,
            color=colour,
            tooltip=tooltip,
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=t.AXIS).encode(x="x:Q")
    return finish(alt.layer(bars, zero), height=30 * len(frame) + 50)
