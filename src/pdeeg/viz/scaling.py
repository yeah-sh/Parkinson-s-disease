"""Figures of the scaling inspection: fluctuation functions and their local slopes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

from pdeeg.features.scaling import SeriesCurves, local_slopes
from pdeeg.viz.style import ACCENT, CONTEXT, GRID, HEADER, INK_SECONDARY, MUTED, new_figure, styled

# Moment orders run from blue (negative) through grey (zero) to red (positive), darker the
# further from zero. The curves are also stacked in that order, since Fq rises with q.
_NEGATIVE = ("#9ec5f4", "#104281")
_POSITIVE = ("#f3a9a8", "#a32122")
_NULL = "#eb6834"
_NULL_DASHES = ((0, (5, 2)), (0, (1.5, 1.5)))
_BAND = "#ecebe6"  # shading of a fit range: one step off the surface, behind everything


def q_colors(qs: Sequence[float]) -> list[str]:
    """One colour per moment order: blue below zero, grey at zero, red above."""
    largest = max(abs(q) for q in qs) or 1.0
    colors = []
    for q in qs:
        if q == 0:
            colors.append(MUTED)
            continue
        light, dark = (to_rgb(c) for c in (_NEGATIVE if q < 0 else _POSITIVE))
        weight = abs(q) / largest
        colors.append(
            to_hex([(1 - weight) * a + weight * b for a, b in zip(light, dark, strict=True)])
        )
    return colors


def _seconds(value: float, _position: float | None = None) -> str:
    return f"{value:g}"


def _log_seconds_axis(axes: Axes) -> None:
    axes.set_xscale("log")
    axes.xaxis.set_major_formatter(FuncFormatter(_seconds))
    axes.xaxis.set_minor_formatter(NullFormatter())


def _shade_fit_ranges(
    axes: Axes, fit_ranges: Mapping[str, tuple[float, float]], *, label: bool
) -> None:
    for name, (low, high) in fit_ranges.items():
        axes.axvspan(low, high, facecolor=_BAND, edgecolor="none", zorder=0)
        if label:
            axes.annotate(
                name,
                (np.sqrt(low * high), 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK_SECONDARY,
            )


@styled
def plot_fluctuations(
    curves: SeriesCurves,
    sfreq: float,
    qs: Sequence[float],
    channels: Sequence[str],
    show: Sequence[str],
    fit_ranges: Mapping[str, tuple[float, float]],
    title: str,
    subtitle: str,
) -> Figure:
    """``Fq(s)`` on log-log axes: one row per recording, one column per channel in ``show``.

    ``channels`` names the channel axis of ``curves.fq``. The fit ranges, in seconds, are
    shaded.
    """
    n_recordings = curves.fq.shape[0]
    seconds = curves.scales / sfreq
    colors = q_colors(qs)
    figure = new_figure(2.3 * len(show) + 1.0, 1.45 * n_recordings + HEADER + 1.0, title, subtitle)
    grid = figure.subplots(n_recordings, len(show), sharex=True, sharey="row", squeeze=False)
    for i, row in enumerate(grid):
        for axes, channel in zip(row, show, strict=True):
            fq = curves.fq[i, channels.index(channel)]
            _shade_fit_ranges(axes, fit_ranges, label=i == 0)
            for values, color in zip(fq, colors, strict=True):
                axes.plot(seconds, values, color=color, lw=1.2)
            axes.set_yscale("log")
            _log_seconds_axis(axes)
            axes.yaxis.set_minor_formatter(NullFormatter())
            axes.tick_params(labelsize=7.5)
            if i == 0:
                axes.set_title(channel, fontsize=9.5, fontweight="bold", pad=14)
        row[0].set_ylabel(f"R{i + 1}", rotation=0, ha="right", va="center", fontweight="bold")
    for axes in grid[-1]:
        axes.set_xlabel("Window size s (seconds)", fontsize=8)
    figure.supylabel("Fq(s) (µV, log scale)", fontsize=9, color=INK_SECONDARY)
    handles = [
        Line2D([], [], color=c, lw=1.6, label=f"q = {q:g}") for q, c in zip(qs, colors, strict=True)
    ]
    handles.append(Patch(facecolor=_BAND, label="fit range"))
    figure.legend(handles=handles, loc="outside lower center", ncols=len(handles), fontsize=8)
    return figure


@styled
def plot_local_slopes(
    curves: Mapping[str, SeriesCurves],
    sfreq: float,
    qs: Sequence[float],
    show_qs: Sequence[float],
    half_width: int,
    fit_ranges: Mapping[str, Mapping[str, tuple[float, float]]],
    titles: Mapping[str, str],
    ylims: Mapping[str, tuple[float, float]],
    title: str,
    subtitle: str,
) -> Figure:
    """Local slope of ``log Fq`` against ``log s``: one row per series, one column per q.

    Grey lines are the recordings (median over channels), the blue line the median over every
    channel of every recording, and the orange lines the noises (mean over realisations). A
    power law is a level stretch; a crossover is where the level changes.
    """
    names = list(curves)
    qs = list(qs)
    figure = new_figure(3.6 * len(show_qs) + 0.6, 2.25 * len(names) + HEADER + 0.9, title, subtitle)
    grid = figure.subplots(len(names), len(show_qs), sharex="row", sharey="row", squeeze=False)
    noises: dict[str, tuple] = {}
    for row, name in zip(grid, names, strict=True):
        series = curves[name]
        seconds = series.scales / sfreq
        slopes = local_slopes(series.scales, series.fq, half_width)
        for dashes, noise in zip(_NULL_DASHES, series.null, strict=False):
            noises.setdefault(noise, dashes)
        for axes, q in zip(row, show_qs, strict=True):
            k = qs.index(q)
            _shade_fit_ranges(axes, fit_ranges[name], label=True)
            for recording in np.median(slopes[:, :, k], axis=1):
                axes.plot(seconds, recording, color=CONTEXT, lw=0.8)
            for noise, values in series.null.items():
                mean = local_slopes(series.scales, values, half_width)[:, k].mean(axis=0)
                axes.plot(seconds, mean, color=_NULL, lw=1.5, linestyle=noises[noise])
            pooled = np.median(slopes[:, :, k], axis=(0, 1))
            axes.plot(seconds, pooled, color=ACCENT, lw=2.0)
            axes.axhline(0.5, color=GRID, lw=0.8, zorder=0.5)
            _log_seconds_axis(axes)
            axes.set_ylim(*ylims[name])
            axes.set_title(
                f"{titles[name]}, q = {q:g}", loc="left", fontsize=9.5, fontweight="bold", pad=14
            )
        row[0].set_ylabel("Local slope")
    for axes in grid[-1]:
        axes.set_xlabel("Window size s (seconds, log scale)")
    handles = [
        Line2D([], [], color=ACCENT, lw=2.0, label="EEG: median of all channels and recordings"),
        Line2D([], [], color=CONTEXT, lw=0.8, label="EEG: each recording (median of channels)"),
        *(
            Line2D([], [], color=_NULL, lw=1.5, linestyle=dashes, label=f"filtered {noise}")
            for noise, dashes in noises.items()
        ),
        Patch(facecolor=_BAND, label="fit range"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncols=len(handles), fontsize=8)
    return figure
