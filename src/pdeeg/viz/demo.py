"""Figures of the research demo: fluctuation functions and the singularity spectrum."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pdeeg.viz.style import ACCENT, GRID, HEADER, SURFACE, new_figure, styled

# q is ordered, so its lines take steps of one hue, light (most negative q) to dark. The
# lightest step still stands out from the surface.
Q_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")


def _q_colours(n: int) -> list[str]:
    if n <= len(Q_RAMP):
        return list(Q_RAMP[len(Q_RAMP) - n :])
    return [Q_RAMP[round(i * (len(Q_RAMP) - 1) / (n - 1))] for i in range(n)]


@styled
def plot_fluctuations(
    scales: np.ndarray,
    qs: np.ndarray,
    fq: np.ndarray,
    ranges: Mapping[str, tuple[float, float]],
    title: str,
) -> Figure:
    """``log Fq(s)`` against ``log s`` for every q, with the configured fit ranges shaded.

    ``scales`` are in seconds and ``fq`` has one row per q.
    """
    figure = new_figure(
        6.4, 3.9 + HEADER, title, "Shaded: the fit ranges h(q) is read from (configured)."
    )
    axes = figure.subplots()
    for name, (low, high) in ranges.items():
        axes.axvspan(low, high, color=GRID, alpha=0.6, linewidth=0, zorder=0)
        axes.text(
            np.sqrt(low * high), 1.0, name, transform=axes.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8,
        )  # fmt: skip
    for q, row, colour in zip(qs, fq, _q_colours(len(qs)), strict=True):
        axes.plot(scales, row, color=colour, lw=2, label=f"q = {q:g}")
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("Window size s (seconds)")
    axes.set_ylabel("Fq(s)")
    axes.legend(fontsize=8, loc="upper left")
    return figure


@styled
def plot_spectrum(curves: pd.DataFrame, title: str) -> Figure:
    """h(q) beside the singularity spectrum f(alpha) it gives."""
    figure = new_figure(7.4, 3.4 + HEADER, title, "Bad windows left out. One point per q.")
    h_axes, f_axes = figure.subplots(1, 2)
    h_axes.plot(curves["q"], curves["h"], color=ACCENT, lw=2)
    h_axes.set_xlabel("q")
    h_axes.set_ylabel("h(q)")
    f_axes.plot(
        curves["alpha"], curves["f_alpha"], "o-", color=ACCENT, lw=2, markersize=5,
        markeredgecolor=SURFACE, markeredgewidth=1,
    )  # fmt: skip
    f_axes.set_xlabel("α")
    f_axes.set_ylabel("f(α)")
    return figure
