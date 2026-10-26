"""Figures of the two-stage cascade: its confusion matrix, reliability, and permutation nulls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from pdeeg.viz.style import (
    ACCENT,
    AXIS,
    CONTEXT,
    HEADER,
    INK,
    INK_SECONDARY,
    SURFACE,
    new_figure,
    styled,
)

# The second series of a figure (Platt-scaled probabilities): slot 2 of the categorical palette
# whose slot 1 is ACCENT.
SECOND = "#eb6834"
# One hue, light to dark, for shares from 0 to 1.
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "share", ["#f3f7fd", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#104281"]
)


@styled
def plot_confusion(counts: np.ndarray, classes: Sequence[str], title: str, subtitle: str) -> Figure:
    """Recordings by true class (rows) and the cascade's call (columns).

    The shade is the share of the row; each cell gives the count and that share.
    """
    counts = np.asarray(counts)
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    figure = new_figure(4.6, 3.4 + HEADER, title, subtitle)
    axes = figure.subplots()
    axes.imshow(shares, cmap=_SEQUENTIAL, vmin=0.0, vmax=1.0, aspect="equal")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            dark = shares[i, j] > 0.55
            axes.text(
                j,
                i,
                f"{counts[i, j]}\n{shares[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color=SURFACE if dark else INK,
                fontweight="bold" if i == j else "normal",
            )
    # Surface-coloured gaps between the cells.
    for edge in np.arange(-0.5, counts.shape[0] + 0.5):
        axes.axhline(edge, color=SURFACE, lw=3)
        axes.axvline(edge, color=SURFACE, lw=3)
    labels = list(classes)
    axes.set_xticks(range(len(labels)), labels)
    axes.set_yticks(range(len(labels)), labels)
    axes.set_xlabel("Called by the cascade")
    axes.set_ylabel("True class")
    axes.grid(visible=False)
    for side in axes.spines.values():
        side.set_visible(False)
    axes.tick_params(length=0)
    return figure


@styled
def plot_reliability(
    panels: Mapping[str, Mapping[str, pd.DataFrame]], title: str, subtitle: str
) -> Figure:
    """Observed share of positives against mean probability, one panel per stage.

    ``panels`` maps a panel title to {"raw": table, "platt": table}, tables as
    :func:`pdeeg.models.calibration.reliability_table` makes them. The diagonal is perfect
    calibration; bars are 95 % Wilson intervals.
    """
    figure = new_figure(3.4 * len(panels) + 0.4, 4.0 + HEADER, title, subtitle)
    grid = figure.subplots(1, len(panels), squeeze=False)[0]
    styles = {"raw": (ACCENT, "Raw"), "platt": (SECOND, "Platt-scaled")}
    for axes, (panel, tables) in zip(grid, panels.items(), strict=True):
        axes.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
        for version, table in tables.items():
            colour, label = styles[version]
            axes.plot(table["mean_probability"], table["observed"], color=colour, lw=2, zorder=2)
            axes.errorbar(
                table["mean_probability"],
                table["observed"],
                yerr=[
                    table["observed"] - table["observed_low"],
                    table["observed_high"] - table["observed"],
                ],
                fmt="o",
                color=colour,
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                elinewidth=1.5,
                capsize=0,
                label=label,
                zorder=3,
            )
        axes.set_xlim(-0.02, 1.02)
        axes.set_ylim(-0.02, 1.02)
        axes.set_aspect("equal")
        axes.set_title(panel, loc="left", fontsize=9)
        axes.set_xlabel("Mean probability in the bin")
        axes.tick_params(labelsize=7.5)
    grid[0].set_ylabel("Observed share of positives")
    grid[0].legend(loc="upper left", fontsize=8, handletextpad=0.3)
    return figure


@styled
def plot_nulls(
    panels: Mapping[str, tuple[pd.Series, float, float]], title: str, subtitle: str
) -> Figure:
    """Each metric's permutation null as a histogram, with the observed value and chance.

    ``panels`` maps a panel title to (null values, observed value, chance level).
    """
    figure = new_figure(2.8 * len(panels) + 0.4, 2.9 + HEADER, title, subtitle)
    grid = figure.subplots(1, len(panels), squeeze=False, sharey=False)[0]
    for axes, (panel, (null, observed, chance)) in zip(grid, panels.items(), strict=True):
        values = np.asarray(null, dtype=float)
        values = values[np.isfinite(values)]
        axes.hist(
            values,
            bins=np.linspace(0.0, 1.0, 41),
            color=CONTEXT,
            edgecolor=SURFACE,
            linewidth=1,
        )
        axes.axvline(chance, color=INK_SECONDARY, lw=1, ls=(0, (3, 2)))
        axes.axvline(observed, color=ACCENT, lw=2)
        top = axes.get_ylim()[1]
        for x, height, text, colour in (
            (observed, top, f" observed {observed:.2f} ", INK),
            (chance, 0.88 * top, f" chance {chance:.2f} ", INK_SECONDARY),
        ):
            side = "left" if x < 0.75 else "right"
            axes.text(x, height, text, color=colour, fontsize=7.5, ha=side, va="top")
        axes.set_xlim(0.0, 1.0)
        axes.set_title(panel, loc="left", fontsize=9)
        axes.grid(axis="x", visible=False)
        axes.tick_params(labelsize=7.5)
    grid[0].set_ylabel("Relabellings")
    return figure
