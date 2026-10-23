"""Figures of the statistical analysis: topographies, h(q) and f(alpha), paired values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from pdeeg.viz.style import CONTEXT, HEADER, INK, INK_SECONDARY, MUTED, new_figure, styled

# One colour per group, the first three slots of the categorical palette, which stay apart
# for colour-blind readers when all three are on screen. A group keeps its colour everywhere.
GROUP_COLOURS = {"HC": "#2a78d6", "PD OFF": "#eb6834", "PD ON": "#1baf7a"}
# Magnitude: one hue, light to dark. Polarity: blue below zero, red above, grey at zero.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "pdeeg_sequential", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "pdeeg_diverging",
    ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a19e", "#e34948", "#8e1f1f"],
)
# Channels in a significant cluster.
_MASK_STYLE = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": INK,
    "markeredgewidth": 1.0,
    "linewidth": 0,
    "markersize": 6,
}
_PANEL = 1.85  # inches per topography


@dataclass(frozen=True)
class Topography:
    """One head: a value per channel, how to colour it, and which channels to mark.

    ``kind`` is ``mean`` (sequential colours) or ``t`` (diverging, centred on 0).
    """

    values: np.ndarray
    title: str
    kind: str
    limits: tuple[float, float]
    mask: np.ndarray | None = None


def montage_info(channels: Sequence[str], montage: str) -> mne.Info:
    """An Info with ``channels`` placed by the standard ``montage``, for topographies."""
    info = mne.create_info(list(channels), 1.0, "eeg")
    info.set_montage(montage, verbose="error")
    return info


@styled
def plot_topographies(
    info: mne.Info,
    rows: Sequence[tuple[str, Sequence[Topography | None]]],
    title: str,
    subtitle: str,
) -> Figure:
    """A grid of topographies: one row per (label, heads) pair of ``rows``.

    A None leaves its place empty. Every head has its own colour bar underneath.
    """
    n_columns = max(len(panels) for _, panels in rows)
    figure = new_figure(
        _PANEL * n_columns + 0.9, (_PANEL + 0.55) * len(rows) + HEADER + 0.3, title, subtitle
    )
    grid = figure.subplots(len(rows), n_columns, squeeze=False)
    for axes_row, (label, panels) in zip(grid, rows, strict=True):
        for column, axes in enumerate(axes_row):
            panel = panels[column] if column < len(panels) else None
            if panel is None:
                axes.set_axis_off()
                continue
            image, _ = mne.viz.plot_topomap(
                panel.values,
                info,
                axes=axes,
                show=False,
                cmap=SEQUENTIAL if panel.kind == "mean" else DIVERGING,
                vlim=panel.limits,
                mask=panel.mask,
                mask_params=_MASK_STYLE,
                contours=0,
                sensors=True,
            )
            axes.set_title(panel.title, fontsize=8.5, color=INK_SECONDARY)
            bar = figure.colorbar(image, ax=axes, orientation="horizontal", shrink=0.75, pad=0.03)
            bar.outline.set_visible(False)
            bar.ax.tick_params(labelsize=7, length=2)
        axes_row[0].text(
            -0.12, 0.5, label, transform=axes_row[0].transAxes, rotation=90, ha="right",
            va="center", fontsize=9, fontweight="bold",
        )  # fmt: skip
    return figure


@dataclass(frozen=True)
class GroupCurves:
    """Group mean of h(q), alpha(q) and f(alpha(q)) with bootstrap intervals, one value per q.

    Each of ``h``, ``alpha`` and ``f`` is the triple (mean, lower end, upper end).
    """

    q: np.ndarray
    h: tuple[np.ndarray, np.ndarray, np.ndarray]
    alpha: tuple[np.ndarray, np.ndarray, np.ndarray]
    f: tuple[np.ndarray, np.ndarray, np.ndarray]
    n: int


@styled
def plot_curves(
    curves: Mapping[str, Mapping[str, Mapping[str, GroupCurves]]],
    title: str,
    subtitle: str,
) -> Figure:
    """h(q) and the singularity spectrum, group against group.

    ``curves`` maps a row title (a series) to comparisons, and each comparison maps a group
    (a key of ``GROUP_COLOURS``) to its curves. Each comparison takes two columns: h(q) with
    a shaded interval, then f(alpha) with intervals in both directions at every q.
    """
    comparisons = list(next(iter(curves.values())))
    n_columns = 2 * len(comparisons)
    figure = new_figure(2.6 * n_columns + 0.4, 2.55 * len(curves) + HEADER + 0.6, title, subtitle)
    grid = figure.subplots(len(curves), n_columns, squeeze=False)
    for axes_row, (series, by_comparison) in zip(grid, curves.items(), strict=True):
        for i, (comparison, groups) in enumerate(by_comparison.items()):
            h_axes, f_axes = axes_row[2 * i], axes_row[2 * i + 1]
            for group, curve in groups.items():
                colour = GROUP_COLOURS[group]
                label = f"{group} (n = {curve.n})"
                mean, low, high = curve.h
                h_axes.fill_between(curve.q, low, high, color=colour, alpha=0.18, linewidth=0)
                h_axes.plot(curve.q, mean, color=colour, lw=2, label=label)
                alpha, f = curve.alpha, curve.f
                f_axes.errorbar(
                    alpha[0],
                    f[0],
                    xerr=[alpha[0] - alpha[1], alpha[2] - alpha[0]],
                    yerr=[f[0] - f[1], f[2] - f[0]],
                    color=colour,
                    lw=2,
                    elinewidth=0.8,
                    capsize=0,
                    marker="o",
                    markersize=3,
                    label=label,
                )
            h_axes.set_xlabel("q")
            h_axes.set_ylabel("h(q)")
            f_axes.set_xlabel("α")
            f_axes.set_ylabel("f(α)")
            h_axes.set_title(f"{series}\n{comparison}: h(q)", loc="left", fontsize=9)
            f_axes.set_title(f"{series}\n{comparison}: f(α)", loc="left", fontsize=9)
            h_axes.legend(fontsize=7.5, loc="best")
            f_axes.legend(fontsize=7.5, loc="lower center")
            for axes in (h_axes, f_axes):
                axes.tick_params(labelsize=7.5)
    return figure


@dataclass(frozen=True)
class PairedPanel:
    """One patient per row of ``table`` (columns ``first``, ``second``, ``included``), and the
    mean of each session over the included patients with the ends of its interval, each an
    array of two values (first, second)."""

    table: pd.DataFrame
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray


@styled
def plot_paired(
    panels: Mapping[str, PairedPanel], sessions: tuple[str, str], title: str, subtitle: str
) -> Figure:
    """Each patient's value in both sessions, joined by a line, with the means of both.

    ``panels`` maps a panel title to its values. ``sessions`` names the two sessions, which are
    keys of ``GROUP_COLOURS``. Patients left out of the main analysis are drawn dashed.
    """
    figure = new_figure(2.6 * len(panels) + 0.4, 3.3 + HEADER, title, subtitle)
    grid = figure.subplots(1, len(panels), squeeze=False)[0]
    for axes, (panel, values) in zip(grid, panels.items(), strict=True):
        table, means, low, high = values.table, values.mean, values.low, values.high
        for row in table.itertuples():
            axes.plot(
                [0, 1],
                [row.first, row.second],
                color=CONTEXT if row.included else MUTED,
                lw=1,
                ls="-" if row.included else (0, (3, 2)),
                zorder=1,
            )
        for x, session in enumerate(sessions):
            axes.errorbar(
                x,
                means[x],
                yerr=[[means[x] - low[x]], [high[x] - means[x]]],
                color=GROUP_COLOURS[session],
                marker="o",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.5,
                elinewidth=2,
                capsize=0,
                zorder=3,
            )
        axes.set_xticks([0, 1], sessions)
        axes.set_xlim(-0.35, 1.35)
        axes.grid(axis="x", visible=False)
        axes.set_title(panel, loc="left", fontsize=9)
        axes.tick_params(labelsize=7.5)
    grid[0].set_ylabel("Mean over channels")
    return figure
