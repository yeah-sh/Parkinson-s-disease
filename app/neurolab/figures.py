"""Scalp maps for the dark interface, drawn by MNE's plot_topomap on a transparent figure.

The values are the channels' own; every map's sensor positions come from the standard montage
of configs/data.yaml (pdeeg.viz.stats.montage_info), as in the project's report figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mne
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from neurolab import theme as t
from pdeeg.viz.stats import montage_info

SEQUENTIAL = LinearSegmentedColormap.from_list("neurolab_sequential", t.SEQUENTIAL)
DIVERGING = LinearSegmentedColormap.from_list("neurolab_diverging", t.DIVERGING)


@dataclass(frozen=True)
class Map:
    """One head: a value per channel, its title, and whether it has a sign (centred on 0)."""

    values: np.ndarray
    title: str
    signed: bool


def limits(values: np.ndarray, signed: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return (-1.0, 1.0) if signed else (0.0, 1.0)
    if signed:
        top = float(np.abs(finite).max()) or 1.0
        return -top, top
    low, high = float(finite.min()), float(finite.max())
    return (low, high) if high > low else (low - 0.5, high + 0.5)


def _restyle(axes) -> None:
    """MNE draws the head, sensors and names in black: lighten them for the dark surface."""
    for line in axes.lines:
        line.set_color(t.TEXT_3)
        line.set_markerfacecolor(t.TEXT_2)
        line.set_markeredgecolor(t.TEXT_2)
    for text in axes.texts:
        text.set_color(t.TEXT)
    axes.title.set_color(t.TEXT_2)


def scalp_maps(
    maps: Sequence[Map], channels: Sequence[str], montage: str, *, names: bool = False,
    unit: str = "", panel: float = 2.3,
) -> Figure:  # fmt: skip
    """A row of heads, each with its own colour bar underneath.

    A channel without a value (NaN) is left off its head rather than filled in.
    """
    figure = Figure(figsize=(panel * len(maps), panel + 0.75), dpi=150, layout="constrained")
    figure.patch.set_alpha(0.0)
    FigureCanvasAgg(figure)
    grid = figure.subplots(1, len(maps), squeeze=False)[0]
    for axes, head in zip(grid, maps, strict=True):
        axes.set_facecolor("none")
        values = np.asarray(head.values, dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 4:
            axes.set_axis_off()
            axes.set_title(f"{head.title}\n(too few values)", fontsize=9, color=t.TEXT_3)
            continue
        kept = [channel for channel, ok in zip(channels, finite, strict=True) if ok]
        image, _ = mne.viz.plot_topomap(
            values[finite],
            montage_info(kept, montage),
            axes=axes,
            show=False,
            cmap=DIVERGING if head.signed else SEQUENTIAL,
            vlim=limits(values, head.signed),
            contours=0,
            sensors=True,
            names=kept if names else None,
            image_interp="cubic",
        )
        axes.set_title(head.title, fontsize=9 if len(maps) > 1 else 10, pad=6)
        _restyle(axes)
        bar = figure.colorbar(image, ax=axes, orientation="horizontal", shrink=0.78, pad=0.04)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7, length=2, colors=t.TEXT_2)
        if unit:
            bar.set_label(unit, color=t.TEXT_2, fontsize=7.5)
    return figure
