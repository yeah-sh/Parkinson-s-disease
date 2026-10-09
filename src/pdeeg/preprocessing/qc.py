"""QC figures for the preprocessing stage, in the shared style of :mod:`pdeeg.viz.style`."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import mne
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from mne.preprocessing import ICA

from pdeeg.config import PreprocessingConfig
from pdeeg.viz.style import (
    ACCENT,
    CONTEXT,
    HEADER,
    INK,
    INK_SECONDARY,
    MUTED,
    PX,
    SURFACE,
    new_figure,
    styled,
)

# One fixed colour per ICLabel artefact class, so a class looks the same in every figure.
CLASS_COLORS = {
    "eye blink": "#2a78d6",
    "muscle artifact": "#eb6834",
    "heart beat": "#1baf7a",
    "line noise": "#eda100",
    "channel noise": "#e87ba4",
    "other": "#008300",
}

_BAR_MAX_THICKNESS = 24 * PX
_BAR_RADIUS = 4 * PX
_BAR_GAP = 2 * PX


def count_column(label: str) -> str:
    """Name of the QC table column that counts removed components of one ICLabel class."""
    return "n_" + label.replace(" ", "_")


# --- one recording ------------------------------------------------------------------------------


@styled
def plot_psd(log: Mapping[str, Any], config: PreprocessingConfig, title: str) -> Figure:
    """Spectrum of one recording before and after cleaning, from its log."""
    freqs = np.asarray(log["psd"]["freqs_hz"])
    show = freqs >= config.filter.l_freq
    freqs = freqs[show]
    before = np.asarray(log["psd"]["before_db"])[show]
    after = np.asarray(log["psd"]["after_db"])[show]

    line = log["line_noise"]
    mains = "none detected" if line["detected_hz"] is None else f"{line['detected_hz']:g} Hz"
    figure = new_figure(
        7.0,
        3.9,
        f"{title}: power spectrum before and after cleaning",
        f"Median over {log['n_channels']} channels. Band kept: {config.filter.l_freq:g} to "
        f"{config.filter.h_freq:g} Hz. Mains peak: {mains}.",
    )
    axes = figure.subplots()
    axes.plot(freqs, before, color=MUTED, label="before: cropped and re-referenced only")
    axes.plot(freqs, after, color=ACCENT, label="after cleaning")
    # The cleaned spectrum falls off a cliff above the low-pass; let it run out of the frame
    # instead of squashing the band that matters.
    axes.set_ylim(before.min() - 15, max(before.max(), after.max()) + 5)
    axes.set_xlim(0, freqs[-1])
    axes.set_xlabel("Frequency (Hz)")
    axes.set_ylabel("Power (dB re 1 µV²/Hz)")
    axes.legend(loc="upper right")

    def label(text: str, at_hz: float, values: np.ndarray, above: bool) -> None:
        i = int(np.argmin(np.abs(freqs - at_hz)))
        axes.annotate(
            text,
            (freqs[i], values[i]),
            xytext=(0, 6 if above else -6),
            textcoords="offset points",
            ha="center",
            va="bottom" if above else "top",
            color=INK_SECONDARY,
        )

    label("before", 0.8 * freqs[-1], before, above=True)
    label("after", 0.5 * config.filter.h_freq, after, above=False)
    return figure


@styled
def plot_removed_components(
    ica: ICA, info: mne.Info, removed: Sequence[Mapping[str, Any]], title: str
) -> Figure:
    """Scalp maps of the ICA components that were removed, with ICLabel's verdict on each."""
    n_cols = min(len(removed), 6)
    n_rows = math.ceil(len(removed) / n_cols)
    count = f"{len(removed)} ICA component{'s' if len(removed) != 1 else ''} removed"
    figure = new_figure(
        max(1.6 * n_cols, 6.0),
        1.95 * n_rows + HEADER + 0.1,
        f"{title}: {count}",
        f"Of {ica.n_components_}. Each map is titled with ICLabel's class and its probability.",
    )
    grid = figure.subplots(n_rows, n_cols, squeeze=False)
    maps = ica.get_components()
    for axes in grid.ravel():
        axes.set_axis_off()
    for axes, component in zip(grid.ravel(), removed, strict=False):
        # Red and blue are the two polarities of the map; white is zero.
        mne.viz.plot_topomap(
            maps[:, component["index"]], info, axes=axes, cmap="RdBu_r", show=False
        )
        axes.set_title(
            f"IC{component['index']:02d}\n{component['label']} {component['probability']:.2f}",
            fontsize=8,
        )
    return figure


# --- all recordings -----------------------------------------------------------------------------


def _pixel_scale(axes: Axes) -> tuple[float, float]:
    """Pixels per data unit along x and y. Only valid once the figure has been laid out."""
    (x0, y0), (x1, y1) = axes.transData.transform([(0.0, 0.0), (1.0, 1.0)])
    return abs(x1 - x0), abs(y1 - y0)


def _bar(axes: Axes, left: float, right: float, y: float, color: str, *, rounded: bool) -> None:
    """A horizontal bar segment from ``left`` to ``right``; ``rounded`` rounds its right end."""
    scale_x, scale_y = _pixel_scale(axes)
    thickness = min(0.64, _BAR_MAX_THICKNESS / scale_y)
    bottom = y - thickness / 2
    body = Rectangle((left, bottom), right - left, thickness, facecolor=color, edgecolor="none")
    if not rounded or (right - left) * scale_x < 2 * _BAR_RADIUS:
        axes.add_patch(body)
        return
    # A box rounded at both ends, started early and clipped back to the segment, so that only
    # the data end is round and the baseline end stays square.
    radius = _BAR_RADIUS / scale_x
    box = FancyBboxPatch(
        (left - radius, bottom),
        right - left + radius,
        thickness,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        mutation_aspect=scale_x / scale_y,
        facecolor=color,
        edgecolor="none",
    )
    axes.add_patch(box)
    body.set_transform(axes.transData)
    box.set_clip_path(body)


def _prepare_bars(
    axes: Axes, table: pd.DataFrame, columns: Sequence[str], title: str, xlabel: str, xmax: float
) -> None:
    """Scales and labels of a bar panel. Bars are added once every panel has been prepared."""
    totals = table[list(columns)].sum(axis=1).to_numpy()
    axes.set_xlim(0, xmax if xmax else max(float(np.nanmax(totals)) * 1.18, 1e-9))
    axes.set_ylim(len(table) - 0.4, -0.6)
    axes.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    axes.set_xlabel(xlabel)
    axes.grid(axis="y", visible=False)
    axes.tick_params(axis="y", length=0)


def _draw_bars(axes: Axes, table: pd.DataFrame, columns: Mapping[str, str], fmt: str) -> None:
    """Horizontal bars, one row per recording; several ``columns`` (name -> colour) stack."""
    gap = _BAR_GAP / _pixel_scale(axes)[0]
    for y, (_, row) in enumerate(table.iterrows()):
        left = 0.0
        parts = [(name, float(row[name])) for name in columns if row[name] > 0]
        for i, (name, value) in enumerate(parts):
            last = i == len(parts) - 1
            # Stacked segments are separated by a sliver of the surface, not by an outline.
            right = left + value - (0.0 if last else gap)
            _bar(axes, left, max(right, left), y, columns[name], rounded=last)
            left += value
        if row["flagged"]:
            axes.annotate(
                format(left, fmt),
                (left, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=INK_SECONDARY,
                # Keeps the number readable where it lands on a limit line.
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0},
            )


@styled
def plot_overview(table: pd.DataFrame, config: PreprocessingConfig) -> Figure:
    """Bad time, longest clean stretch and ICA components removed, one row per recording.

    ``table`` needs ``label``, ``flagged``, ``bad_percent``, ``longest_clean_s`` and one
    :func:`count_column` per excluded ICLabel class.
    """
    # Only the classes that occur get a segment colour and a legend entry.
    labels = [name for name in config.ica.exclude_labels if table[count_column(name)].sum() > 0]
    classes = {count_column(name): CLASS_COLORS[name] for name in labels}
    # The dashed lines are the flagging limits; the axis labels say so.
    limits = (config.qc.max_bad_percent, config.qc.min_clean_stretch_s, None)
    panels = (
        (
            {"bad_percent": CONTEXT},
            "Time marked bad",
            f"% of the recording (dashed: flagged above {limits[0]:g})",
            ".1f",
            0.0,
        ),
        (
            {"longest_clean_s": CONTEXT},
            "Longest stretch with no bad mark",
            f"seconds (dashed: flagged below {limits[1]:g})",
            ".0f",
            config.crop.duration * 1.12,
        ),
        (classes, "ICA components removed", "components", ".0f", 0.0),
    )
    figure = new_figure(
        11.0,
        0.205 * len(table) + HEADER + 1.3,
        f"Preprocessing QC: {len(table)} recordings",
        "Flagged recordings are in bold and carry their value.",
    )
    all_axes = figure.subplots(1, 3, sharey=True, width_ratios=[1.0, 1.0, 1.15])
    all_axes[0].set_yticks(range(len(table)), table["label"])
    for tick, flagged in zip(all_axes[0].get_yticklabels(), table["flagged"], strict=True):
        tick.set_fontsize(8)
        if flagged:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    for axes, (columns, title, xlabel, _, xmax) in zip(all_axes, panels, strict=True):
        _prepare_bars(axes, table, list(columns), title, xlabel, xmax)
    all_axes[2].xaxis.get_major_locator().set_params(integer=True)
    if labels:
        figure.legend(
            handles=[Patch(facecolor=CLASS_COLORS[name], label=name) for name in labels],
            title="Class of the removed components (right panel)",
            loc="outside lower center",
            ncols=len(labels),
            handlelength=1.0,
            handleheight=1.0,
            fontsize=8,
            title_fontsize=8,
        )

    # Bar geometry is in pixels, so the layout has to be final before the bars go in.
    figure.draw_without_rendering()
    for axes, limit, (columns, _, _, fmt, _) in zip(all_axes, limits, panels, strict=True):
        if limit is not None and limit <= axes.get_xlim()[1]:
            axes.axvline(limit, color=INK_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)))
        _draw_bars(axes, table, columns, fmt)
    return figure


@styled
def plot_psd_overview(
    freqs: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    flagged: np.ndarray,
    config: PreprocessingConfig,
) -> Figure:
    """Every recording's spectrum before and after cleaning; flagged recordings stand out.

    ``before`` and ``after`` are recordings x frequencies in dB; ``flagged`` is one boolean per
    recording.
    """
    figure = new_figure(
        11.0,
        4.3,
        f"Power spectra of all {len(before)} recordings",
        "Median over channels, one line per recording. The two panels have their own scales.",
    )
    panels = figure.subplots(1, 2)
    spans = (
        (panels[0], before, freqs[-1], "Before: cropped and re-referenced only"),
        (panels[1], after, config.filter.h_freq + 10.0, "After cleaning"),
    )
    for axes, spectra, fmax, heading in spans:
        show = (freqs >= config.filter.l_freq) & (freqs <= fmax)
        for mask, color, width, alpha, order in (
            (~flagged, MUTED, 0.8, 0.55, 2),
            (flagged, ACCENT, 1.3, 0.95, 3),
        ):
            for spectrum in spectra[mask]:
                axes.plot(
                    freqs[show], spectrum[show], color=color, lw=width, alpha=alpha, zorder=order
                )
        visible = spectra[:, show]
        in_band = visible[:, freqs[show] <= config.filter.h_freq]
        axes.set_ylim(np.percentile(in_band, 0.5) - 8, visible.max() + 4)
        axes.set_xlim(0, fmax)
        axes.set_title(heading, loc="left", fontsize=9.5, fontweight="bold")
        axes.set_xlabel("Frequency (Hz)")
    panels[0].set_ylabel("Power (dB re 1 µV²/Hz)")
    panels[1].legend(
        handles=[
            Line2D([], [], color=ACCENT, lw=1.3, label="flagged"),
            Line2D([], [], color=MUTED, lw=0.8, label="not flagged"),
        ],
        loc="upper right",
    )
    return figure
