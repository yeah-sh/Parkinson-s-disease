"""Figures of the feature QC: surrogate z-scores, and whole recording against clean windows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure

from pdeeg.viz.style import ACCENT, AXIS, HEADER, INK_SECONDARY, MUTED, new_figure, styled

# z-scores are drawn between these percentiles of all panels of a row; the rest is counted in
# the outermost bars, so that a few extreme channels do not flatten the picture.
_CLIP_PERCENTILES = (0.5, 99.5)
_N_BINS = 40


def _histogram(axes, values: np.ndarray, limits: tuple[float, float], color: str) -> None:
    values = values[np.isfinite(values)]
    edges = np.linspace(*limits, _N_BINS + 1)
    counts, _ = np.histogram(np.clip(values, *limits), bins=edges)
    width = edges[1] - edges[0]
    # A sliver of the surface between neighbouring bars instead of an outline.
    axes.bar(edges[:-1] + width / 2, counts, width=0.86 * width, color=color, linewidth=0)
    axes.set_xlim(*limits)
    axes.grid(axis="x", visible=False)


@styled
def plot_surrogate_z(
    z: Mapping[str, Mapping[str, np.ndarray]],
    kinds: Mapping[str, str],
    threshold: float,
    title: str,
) -> Figure:
    """Histograms of the z-score of ``delta_alpha`` against surrogates.

    ``z`` maps a panel title (a series and fit range) to the z-scores of every channel of every
    recording for each kind of surrogate; ``kinds`` maps a kind to its row heading. One row per
    kind, one column per panel. Values beyond a row's axis are counted in its outermost bars.
    """
    panels = list(z)
    figure = new_figure(
        2.35 * len(panels) + 0.5,
        1.95 * len(kinds) + HEADER + 0.5,
        title,
        f"One value per channel and recording. Grey line: 0, no wider than the surrogates. "
        f"Dark line: {threshold:g}. Values beyond an axis are counted in its outermost bar.",
    )
    grid = figure.subplots(len(kinds), len(panels), sharey="row", squeeze=False)
    for row, (kind, heading) in zip(grid, kinds.items(), strict=True):
        pooled = np.concatenate([z[panel][kind] for panel in panels])
        low, high = np.percentile(pooled[np.isfinite(pooled)], _CLIP_PERCENTILES)
        limits = (min(low, -1.0), max(high, threshold + 1.0))
        for axes, panel in zip(row, panels, strict=True):
            _histogram(axes, z[panel][kind], limits, ACCENT)
            axes.axvline(0.0, color=AXIS, lw=0.8)
            axes.axvline(threshold, color=INK_SECONDARY, lw=0.8)
            axes.set_title(panel, loc="left", fontsize=9.5, fontweight="bold")
            axes.set_xlabel(heading, fontsize=8)
        row[0].set_ylabel("Channels")
    return figure


@styled
def plot_full_against_clean(
    values: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    features: Sequence[str],
    title: str,
    subtitle: str,
) -> Figure:
    """Each feature on the whole recording (x) against clean windows only (y).

    ``values`` maps a panel title to, for each feature, the pair of arrays (full, clean) with
    one value per channel of every recording that has a bad stretch. One row per feature, one
    column per panel; a point on the line is a channel the bad stretches did not move.
    """
    panels = list(values)
    figure = new_figure(
        2.35 * len(panels) + 0.5, 2.3 * len(features) + HEADER + 0.5, title, subtitle
    )
    grid = figure.subplots(len(features), len(panels), squeeze=False)
    for row, feature in zip(grid, features, strict=True):
        for axes, panel in zip(row, panels, strict=True):
            full, clean = values[panel][feature]
            both = np.isfinite(full) & np.isfinite(clean)
            full, clean = full[both], clean[both]
            if full.size:
                low = min(full.min(), clean.min())
                high = max(full.max(), clean.max())
                pad = 0.04 * (high - low or 1.0)
                axes.plot([low - pad, high + pad], [low - pad, high + pad], color=MUTED, lw=0.8)
                axes.scatter(full, clean, s=6, color=ACCENT, alpha=0.45, linewidths=0)
                # The same span on both axes, so that the line is the diagonal of the panel.
                axes.set_xlim(low - pad, high + pad)
                axes.set_ylim(low - pad, high + pad)
            axes.tick_params(labelsize=7.5)
            axes.set_title(panel, loc="left", fontsize=9.5, fontweight="bold")
            axes.set_xlabel(f"{feature}, whole recording", fontsize=8)
        row[0].set_ylabel(f"{feature}, clean windows", fontsize=8)
    return figure
