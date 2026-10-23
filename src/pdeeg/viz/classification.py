"""Figures of the classification: balanced accuracy of every feature set and model."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from pdeeg.viz.style import (
    ACCENT,
    AXIS,
    CONTEXT,
    HEADER,
    INK_SECONDARY,
    SURFACE,
    new_figure,
    styled,
)

_ROW = 0.3  # inches per evaluation


@styled
def plot_ablation(
    table: pd.DataFrame,
    tasks: Mapping[str, str],
    labels: Mapping[str, str],
    title: str,
    subtitle: str,
) -> Figure:
    """Balanced accuracy with its interval, one row per feature set and model, a panel per task.

    ``table`` is :func:`pdeeg.models.classify.summary_table`; ``tasks`` maps a task to its
    panel title and ``labels`` a row's (feature set, model) key ``"<set> / <model>"`` to its
    label. Baselines are grey. A tick marks the 95th percentile of the permutation null.
    """
    sizes = [int((table["task"] == task).sum()) for task in tasks]
    figure = new_figure(7.2, _ROW * sum(sizes) + 0.75 * len(sizes) + HEADER + 0.55, title, subtitle)
    grid = figure.subplots(
        len(sizes), 1, squeeze=False, sharex=True, gridspec_kw={"height_ratios": sizes}
    )[:, 0]
    for axes, (task, heading) in zip(grid, tasks.items(), strict=True):
        rows = table[table["task"] == task].reset_index(drop=True)
        positions = np.arange(len(rows))[::-1]
        for position, row in zip(positions, rows.to_dict("records"), strict=True):
            colour = CONTEXT if row["baseline"] else ACCENT
            axes.plot(
                [row["balanced_accuracy_low"], row["balanced_accuracy_high"]],
                [position, position],
                color=colour,
                lw=2,
            )
            axes.plot(
                row["balanced_accuracy"],
                position,
                "o",
                color=colour,
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
            axes.plot(row["null_ba_95"], position, "|", color=INK_SECONDARY, markersize=9)
        axes.axvline(0.5, color=AXIS, lw=1, zorder=0)
        axes.set_yticks(
            positions,
            [labels[f"{row['feature_set']} / {row['model']}"] for row in rows.to_dict("records")],
        )
        axes.set_ylim(-0.7, len(rows) - 0.3)
        axes.grid(axis="y", visible=False)
        axes.tick_params(labelsize=8)
        axes.set_title(heading, loc="left", fontsize=9.5, fontweight="bold")
    grid[-1].set_xlim(0.0, 1.0)
    grid[-1].set_xlabel("Balanced accuracy, 95 % bootstrap interval over participants")
    return figure
