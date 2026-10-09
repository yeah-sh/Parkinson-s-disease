"""Look shared by every figure: colours, fonts, and a figure with a title block.

Figures are built with matplotlib's object interface and saved as PNG; nothing here opens a
window or changes the global matplotlib state.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

DPI = 150
PX = DPI / 96  # device pixels per CSS pixel, for mark sizes given in CSS pixels

# Chart chrome and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# One accent for the series a chart is about; grey for context.
ACCENT = "#2a78d6"
CONTEXT = "#c3c2b7"

_FONTS = ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans")
# Inches reserved at the top of every figure for its title and subtitle.
HEADER = 0.58
_MARGIN = 0.1


@functools.cache
def _rc() -> dict[str, Any]:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in _FONTS if name in installed), "sans-serif")
    return {
        "font.family": family,
        "font.size": 9,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
    }


def styled(plot: Callable[..., Figure]) -> Callable[..., Figure]:
    """Decorator: draw the figure a function returns with the shared style."""

    @functools.wraps(plot)
    def wrapper(*args: Any, **kwargs: Any) -> Figure:
        with matplotlib.rc_context(_rc()):
            return plot(*args, **kwargs)

    return wrapper


def new_figure(width: float, height: float, title: str, subtitle: str) -> Figure:
    """A figure with its title block; the plots are laid out in the space below it."""
    figure = Figure(figsize=(width, height), dpi=DPI, layout="constrained")
    FigureCanvasAgg(figure)
    figure.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 1.0 - HEADER / height))
    x = _MARGIN / width
    figure.text(x, 1 - 0.1 / height, title, ha="left", va="top", fontsize=11, fontweight="bold")
    figure.text(x, 1 - 0.33 / height, subtitle, ha="left", va="top", color=INK_SECONDARY)
    return figure


def save(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
