"""What the app shows of a prediction: a probability bar, an interval against chance, and the
features that moved the prediction.

The two bars are HTML for the page, drawn with translucent greys and the text colour so that
they read on a light or a dark theme. Nothing here computes a prediction: every number comes
from :func:`pdeeg.predict.predictor.predict_files`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from pdeeg.viz.cascade import SECOND
from pdeeg.viz.style import ACCENT, AXIS, CONTEXT, HEADER, INK_SECONDARY, new_figure, styled

# A contribution pulls towards the positive class (ACCENT) or the negative one (SECOND): the
# first two slots of the categorical palette, one per class.
_TRACK = "rgba(128, 128, 128, 0.2)"
_HATCH = "rgba(128, 128, 128, 0.6)"
# A stage that is not better than chance is drawn in grey.
_GREYED = "rgba(128, 128, 128, 0.55)"
_ROW = 0.34  # inches per feature


def _percent(x: float) -> str:
    return f"{100.0 * min(max(float(x), 0.0), 1.0):.2f}%"


def probability_bar(
    probability: float,
    threshold: float,
    band: Sequence[float],
    ends: tuple[str, str],
    *,
    greyed: bool = False,
) -> str:
    """A bar from 0 to 1 filled up to ``probability``, a tick at ``threshold``, and the
    uncertain ``band`` hatched. ``ends`` names the class at 0 and the class at 1."""
    low, high = band
    fill = _GREYED if greyed else ACCENT
    hatch = f"repeating-linear-gradient(135deg, {_HATCH} 0 2px, transparent 2px 6px)"
    left, right = (escape(end) for end in ends)
    label = (
        f"probability {probability:.2f} of {ends[1]}; threshold {threshold:g}; "
        f"no label from {low:g} to {high:g}"
    )
    return (
        '<div style="margin:0.2rem 0 0.4rem 0">'
        f'<div role="img" aria-label="{escape(label)}" style="position:relative;height:16px;'
        f'border-radius:8px;background:{_TRACK};overflow:hidden">'
        '<div style="position:absolute;left:0;top:0;bottom:0;'
        f'width:{_percent(probability)};background:{fill}"></div>'
        f'<div style="position:absolute;top:0;bottom:0;left:{_percent(low)};'
        f'width:{_percent(high - low)};background:{hatch}"></div>'
        f'<div style="position:absolute;top:0;bottom:0;left:calc({_percent(threshold)} - 1px);'
        'width:2px;background:currentColor;opacity:0.7"></div>'
        "</div>"
        '<div style="display:flex;justify-content:space-between;font-size:0.8rem;opacity:0.75;'
        f'margin-top:2px"><span>0 · {left}</span><span>{right} · 1</span></div>'
        "</div>"
    )


def interval_bar(
    value: float, low: float, high: float, chance: float, *, greyed: bool = False
) -> str:
    """A line from 0 to 1 with the 95 % interval as a band, the estimate as a dot, and a tick
    at ``chance``."""
    colour = _GREYED if greyed else ACCENT
    label = f"{value:.2f}, 95 % interval {low:.2f} to {high:.2f}; chance {chance:g}"
    return (
        f'<div role="img" aria-label="{escape(label)}" style="position:relative;height:22px;'
        'margin-top:0.2rem">'
        f'<div style="position:absolute;left:0;right:0;top:10px;height:2px;background:{_TRACK}">'
        "</div>"
        f'<div style="position:absolute;top:2px;height:18px;left:calc({_percent(chance)} - 1px);'
        'width:2px;background:currentColor;opacity:0.6"></div>'
        f'<div style="position:absolute;top:8px;height:6px;border-radius:3px;'
        f'left:{_percent(low)};width:{_percent(high - low)};background:{colour};opacity:0.45">'
        "</div>"
        '<div style="position:absolute;top:5px;width:12px;height:12px;border-radius:50%;'
        f'left:calc({_percent(value)} - 6px);background:{colour}"></div>'
        "</div>"
        '<div style="position:relative;height:1.1rem;font-size:0.8rem;opacity:0.75">'
        '<span style="position:absolute;left:0">0</span>'
        f'<span style="position:absolute;left:{_percent(chance)};transform:translateX(-50%)">'
        f"chance {chance:g}</span>"
        '<span style="position:absolute;right:0">1</span>'
        "</div>"
    )


@styled
def plot_contributions(
    top: Mapping[str, Any], ends: tuple[str, str], title: str, *, greyed: bool = False
) -> Figure:
    """The features that moved one prediction most, as ``top_features`` of a stage's result.

    For a linear model, signed bars: towards ``ends[1]`` (the positive class) to the right,
    towards ``ends[0]`` to the left. For another model, its global importances.
    """
    features = top["features"]
    signed = "contribution" in features[0]
    values = np.array([f["contribution"] if signed else f["importance"] for f in features])
    names = [f["feature"].replace("_", " ") for f in features]
    n = len(values)
    if signed:
        subtitle = f"The {n} largest: coefficient × standardised feature value."
        colours = [ACCENT if v > 0 else SECOND for v in values]
    else:
        subtitle = f"The {n} largest impurity importances; the same for every recording."
        colours = [ACCENT] * n
    if greyed:
        colours = [CONTEXT] * n
        subtitle += " Grey: the model is not better than chance."
    figure = new_figure(7.2, _ROW * n + 1.1 + HEADER, title, subtitle)
    axes = figure.subplots()
    y = np.arange(n)[::-1]
    axes.barh(y, values, height=0.6, color=colours)
    axes.set_yticks(y, names)
    axes.tick_params(axis="y", length=0)
    axes.grid(axis="y", visible=False)
    axes.axvline(0.0, color=AXIS, lw=0.8)
    limit = float(np.abs(values).max()) * 1.35 or 1.0
    for yi, value in zip(y, values, strict=True):
        axes.text(
            value,
            yi,
            f" {value:+.2f} " if signed else f" {value:.3f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    if signed:
        axes.set_xlim(-limit, limit)
        axes.set_xlabel(
            f"← towards {ends[0]}      contribution to the score      towards {ends[1]} →"
        )
        if not greyed:
            axes.legend(
                handles=[
                    Patch(color=ACCENT, label=f"towards {ends[1]}"),
                    Patch(color=SECOND, label=f"towards {ends[0]}"),
                ],
                loc="lower left",
                bbox_to_anchor=(0.0, 1.0),
                ncols=2,
                fontsize=8,
                borderaxespad=0.2,
            )
    else:
        axes.set_xlim(0.0, limit)
        axes.set_xlabel("Importance")
    return figure
