"""The page's HTML pieces: hero, input chips, steps, verdict cards, notes and footer.

Only layout: the words of a verdict come from restwave.verdict, those of a check from
restwave.plain. Uploaded names and pdeeg's messages are escaped. st.html removes inline SVG, so
every drawing is an image (an SVG data URI) with its colours set here.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Sequence
from html import escape

from restwave import theme as t
from restwave.verdict import CLEAR, DETECTED, INCONCLUSIVE, NONE, Call

# --- drawings as images -------------------------------------------------------------------------


def image(svg: str, alt: str = "", width: float | None = None, height: float | None = None) -> str:
    """An <img> of ``svg``; ``alt`` empty marks it as decoration."""
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    size = "".join(
        f' {name}="{value:g}"' for name, value in (("width", width), ("height", height)) if value
    )
    hidden = "" if alt else ' aria-hidden="true"'
    return f'<img src="data:image/svg+xml;base64,{data}" alt="{escape(alt)}"{size}{hidden}>'


def _svg(body: str, colour: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{colour}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
        f"{body.replace('currentColor', colour)}</svg>"
    )


_PILL = '<rect x="3.5" y="8.5" width="17" height="7" rx="3.5" transform="rotate(-35 12 12)"/>'
ICONS = {
    "check": '<path d="M5 12.5l4.2 4.2L19 7"/>',
    "cross": '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    "dash": '<path d="M7 12h10"/>',
    "attention": '<path d="M12 6.5v7"/><circle cx="12" cy="17.6" r="0.6" fill="currentColor"/>',
    "question": '<path d="M9.2 9.2a2.9 2.9 0 1 1 4.3 2.5c-.9.5-1.5 1.1-1.5 2.1v.6"/>'
    '<circle cx="12" cy="17.8" r="0.6" fill="currentColor"/>',
    "pill": f'{_PILL}<path d="M10 9.3l4 5.4"/>',
    "pill-off": f'{_PILL}<path d="M4 4l16 16"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5"/>'
    '<circle cx="12" cy="7.6" r="0.6" fill="currentColor"/>',
    "flask": '<path d="M9.5 3.5h5M10.5 3.5v5.2L5.6 17.4A2 2 0 0 0 7.4 20.5h9.2a2 2 0 0 0 1.8-3.1'
    'l-4.9-8.7V3.5"/><path d="M7.8 14.5h8.4"/>',
}


def icon(name: str, size: int = 24, colour: str = t.ON_INK, label: str = "") -> str:
    return image(_svg(ICONS[name], colour), label, size, size)


LOGO = image(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 30">'
    f'<circle cx="15" cy="15" r="13.5" fill="none" stroke="{t.TEAL}" stroke-width="1.6"/>'
    f'<path d="M4.5 15.5h4l2-5 3 10 2.6-12 2.4 9 1.7-3.5h5.3" fill="none" stroke="{t.TEAL}" '
    'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    width=30,
    height=30,
)

# --- the hero --------------------------------------------------------------------------------


def _wave_path(width: float = 1200.0, height: float = 56.0) -> str:
    """A calm EEG-like line, periodic over half its width so that it can scroll for ever."""
    period = width / 2
    points = []
    for i in range(0, int(width) + 1, 3):
        x = float(i)
        phase = 2 * math.pi * x / period
        y = (
            0.42 * math.sin(3 * phase)
            + 0.22 * math.sin(7 * phase + 1.3)
            + 0.18 * math.sin(13 * phase + 0.4)
            + 0.09 * math.sin(29 * phase + 2.1)
            + 0.35 * math.exp(-((((x % period) - 0.62 * period) / 9.0) ** 2))
        )
        points.append(f"{x:.0f},{height / 2 - y * height * 0.5:.1f}")
    return "M" + " L".join(points)


def brand() -> str:
    tag = '<span class="rw-tag">Research demo · not a medical diagnosis</span>'
    return f'<div class="rw-brand"><span class="rw-logo">{LOGO}Restwave</span>{tag}</div>'


def hero() -> str:
    return (
        f'<div class="rw-hero">{brand()}'
        "<h1>Is there a Parkinson's pattern in this brain signal?</h1>"
        '<p class="rw-lede">Upload a resting-state EEG recording. We analyse its brain-wave '
        "patterns and tell you whether they resemble Parkinson's disease, and if so, whether "
        "the person appears to be on or off medication.</p>"
        f'<div class="rw-wave" aria-hidden="true">{_wave()}</div></div>'
    )


def _wave() -> str:
    return image(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 56" preserveAspectRatio="none">'
        f'<path d="{_wave_path()}" fill="none" stroke="{t.TEAL}" stroke-width="1.6" '
        'stroke-linejoin="round" vector-effect="non-scaling-stroke" opacity="0.85"/></svg>'
    )


def drop_hint() -> str:
    return (
        '<p class="rw-drop-hint">For one patient’s OFF- and ON-medication recordings, drop both '
        "files together. Files stay on this computer.</p>"
    )


def files(names: Sequence[str]) -> str:
    """The files being looked at, A and B for a pair."""
    letters = "AB" if len(names) == 2 else ""
    pills = "".join(
        f'<span class="rw-file">{f"<b>{letters[i]}</b>" if letters else ""}{escape(name)}</span>'
        for i, name in enumerate(names)
    )
    return f'<div class="rw-files">{pills}</div>'


# --- the input check --------------------------------------------------------------------------


def chips(items: Sequence[tuple[str, str, str]], which: str | None = None) -> str:
    """One row of (status, title, words) chips; ``which`` names the recording in a pair."""
    marks = {"passed": "check", "note": "check", "failed": "cross", "pending": "dash"}
    colours = {"passed": t.TEAL, "note": t.TEAL, "failed": t.RED_ON_INK, "pending": t.ON_INK_3}
    words = {"passed": "passed", "note": "passed", "failed": "failed", "pending": "not checked"}
    parts = [f'<span class="rw-of">{escape(which)}</span>'] if which else []
    for status, title, text in items:
        state = "passed" if status == "note" else status
        parts.append(
            f'<span class="rw-chip {state}">'
            f"{icon(marks[status], 18, colours[status], words[status])}"
            f'<span class="k">{escape(title)}</span>{escape(text)}</span>'
        )
    return f'<div class="rw-chips" role="list">{"".join(parts)}</div>'


def stop(title: str, fix: str, eyebrow: str = "Can't analyse this file") -> str:
    return (
        f'<section class="rw-card rw-stop" role="alert"><p class="rw-eyebrow">{escape(eyebrow)}</p>'
        f'<h2 class="rw-headline">{escape(title)}</h2>'
        f'<p class="fix"><b>How to fix it:</b> {escape(fix)}</p></section>'
    )


def banner(text: str, kind: str = "info", strong: str | None = None) -> str:
    lead = f"<b>{escape(strong)}</b> " if strong else ""
    return f'<div class="rw-banner">{icon(kind, 20, t.TEAL)}<div>{lead}{escape(text)}</div></div>'


# --- processing ---------------------------------------------------------------------------------


def steps(titles: Sequence[str], current: int, detail: str) -> str:
    items = []
    for i, title in enumerate(titles):
        state = "done" if i < current else "active" if i == current else "pending"
        mark = (
            icon("check", 15, t.INK)
            if state == "done"
            else ("" if state == "active" else str(i + 1))
        )
        spoken = {"done": "done", "active": "in progress", "pending": "to do"}[state]
        items.append(
            f'<li class="{state}"><span class="dot">{mark}</span>{escape(title)}'
            f'<span class="sr-only" style="position:absolute;left:-9999px">({spoken})</span></li>'
        )
    return (
        f'<div class="rw-steps" role="status" aria-live="polite"><ol>{"".join(items)}</ol>'
        f'<p class="now">{escape(detail)}…</p></div>'
    )


# --- the verdict --------------------------------------------------------------------------------

_VERDICT_ICONS = {
    "stage1": {DETECTED: "attention", CLEAR: "check", INCONCLUSIVE: "question"},
    "stage2_single": {DETECTED: "pill-off", CLEAR: "pill", INCONCLUSIVE: "question"},
    "stage2_paired": {DETECTED: "pill-off", CLEAR: "pill", INCONCLUSIVE: "question"},
}


_SAY = {DETECTED: t.AMBER_TEXT, CLEAR: t.SAGE_TEXT, INCONCLUSIVE: t.SLATE_TEXT, NONE: t.SLATE_TEXT}


def _icon_for(call: Call) -> str:
    stage = call.entry.get("stage", "stage1")
    name = _VERDICT_ICONS.get(stage, {}).get(call.kind, "dash")
    return f'<div class="rw-icon">{icon(name, 34, _SAY[call.kind])}</div>'


def _trust(call: Call) -> str:
    if call.trust is None:
        return ""
    return (
        f'<p class="rw-trust {call.trust.level}"><b>How much to trust this: {call.trust.level}.</b>'
        f" {escape(call.trust.line)}</p>"
    )


def _head(call: Call, which: str | None = None) -> str:
    label = f'<p class="rw-which">{escape(which)}</p>' if which else ""
    return (
        f'<div class="rw-verdict-head">{_icon_for(call)}<div>{label}'
        f'<p class="rw-eyebrow">{escape(call.eyebrow)}</p>'
        f'<h2 class="rw-headline">{escape(call.headline)}</h2>'
        f'<p class="rw-body">{escape(call.body)}</p></div></div>'
    )


def verdict(call: Call, gauge: str, *, small: bool = False) -> str:
    """The card of one stage's call: icon and words, the gauge, and how much to trust it."""
    note = f'<p class="rw-note">{escape(call.note)}</p>' if call.note else ""
    size = " small" if small else ""
    return (
        f'<section class="rw-card rw-verdict {call.kind}{size}">'
        f'<div class="rw-verdict-grid"><div>{_head(call)}{note}</div>{gauge}</div>'
        f"{_trust(call)}</section>"
    )


def verdict_pair(calls: Sequence[Call], gauges: Sequence[str], names: Sequence[str]) -> str:
    """Stage 1 for both recordings of a pair, side by side, with one trust line."""
    kind = calls[0].kind if calls[0].kind == calls[1].kind else INCONCLUSIVE
    columns = "".join(
        f'<div class="rw-verdict {call.kind}" style="padding:0">'
        f'<div class="rw-verdict-grid"><div>{_head(call, f"Recording {letter} · {name}")}</div>'
        f"{gauge}</div></div>"
        for call, gauge, letter, name in zip(calls, gauges, "AB", names, strict=True)
    )
    return (
        f'<section class="rw-card rw-verdict {kind}"><div class="rw-pair">{columns}</div>'
        f"{_trust(calls[0])}</section>"
    )


# --- cards and footer -----------------------------------------------------------------------------


def card_head(title: str, caption: str) -> str:
    """A graph card's title and its one sentence; ``caption`` may hold <b> (it is ours)."""
    return f'<h3 class="rw-title">{escape(title)}</h3><p class="rw-caption">{caption}</p>'


def legend(items: Sequence[tuple[str, str, str]]) -> str:
    """(mark: bar, dot, diamond, line or dash; colour; words) per item."""
    return (
        '<div class="rw-legend">'
        + "".join(
            f'<span><i class="{mark}" style="background:{colour};border-color:{colour}"></i>'
            f"{escape(words)}</span>"
            for mark, colour, words in items
        )
        + "</div>"
    )


def missing(text: str) -> str:
    return f'<p class="rw-missing">{escape(text)}</p>'


def section(title: str, lede: str) -> str:
    return (
        f'<h2 class="rw-section">{escape(title)}</h2><p class="rw-section-lede">{escape(lede)}</p>'
    )


def footer(research_url: str, trained: str) -> str:
    return (
        '<div class="rw-footer">'
        f'<p style="margin:0 0 .5rem 0"><a href="{escape(research_url)}" target="_blank" '
        'rel="noopener">Open the research view ↗</a> for every check, figure and number '
        "(start it with <code>make research</code>).</p>"
        f'<p style="margin:0">{escape(trained)}</p></div>'
    )
