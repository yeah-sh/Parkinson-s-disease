"""The building blocks of every page: header, chips, tiles, steps, checklists, empty states.

Text that comes from a file (its name) or from pdeeg (a message) is escaped before it goes
into HTML or Markdown.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Any

import mne
import streamlit as st

from neurolab import engine
from neurolab.glossary import TERMS
from neurolab.pages import PAGES

# Text glyphs: st.html keeps them (it strips inline SVG), and they take the text colour.
_ICONS = {
    "check": "✓",
    "note": "!",
    "x": "✕",
    "pending": "–",
    "dot": "●",
    "alert": "!",
    "shield": "✓",
    "running": "",
}
_STEP_ICONS = {
    "done": "check",
    "failed": "x",
    "attention": "note",
    "ready": "dot",
    "running": "running",
}
_CHECK_ICONS = {"passed": "check", "note": "note", "failed": "x", "pending": "pending"}


def icon(name: str) -> str:
    return f'<span class="nl-glyph" aria-hidden="true">{_ICONS[name]}</span>'


def md(text: str) -> str:
    """``text`` escaped for Markdown."""
    for character in "\\*_$[]`<>#":
        text = text.replace(character, f"\\{character}")
    return text


def chip(text: str, kind: str = "mute", *, title: str | None = None) -> str:
    tip = f' title="{escape(title)}"' if title else ""
    return f'<span class="nl-chip {kind}"{tip}><span class="nl-dot"></span>{escape(text)}</span>'


def section(label: str) -> None:
    st.html(f'<div class="nl-section">{escape(label)}</div>')


def size(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:.1f} MB" if n_bytes >= 1e5 else f"{n_bytes / 1e3:.0f} kB"


def duration(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60.0)
    return f"{int(minutes)} min {rest:.0f} s" if minutes else f"{seconds:.1f} s"


# --- shell ----------------------------------------------------------------------------------------


def page_header(page: str) -> None:
    """The page's title and lede, the recording and its status, and the page actions."""
    info = PAGES[page]
    current = engine.files()
    status, kind = engine.headline()
    chips = []
    if current:
        names = " + ".join(name for name, _ in current)
        chips.append(chip(names, "run", title="Recording(s) loaded in this session"))
    chips.append(chip(status, kind))
    chips.append(
        chip(
            "Research mode",
            "research",
            title="Research demonstration on one dataset. Not a diagnostic tool.",
        )
    )
    with st.container(key="nl-header"):
        left, right = st.columns([5, 4], vertical_alignment="bottom", gap="medium")
        left.html(
            '<p class="nl-eyebrow">NeuroLab · Parkinson\'s EEG research workbench</p>'
            f'<h1 class="nl-title">{escape(info.title)}</h1>'
            f'<p class="nl-lede">{escape(info.lede).replace("`", "")}</p>'
        )
        with right:
            st.html(f'<div class="nl-chips">{"".join(chips)}</div>')
            with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
                if current:
                    st.button(
                        "Clear recording",
                        key=f"nl-clear-{page}",
                        icon=":material/close:",
                        type="tertiary",
                        on_click=engine.clear,
                    )
                if page != "report" and engine.analysis() is not None:
                    st.page_link(
                        PAGES["report"].path, label="Export report", icon=":material/download:"
                    )
    activity(expanded=page in ("overview", "recording"))


def sidebar() -> None:
    """Under the navigation: the recording in this session, and the disclaimer."""
    with st.sidebar:
        current = engine.files()
        if current:
            status, kind = engine.headline()
            names = "".join(
                f'<div class="name" style="font-size:0.82rem">{escape(name)}</div>'
                for name, _ in current
            )
            st.html(
                '<div class="nl-section" style="margin-top:0.4rem">Recording</div>'
                f'<div class="nl-file" style="grid-template-columns:1fr">{names}</div>'
                f'<div style="margin-top:6px">{chip(status, kind)}</div>'
            )
        with st.container(key="nl-disclaimer"):
            st.markdown(engine.disclaimer())
        who = engine.trained()
        if who is not None:
            trained = engine.evidence().get("stage1", {}).get("trained_at", "")[:10]
            st.caption(
                f"Models: OpenNeuro {who['dataset']['id']} {who['dataset']['version']}, "
                f"{who['participants']} people, {who['recordings']} recordings"
                + (f", trained {trained}" if trained else "")
                + "."
            )


# --- blocks ---------------------------------------------------------------------------------------


def tiles(items: Sequence[dict[str, Any]]) -> None:
    """Stat tiles: ``label``, ``value`` (None shows a dash), ``sub``, and ``warn``."""
    cells = []
    for item in items:
        value = item.get("value")
        shown = "—" if value is None else escape(str(value))
        cls = "v" if value is not None else "v empty"
        cells.append(
            f'<div class="nl-tile{" warn" if item.get("warn") else ""}">'
            f'<div class="k">{escape(item["label"])}</div><div class="{cls}">{shown}</div>'
            f'<div class="s">{escape(item.get("sub", ""))}</div></div>'
        )
    st.html(f'<div class="nl-tiles">{"".join(cells)}</div>')


def steps(items: Sequence[engine.Step], *, row: bool = False) -> None:
    cells = []
    for number, item in enumerate(items, start=1):
        mark = icon(_STEP_ICONS[item.state]) if item.state in _STEP_ICONS else str(number)
        if row:
            cells.append(
                f'<li class="nl-step {item.state}"><div class="ic">{mark}</div>'
                f'<div class="t">{escape(item.title)}</div>'
                f'<div class="m">{escape(item.detail)}</div></li>'
            )
        else:
            cells.append(
                f'<li class="nl-step {item.state}"><div class="ic">{mark}</div>'
                f'<div><div class="t">{escape(item.title)}</div>'
                f'<div class="d">{escape(item.detail)}</div></div>'
                f'<div class="m">{escape(_STATE_WORDS[item.state])}</div></li>'
            )
    st.html(f'<ol class="nl-steps{" row" if row else ""}">{"".join(cells)}</ol>')


_STATE_WORDS = {
    "done": "done",
    "failed": "failed",
    "attention": "see note",
    "ready": "next",
    "running": "running",
    "pending": "",
}


def kv(rows: Sequence[tuple[str, str]]) -> None:
    body = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows)
    st.html(f'<dl class="nl-kv">{body}</dl>')


def empty(title: str, body: str, *, link: str | None = None, label: str = "") -> None:
    st.html(
        f'<div class="nl-empty"><div class="t">{escape(title)}</div>'
        f'<div class="d">{escape(body)}</div></div>'
    )
    if link:
        with st.container(horizontal=True, horizontal_alignment="center"):
            st.page_link(PAGES[link].path, label=label or PAGES[link].title, icon=PAGES[link].icon)


def verdict(strong: bool, title: str, body: str) -> None:
    """How a model did against chance: amber and prominent when it was not better."""
    if strong:
        st.info(f"**{title}.** {md(body)}", icon=":material/verified:")
    else:
        st.warning(f"**{title}.** {md(body)}", icon=":material/warning:")


def legend(items: Sequence[tuple[str, str, str]]) -> None:
    """``(label, colour, kind)`` with kind ``box`` or ``line``."""
    spans = "".join(
        f'<span><i class="{kind}" style="background:{colour}"></i>{escape(label)}</span>'
        for label, colour, kind in items
    )
    st.html(f'<div class="nl-legend">{spans}</div>')


def glossary(terms: Sequence[str], *, label: str = "Terms on this page") -> None:
    with st.expander(label, icon=":material/school:"):
        for term in terms:
            st.markdown(f"**{term}.** {TERMS[term]}")


# --- the input ------------------------------------------------------------------------------------


def requirements() -> dict[str, str]:
    """What each check asks of a file, from the configuration."""
    config = engine.settings()
    montage = config.data.dataset.montage
    n = len(mne.channels.make_standard_montage(montage).ch_names)
    crop = config.preprocessing.crop
    low, high = config.predict.amplitude_uv
    aliases = ", ".join(f"{a}→{b}" for a, b in config.predict.channel_aliases.items())
    start = f"trigger “{crop.start_event}”" if crop.start_event is not None else "the file start"
    return {
        "file": "A .bdf or .edf file that can be read as EEG",
        "channels": f"All {n} channels of the {montage} montage (older names accepted: {aliases})",
        "sampling rate": f"{config.predict.sfreq:g} Hz, as in training; other rates are "
        "resampled, with a note",
        "duration": f"At least {crop.duration:g} s of EEG from {start}, or from the file start "
        "without it",
        "amplitude": f"Median channel amplitude (1–40 Hz) within {low:g}–{high:g} µV",
    }


def fixes() -> dict[str, str]:
    """What to do when a check fails. Advice only: the checks themselves are pdeeg's."""
    config = engine.settings()
    return {
        "file": "Export the recording as BDF or EDF (for example with mne.export.export_raw) "
        "and upload that file.",
        "channels": "Name the channels as the montage does, or include the missing electrodes. "
        "An 'EEG ' prefix, a '-REF' suffix and the older names listed are understood.",
        "duration": f"Upload a longer recording: the analysis needs "
        f"{config.preprocessing.crop.duration:g} s of continuous EEG.",
        "amplitude": "Check the physical units in the file header: values this far off usually "
        "mean volts were stored as microvolts, or the reverse.",
        "sampling rate": "Nothing to do: the file was resampled, but its features may differ "
        "from those of a recording made at the training rate.",
    }


CHECK_ORDER = ("file", "channels", "sampling rate", "duration", "amplitude")


def checklist(report: dict[str, Any]) -> None:
    """Every check of one file, in order: passed, noted, failed, or not reached."""
    needs, fix = requirements(), fixes()
    done = {check["name"]: check for check in report["checks"]}
    rows = []
    for name in CHECK_ORDER:
        check = done.get(name)
        status = check["status"] if check else "pending"
        detail = check["detail"] if check else "Not checked: an earlier check failed."
        advice = ""
        if status == "failed" or (status == "note" and name == "sampling rate"):
            advice = f'<div class="fix"><b>How to fix:</b> {escape(fix[name])}</div>'
        rows.append(
            f'<div class="nl-check {status}"><div class="ic">{icon(_CHECK_ICONS[status])}</div>'
            f'<div><div class="n">{escape(name)}</div><div class="r">{escape(needs[name])}</div>'
            f'</div><div><div class="x">{escape(detail)}</div>{advice}</div></div>'
        )
    st.html(f'<div class="nl-checks">{"".join(rows)}</div>')


def file_card(report: dict[str, Any]) -> None:
    header = report.get("header")
    meta = [report["format"], size(report["size"])]
    if header:
        meta += [
            f"{header['n_channels']} channels in file",
            f"{header['sfreq']:g} Hz",
            duration(header["duration_s"]),
        ]
    status = chip("Checks passed", "ok") if report["error"] is None else chip("Failed", "err")
    notes = sum(c["status"] == "note" for c in report["checks"])
    if report["error"] is None and notes:
        status = chip(f"Passed · {notes} note(s)", "warn")
    st.html(
        f'<div class="nl-file"><div class="badge">{escape(report["format"])}</div>'
        f'<div><div class="name">{escape(report["name"])}</div>'
        f'<div class="meta">{escape(" · ".join(meta))}</div></div><div>{status}</div></div>'
    )


# --- the analysis ---------------------------------------------------------------------------------


def run_button(where: str) -> None:
    """The button that runs the analysis of a checked upload, and the run itself."""
    current = engine.files()
    checks = engine.validation()
    if current is None or not engine.usable(checks) or engine.analysis() is not None:
        return
    if engine.running():
        return
    blocked = None
    if len(current) == 2 and not st.session_state.get(engine.SAME_PATIENT):
        blocked = "Confirm that both recordings are from the same patient first."
    if st.button(
        "Run analysis",
        key=f"nl-run-{where}",
        type="primary",
        icon=":material/play_arrow:",
        disabled=blocked is not None,
        help=blocked
        or "Cleaning (with ICA), MFDFA and band power, then the models: a few minutes. It "
        "runs in the background, so you can keep using the app.",
    ):
        engine.start_analysis(engine.key(), current)
        st.rerun()


@st.fragment(run_every=2)
def _progress(expanded: bool) -> None:
    """The running analysis, refreshed every 2 s; the whole page reruns when it is done."""
    work = engine.job()
    if work is None or work.done:
        st.rerun()
    # Not used as a context manager: leaving the block would mark the status complete.
    box = st.status(
        f"Analysing the recording · {work.elapsed:.0f} s", expanded=expanded, state="running"
    )
    for seconds, message in list(work.messages):
        box.markdown(f"`{seconds:4.0f} s` {md(message)}")
    box.caption(
        "Running in the background: the pages stay usable, and this refreshes until the "
        "analysis is done."
    )


def activity(expanded: bool) -> None:
    """The analysis of the current upload while it runs, or why it failed."""
    work = engine.job()
    if work is None:
        return
    if not work.done:
        _progress(expanded)
    elif work.crashed and engine.analysis() is None:
        st.error(f"**The analysis failed unexpectedly.** {md(work.crashed)}")
        st.button(
            "Try again",
            key="nl-retry",
            icon=":material/refresh:",
            on_click=engine.forget_job,
            args=(work.digest,),
        )


def need_analysis(where: str) -> engine.Analysis | None:
    """The analysis of the current upload, or an empty state that says how to get one."""
    found = engine.analysis()
    if found is not None:
        if not any(d is not None for d in found.described):
            st.error(f"**The recording could not be cleaned.** {md(found.error or '')}")
            return None
        return found
    if engine.running():
        empty(
            "The analysis is running",
            "Cleaning, MFDFA and the models take a few minutes. The figures appear here as "
            "soon as it is done; its progress is shown above.",
        )
        return None
    if engine.files() is None:
        empty(
            "No recording loaded",
            "Upload a .bdf or .edf recording on the Overview page. It is checked first; the "
            "figures here appear once it has been analysed.",
            link="overview",
            label="Go to the upload",
        )
    elif not engine.usable(engine.validation()):
        empty(
            "The recording did not pass its checks",
            "Nothing is computed from a file that fails a check. The Recording Analysis page "
            "says which one failed and how to fix it.",
            link="recording",
            label="See the checks",
        )
    else:
        empty(
            "Ready to analyse",
            "The recording passed its checks. Run the analysis to clean it, compute its "
            "features and apply the models; the figures appear here afterwards.",
        )
        with st.container(horizontal=True, horizontal_alignment="center"):
            run_button(where)
        if len(engine.files()) == 2 and not st.session_state.get(engine.SAME_PATIENT):
            st.caption("Two recordings: confirm on the Overview page that they are one patient's.")
    return None


def pick_recording(found: engine.Analysis, where: str) -> int:
    """Which of two recordings a page shows (0 with one)."""
    names = [name for name, _ in engine.files()]
    if len(names) < 2:
        return 0
    index = st.segmented_control(
        "Recording",
        list(range(len(names))),
        format_func=lambda i: names[i],
        default=0,
        required=True,
        key=f"nl-pick-{where}",
        persist_state="session",
    )
    if found.described[index] is None:
        st.info("This recording could not be cleaned, so it is not described.")
        st.stop()
    return index
