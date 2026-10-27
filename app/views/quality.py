"""EEG Signal Quality: the analysed window before and after cleaning, and how clean it is."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from neurolab import charts, engine
from neurolab import components as ui
from neurolab import theme as t

ui.page_header("quality")
found = ui.need_analysis("quality")
if found is None:
    st.stop()
config = engine.settings()
index = ui.pick_recording(found, "quality")
described = found.described[index]
log = described.log
sfreq = described.series.sfreq
channels = described.series.channels
total = described.cleaned_uv.shape[1] / sfreq
bad = log["bad_segments"]["segments"]

CLEANED, BEFORE, BOTH = "Cleaned", "Before cleaning", "Both"
COLOURS = {CLEANED: "#22aacb", BEFORE: "#8a97ad"}
ALL, CUSTOM = "All channels", "Custom"


def nice(value: float) -> float:
    """``value`` rounded up to 1, 2 or 5 times a power of ten."""
    if not np.isfinite(value) or value <= 0:
        return 50.0
    power = 10 ** np.floor(np.log10(value))
    return float(next(m * power for m in (1, 2, 5, 10) if m * power >= value))


# --- the viewer ---------------------------------------------------------------------------------

ui.section("Signal viewer")
with st.container(key="nl-card-viewer"):
    top = st.columns([1.5, 1.4, 0.8, 0.9], gap="small", vertical_alignment="bottom")
    view = top[0].segmented_control(
        "Signal",
        [CLEANED, BEFORE, BOTH],
        default=CLEANED,
        required=True,
        key="nl-q-view",
        persist_state="session",
        help="Before cleaning: the same window, average-referenced only (no filter, no ICA).",
    )
    group = top[1].selectbox(
        "Channels",
        [ALL, *config.data.regions, CUSTOM],
        format_func=lambda g: g if g in (ALL, CUSTOM) else f"{g.capitalize()} region",
        key="nl-q-group",
        persist_state="session",
    )
    length = top[2].selectbox(
        "Window",
        [5, 10, 20, 30, 60],
        index=1,
        format_func=lambda s: f"{s} s",
        key="nl-q-length",
        persist_state="session",
    )
    spacing_choice = top[3].selectbox(
        "Row height",
        ["auto", 20, 50, 100, 200, 500],
        format_func=lambda v: "Auto" if v == "auto" else f"{v} µV",
        key="nl-q-spacing",
        persist_state="session",
        help="Microvolts between neighbouring channels' baselines.",
    )
    if group == CUSTOM:
        picks = st.multiselect(
            "Channels shown",
            channels,
            default=channels[:8],
            key="nl-q-picks",
            persist_state="session",
        )
    elif group == ALL:
        picks = list(channels)
    else:
        picks = [c for c in channels if c in config.data.regions[group]]
    latest = max(0.0, float(np.floor(total - length)))
    if st.session_state.get("nl-q-start", 0.0) > latest:
        st.session_state["nl-q-start"] = latest
    start = 0.0
    if latest > 0:
        start = st.slider(
            "Start of the window (s)",
            0.0,
            latest,
            step=0.5,
            key="nl-q-start",
            persist_state="session",
            format="%.1f s",
        )
    if not picks:
        st.info("Choose at least one channel.")
        st.stop()
    first = int(round(start * sfreq))
    last = min(described.cleaned_uv.shape[1], first + int(round(length * sfreq)))
    rows = [channels.index(c) for c in picks]

    def window(data: np.ndarray) -> np.ndarray:
        shown = data[rows, first:last].astype(float)
        return shown - shown.mean(axis=1, keepdims=True)  # display only: the offset

    traces = {}
    if view in (BEFORE, BOTH):
        traces[BEFORE] = window(described.before_uv)
    if view in (CLEANED, BOTH):
        traces[CLEANED] = window(described.cleaned_uv)
    main = traces[CLEANED] if CLEANED in traces else traces[BEFORE]
    spacing = (
        nice(5 * float(np.median(main.std(axis=1))))
        if spacing_choice == "auto"
        else float(spacing_choice)
    )
    colours = dict(COLOURS)
    if view == BOTH:
        colours[BEFORE] = t.CONTEXT
    keys = [(name, colours[name], "line") for name in traces]
    if bad:
        keys.append(("Marked bad", "rgba(245, 181, 68, 0.35)", "box"))
    ui.legend(keys)
    st.altair_chart(
        charts.waveform(traces, picks, sfreq, start, spacing, bad, colours),
        theme=None,
        width="stretch",
    )
    st.caption(
        f"{len(picks)} channel(s), {length} s from {start:g} s, {spacing:g} µV between rows. "
        "Before cleaning: the analysed window average-referenced only, no filter or ICA. Each "
        "trace's mean over the window is removed for display, and the traces are thinned to "
        "the screen keeping every minimum and maximum; the data analysed are not changed. "
        "Scroll on the chart to zoom in time."
    )
    st.altair_chart(
        charts.bad_track(bad, total, (start, start + length)), theme=None, width="stretch"
    )
    if bad:
        st.caption(
            f"The whole {total:g} s window: {len(bad)} bad stretch(es) in amber, the part on "
            "screen outlined. Hover a stretch for its reason."
        )
    else:
        st.caption(f"The whole {total:g} s window: no stretch was marked bad.")

# --- spectrum and channels ------------------------------------------------------------------------

st.space("small")
left, right = st.columns([1, 1], gap="medium")
with left:
    ui.section("Spectrum before and after cleaning")
    with st.container(key="nl-card-psd"):
        psd = log["psd"]
        line = log["line_noise"]["detected_hz"]
        st.altair_chart(
            charts.spectrum_before_after(
                np.asarray(psd["freqs_hz"]),
                np.asarray(psd["before_db"]),
                np.asarray(psd["after_db"]),
                (config.preprocessing.filter.l_freq, config.preprocessing.filter.h_freq),
                line,
            ),
            theme=None,
            width="stretch",
        )
        st.caption(
            f"Median over the {log['n_channels']} channels (Welch). Before: cropped and "
            "re-referenced only. Lines mark the band kept"
            + (f" and the mains peak found at {line:g} Hz" if line else "; no mains peak found")
            + "."
        )
with right:
    ui.section("Channels")
    with st.container(key="nl-card-channels"):
        frame = pd.DataFrame(
            {
                "channel": list(log["channels"]),
                "sd": [c["sd_uv"] for c in log["channels"].values()],
                "bad": [c["peak_percent"] + c["flat_percent"] for c in log["channels"].values()],
            }
        )
        st.altair_chart(
            charts.channel_columns(frame, "sd", "SD after cleaning (µV)", ".2f"),
            theme=None,
            width="stretch",
        )
        st.altair_chart(
            charts.channel_columns(frame, "bad", "Time this channel made bad (%)", ".1f", t.AMBER),
            theme=None,
            width="stretch",
        )
        st.caption(
            "A window is bad when any channel crosses a limit; the lower chart shows which "
            "channels did (peak-to-peak too large or too small), as a share of the time."
        )

if described.qc:
    with st.expander("The pipeline's own QC figures", icon=":material/image:"):
        st.caption(
            "Drawn by the cleaning step itself (pdeeg.preprocessing.qc), as for every training "
            "recording: the spectrum before and after, and the scalp maps of the components "
            "ICA removed."
        )
        for kind in ("psd", "ica"):
            if kind in described.qc:
                st.image(described.qc[kind], width="stretch")

ui.glossary(["Sampling frequency", "ICA artifact removal", "Bad windows"])
