"""Multifractal Analysis: the MFDFA of the cleaned recording, the data the models saw."""

from __future__ import annotations

import numpy as np
import streamlit as st

from neurolab import charts, engine, figures
from neurolab import components as ui
from neurolab.glossary import TERMS
from neurolab.pages import PAGES
from pdeeg import demo
from pdeeg.features.scaling import BROADBAND

ui.page_header("multifractal")
found = ui.need_analysis("multifractal")
if found is None:
    st.stop()
config = engine.settings()
mfdfa = config.mfdfa
index = ui.pick_recording(found, "multifractal")
described = found.described[index]
digest = engine.key()
SIGNED = ("delta_h", "asymmetry")  # features with a sign: maps centred on 0


def band_range(name: str) -> str:
    low, high = mfdfa.envelope.bands[name]
    return f"{low:g}–{high:g} Hz"


def series_label(name: str) -> str:
    return "Broadband EEG" if name == BROADBAND else f"{name.capitalize()} envelope"


# --- what is analysed -----------------------------------------------------------------------------

with st.container(key="nl-card-controls"):
    left, middle, right = st.columns([1.3, 1.3, 1], gap="small")
    name = left.selectbox(
        "Series",
        [BROADBAND, *mfdfa.envelope.bands],
        format_func=lambda n: (
            series_label(n) if n == BROADBAND else f"{series_label(n)} ({band_range(n)})"
        ),
        key="nl-mf-series",
        persist_state="session",
        help="The cleaned EEG itself, or the amplitude envelope of one frequency band.",
    )
    ranges = demo.ranges_in_seconds(name, config)
    if st.session_state.get("nl-mf-range") not in ranges:
        st.session_state.pop("nl-mf-range", None)
    fit_range = middle.selectbox(
        "Fit range",
        list(ranges),
        format_func=lambda r: f"{r} · {ranges[r][0]:g}–{ranges[r][1]:g} s",
        key="nl-mf-range",
        persist_state="session",
        help="The window sizes h(q) is fitted over, set from the label-blind scaling "
        "inspection (configs/features/mfdfa.yaml).",
    )
    default = described.series.channels.index("Cz") if "Cz" in described.series.channels else 0
    channel = right.selectbox(
        "Channel",
        described.series.channels,
        index=default,
        key="nl-mf-channel",
        persist_state="session",
    )
label = f"{series_label(name)}, {channel}"
table = engine.channel_table(digest, index, name, fit_range)
row = table.set_index("channel").loc[channel]

tabs = st.tabs(["Fluctuation functions", "Singularity spectrum", "Scalp maps", "Feature table"])

# --- Fq(s) ---------------------------------------------------------------------------------------

with tabs[0]:
    scales, qs, fq = engine.fluctuations(digest, index, name, channel)
    st.markdown(f"**Fq(s) of {ui.md(label)}**")
    st.caption(
        f"One line per moment order q ({', '.join(f'{q:g}' for q in qs)}), over "
        f"{len(scales)} window sizes. Shaded: the configured fit ranges h(q) is read from. "
        "Bad windows are kept here; the features leave them out."
    )
    st.altair_chart(charts.fluctuations(scales, qs, fq, ranges), theme=None, width="stretch")
    with st.expander("How to read this", icon=":material/help:"):
        st.markdown(
            "On log-log axes, a straight stretch is a range over which the fluctuations scale; "
            "its slope is h(q). Lines that fan out (different slopes for negative and positive "
            "q) mean small and large fluctuations scale differently: multifractality. Bends "
            "are crossovers; in this pipeline two of them come from the band-pass filter, "
            "which is why the fit ranges sit between them (reports/mfdfa_scaling.md)."
        )

# --- h(q) and f(alpha) ---------------------------------------------------------------------------

with tabs[1]:
    curves = engine.spectrum(digest, index, name, fit_range, channel)
    st.markdown(f"**Singularity spectrum of {ui.md(label)}, {fit_range} range**")
    if curves["h"].isna().all():
        st.info(
            "Too few clean windows on this range to fit h(q) for this channel: fewer than "
            f"{mfdfa.bad_windows.min_scales} window sizes kept "
            f"{mfdfa.bad_windows.min_windows} clean windows each."
        )
    else:

        def fmt(value: float, spec: str = ".3f") -> str | None:
            return None if not np.isfinite(value) else format(value, spec)

        ui.tiles(
            [
                {"label": "h2", "value": fmt(row["h2"]), "sub": "Hurst exponent of DFA"},
                {"label": "Δh", "value": fmt(row["delta_h"]), "sub": "h(q min) − h(q max)"},
                {"label": "Δα", "value": fmt(row["delta_alpha"]), "sub": "spectrum width"},
                {"label": "α₀", "value": fmt(row["alpha0"]), "sub": "peak of f(α)"},
                {"label": "Asymmetry", "value": fmt(row["asymmetry"], "+.3f"), "sub": "−1 to 1"},
                {
                    "label": "Fit quality",
                    "value": fmt(row["min_r2"]),
                    "sub": f"worst R²; {int(row['n_scales'])} window sizes",
                    "warn": bool(row["min_r2"] < mfdfa.poor_fit_r2),
                },
            ]
        )
        st.space("small")
        left, right = st.columns(2, gap="medium")
        with left:
            st.caption("h(q): the slope of log Fq(s) on the fit range, for every q.")
            st.altair_chart(charts.h_of_q(curves), theme=None, width="stretch")
        with right:
            st.caption("f(α): derived from h(q); one point per q. The rule marks α₀.")
            st.altair_chart(
                charts.singularity(curves, float(row["alpha0"])), theme=None, width="stretch"
            )
        if row["min_r2"] < mfdfa.poor_fit_r2:
            st.warning(
                f"At least one h(q) fit has R² below {mfdfa.poor_fit_r2:g} (configured "
                "threshold of a poor fit): read these values with care.",
                icon=":material/warning:",
            )
        if curves["h"].iloc[0] < curves["h"].iloc[-1]:
            st.warning(
                "h(q) rises with q on this range, so f(α) opens upwards instead of forming the "
                "usual arch: the fluctuations do not scale as a multifractal does here.",
                icon=":material/warning:",
            )
        if name == BROADBAND and len(mfdfa.broadband.wide_features[fit_range]) == 1:
            st.caption(
                f"On the {fit_range} range only h2 is used as a feature: filtered noise is as "
                "multifractal as EEG there (configs/features/mfdfa.yaml), so the width, Δh and "
                "asymmetry measured on it describe the filter."
            )

# --- maps -----------------------------------------------------------------------------------------

with tabs[2]:
    section = mfdfa.broadband if name == BROADBAND else mfdfa.envelope
    used = list(section.wide_features[fit_range])
    st.markdown(f"**{ui.md(series_label(name))}, {fit_range} range: the features the models use**")
    maps = [
        figures.Map(table[feature].to_numpy(dtype=float), feature, feature in SIGNED)
        for feature in used
    ]
    st.pyplot(
        figures.scalp_maps(maps, list(table["channel"]), config.data.dataset.montage),
        transparent=True,
        width="stretch",
    )
    st.caption(
        "One value per channel, bad windows left out; each map has its own scale. Δh and "
        "asymmetry have a sign and are centred on 0 (rose below, cyan above)."
    )
    st.page_link(
        PAGES["topography"].path,
        label="Inspect one map at a time, with the channels ranked",
        icon=PAGES["topography"].icon,
    )

# --- table ----------------------------------------------------------------------------------------

with tabs[3]:
    help_text = {f: TERMS[f] for f in ("h2", "delta_h", "delta_alpha", "alpha0", "asymmetry",
                                           "min_r2", "n_scales")}  # fmt: skip
    st.markdown(f"**Every channel · {ui.md(series_label(name))}, {fit_range} range**")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=min(36 * (len(table) + 1) + 4, 620),
        column_config={
            "channel": st.column_config.TextColumn("Channel", pinned=True),
            **{
                f: st.column_config.NumberColumn(f, help=help_text[f], format="%.4f")
                for f in ("h2", "delta_h", "delta_alpha", "alpha0", "asymmetry", "min_r2")
            },
            "n_scales": st.column_config.NumberColumn("n_scales", help=help_text["n_scales"]),
        },
    )
    st.caption(
        "Sort by clicking a column; search from the table's toolbar. Values are as computed, "
        "bad windows left out; NaN means too few clean windows for a fit."
    )
    stem = f"{engine.files()[index][0].rsplit('.', 1)[0]}_{name}_{fit_range}_mfdfa"
    st.download_button(
        "Download as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}.csv",
        mime="text/csv",
        icon=":material/download:",
        on_click="ignore",
    )

ui.glossary(
    [
        "Multifractal analysis",
        "Fluctuation function",
        "Generalised Hurst exponent",
        "Singularity spectrum",
        "h2",
        "delta_h",
        "delta_alpha",
        "alpha0",
        "asymmetry",
    ]
)
