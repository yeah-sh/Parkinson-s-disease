"""Research demo: multifractal analysis of one EEG recording. Not a diagnostic tool.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py        # or: make app

A BDF or EDF file with the 32 channels of the biosemi32 cap is cleaned and analysed with the
project's settings (configs/). The page shows how the recording's fluctuations scale. It has no
classifier and makes no statement about disease: the analysis is in pdeeg.demo.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from pdeeg import demo
from pdeeg.config import CONFIG_DIR_ENV, load_config
from pdeeg.features.scaling import BROADBAND
from pdeeg.viz.demo import plot_fluctuations, plot_spectrum
from pdeeg.viz.stats import Topography, montage_info, plot_topographies

DISCLAIMER = "Research demo — not a diagnostic tool."
EXPLANATION = (
    "This page describes how the fluctuations of one EEG recording scale with time. It does "
    "not classify the recording, and nothing on it says whether the person recorded has "
    "Parkinson's disease or any other condition."
)
ROOT = Path(__file__).resolve().parents[1]
# Features with a sign, drawn on a scale centred on 0.
SIGNED = ("delta_h", "asymmetry")

st.set_page_config(page_title="pdeeg research demo", layout="wide")
st.title("MFDFA of resting-state EEG")
st.warning(f"**{DISCLAIMER}** {EXPLANATION}")
st.sidebar.markdown(f"**{DISCLAIMER}**")


@st.cache_resource
def settings():
    return load_config(os.environ.get(CONFIG_DIR_ENV) or ROOT / "configs")


@st.cache_data(show_spinner="Reading and cleaning the recording; ICA takes a minute or two.")
def analysed(content: bytes, suffix: str, line_freq: float):
    config = settings()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"recording{suffix}"
        path.write_bytes(content)
        raw = demo.read_eeg(path, config.data.dataset.montage)
    cleaned, log = demo.clean(raw, config, line_freq)
    return demo.series(cleaned, config), log


@st.cache_data(show_spinner="MFDFA of every channel.")
def features(content: bytes, suffix: str, line_freq: float, name: str, fit_range: str):
    data, _ = analysed(content, suffix, line_freq)
    return demo.channel_features(data, name, fit_range, settings())


config = settings()
mfdfa = config.mfdfa
upload = st.file_uploader("A resting-state recording (.bdf or .edf)", type=["bdf", "edf"])
line_freq = st.sidebar.selectbox(
    "Mains frequency (Hz)",
    [50.0, 60.0],
    index=[50.0, 60.0].index(config.data.dataset.line_freq),
    help="Notched out of the copy that ICA is fitted on.",
)
if upload is None:
    st.info(
        f"The file needs the channels of the {config.data.dataset.montage} montage under their "
        f"standard names and at least {config.preprocessing.crop.duration:g} s of data. It is "
        "cleaned and analysed with the settings in configs/, as the pipeline does for the "
        "dataset."
    )
    st.stop()

content, suffix = upload.getvalue(), Path(upload.name).suffix.lower()
try:
    data, log = analysed(content, suffix, line_freq)
except demo.DemoError as exc:
    st.error(str(exc))
    st.stop()

removed = log["ica"]["removed"]
st.subheader("Cleaning")
st.markdown(
    f"- {len(data.channels)} channels at {data.sfreq:g} Hz; {log['crop']['duration_s']:g} s kept, "
    f"from {log['crop']['anchor']} ({log['crop']['start_s']:.1f} s in).\n"
    f"- ICA removed {len(removed)} of {log['ica']['n_components']} components"
    + (f": {', '.join(sorted({c['label'] for c in removed}))}" if removed else "")
    + f".\n- {log['bad_segments']['percent']:.1f} % of the time marked bad; those windows are "
    "left out of the spectrum and the maps below."
)

names = [BROADBAND, *mfdfa.envelope.bands]
left, middle, right = st.columns(3)
name = left.selectbox(
    "Series",
    names,
    format_func=lambda n: "broadband EEG" if n == BROADBAND else f"{n} envelope",
)
ranges = demo.ranges_in_seconds(name, config)
fit_range = middle.selectbox(
    "Fit range", list(ranges), format_func=lambda r: f"{r} ({ranges[r][0]:g}-{ranges[r][1]:g} s)"
)
channel = right.selectbox("Channel", data.channels)
label = "broadband EEG" if name == BROADBAND else f"{name} envelope"

st.subheader("Fluctuation functions")
scales, qs, fq = demo.fluctuation_functions(data, name, channel, config)
st.pyplot(plot_fluctuations(scales, qs, fq, ranges, f"{label}, {channel}"))

st.subheader("Singularity spectrum")
curves = demo.spectrum(data, name, fit_range, channel, config)
if curves["h"].isna().all():
    st.info("Too few clean windows on this range to fit h(q).")
else:
    st.pyplot(plot_spectrum(curves, f"{label}, {channel}, {fit_range} range"))
    if curves["h"].iloc[0] < curves["h"].iloc[-1]:
        st.caption(
            "h(q) rises with q on this range, so f(α) opens upwards instead of forming the "
            "usual arch: the fluctuations do not scale as a multifractal does here. The scaling "
            "inspection (reports/mfdfa_scaling.md) shows where the dataset's EEG scales."
        )

st.subheader("Over the scalp")
section = mfdfa.broadband if name == BROADBAND else mfdfa.envelope
shown = section.wide_features[fit_range]
table = features(content, suffix, line_freq, name, fit_range)
panels = []
for feature in shown:
    values = table[feature].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if feature in SIGNED:  # either side of 0: two hues, grey at 0
        top = float(np.abs(finite).max()) if finite.size else 1.0
        panels.append(Topography(values, feature, "t", (-top, top)))
    else:
        limits = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        panels.append(Topography(values, feature, "mean", limits))
info = montage_info(data.channels, config.data.dataset.montage)
st.pyplot(
    plot_topographies(
        info,
        [("", panels)],
        f"{label}, {fit_range} range",
        "One value per channel, bad windows left out. Each map has its own scale; "
        "delta_h and asymmetry are centred on 0.",
    )
)
if name == BROADBAND and len(shown) == 1:
    st.caption(
        "Only h2 is shown on this range: filtered noise is as multifractal as EEG here "
        "(configs/features/mfdfa.yaml)."
    )
st.dataframe(table, hide_index=True)
st.caption(DISCLAIMER)
