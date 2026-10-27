"""Research Report: the analysis as one printable HTML file, and the raw result."""

from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from neurolab import components as ui
from neurolab import engine, report
from pdeeg import demo
from pdeeg.features.scaling import BROADBAND
from pdeeg.predict import predictor

ui.page_header("report")
found = ui.need_analysis("report")
if found is None:
    st.stop()
config = engine.settings()
uploads = engine.files()
names = [name for name, _ in uploads]
stem = names[0].rsplit(".", 1)[0] + ("_pair" if len(names) == 2 else "")
channels = next(d for d in found.described if d is not None).series.channels

ui.section("Contents")
with st.container(key="nl-card-contents"):
    has_result = found.result is not None
    ui.steps(
        [
            engine.Step("recording", "Recording and input checks", "done", "Every check"),
            engine.Step("cleaning", "Preprocessing", "done", "Cleaning log, QC figures"),
            engine.Step("signal", "Signal", "done", "Before and after cleaning"),
            engine.Step("mfdfa", "Multifractal analysis", "done", "Fq(s), f(α), maps, table"),
            engine.Step(
                "models",
                "Model evaluation",
                "done" if has_result else "attention",
                "Every stage with its evidence" if has_result else "No prediction to report",
            ),
        ],
        row=True,
    )
    st.space("small")
    st.caption("The MFDFA section shows one series, fit range and channel:")
    a, b, c = st.columns(3, gap="small")
    name = a.selectbox(
        "Series",
        [BROADBAND, *config.mfdfa.envelope.bands],
        format_func=lambda n: "Broadband EEG" if n == BROADBAND else f"{n.capitalize()} envelope",
        key="nl-report-series",
        persist_state="session",
    )
    ranges = list(demo.ranges_in_seconds(name, config))
    if st.session_state.get("nl-report-range") not in ranges:
        st.session_state.pop("nl-report-range", None)
    fit_range = b.selectbox("Fit range", ranges, key="nl-report-range", persist_state="session")
    channel = c.selectbox(
        "Channel",
        channels,
        index=channels.index("Cz") if "Cz" in channels else 0,
        key="nl-report-channel",
        persist_state="session",
    )

choice = (name, fit_range, channel)
cache_key = (engine.key(), choice)
built = st.session_state.get("nl-report")
if built is not None and built[0] != cache_key:
    built = None

with st.container(horizontal=True, gap="small", vertical_alignment="center"):
    if st.button(
        "Build report" if built is None else "Rebuild report",
        type="primary" if built is None else "secondary",
        icon=":material/description:",
        key="nl-build-report",
    ):
        with st.spinner("Drawing the figures and writing the report."):
            html = report.build(
                [report.Upload(n, len(data)) for n, data in uploads],
                engine.validation(),
                found,
                engine.evidence(),
                config,
                choice,
                datetime.now().astimezone().isoformat(timespec="seconds"),
            )
        built = (cache_key, html)
        st.session_state["nl-report"] = built
    if built is not None:
        st.download_button(
            "Report (HTML)",
            built[1].encode("utf-8"),
            file_name=f"{stem}_neurolab_report.html",
            mime="text/html",
            icon=":material/download:",
            on_click="ignore",
        )
    if found.result is not None:
        exported = engine.export_result(found.result, names)
        st.download_button(
            "Result (JSON)",
            json.dumps(exported, indent=1).encode("utf-8"),
            file_name=f"{stem}_prediction.json",
            mime="application/json",
            icon=":material/data_object:",
            on_click="ignore",
        )
        st.download_button(
            "Summary (text)",
            predictor.summary_text(exported).encode("utf-8"),
            file_name=f"{stem}_prediction.txt",
            mime="text/plain",
            icon=":material/article:",
            on_click="ignore",
        )
st.caption(
    "The JSON and the text are what `pdeeg predict` writes for the same files, with the "
    "uploads' names in place of the temporary paths. The report embeds its figures, so it "
    "opens offline and prints (or saves as PDF) from the browser."
)

if built is not None:
    ui.section("Preview")
    st.iframe(built[1], height=900)
