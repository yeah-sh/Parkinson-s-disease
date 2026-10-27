"""Overview: the upload, where the pipeline stands, and what the recording is."""

from __future__ import annotations

from html import escape

import streamlit as st

from neurolab import components as ui
from neurolab import engine
from neurolab.pages import PAGES

ui.page_header("overview")
config = engine.settings()


def upload_area() -> None:
    mode = st.segmented_control(
        "Analysis mode",
        [engine.SINGLE, engine.PAIRED],
        format_func=lambda m: (
            "One recording" if m == engine.SINGLE else "Paired · OFF and ON of one patient"
        ),
        default=engine.SINGLE,
        required=True,
        key=engine.MODE,
        persist_state="session",
        help="Paired mode adds a model that tells which of a patient's two recordings is the "
        "OFF-medication one.",
    )
    labels = ["Recording (.bdf or .edf)"]
    if mode == engine.PAIRED:
        labels = ["First recording (.bdf or .edf)", "Second recording (.bdf or .edf)"]
    columns = st.columns(len(labels)) if len(labels) > 1 else [st.container()]
    generation = st.session_state.get(engine.UPLOADS, 0)
    uploads = [
        column.file_uploader(label, type=["bdf", "edf"], key=f"nl-upload-{generation}-{i}")
        for i, (column, label) in enumerate(zip(columns, labels, strict=True))
    ]
    if all(uploads):
        engine.load(tuple((u.name, u.getvalue()) for u in uploads))
        st.rerun()
    if any(uploads):
        st.caption("Add the second recording of the same patient.")
    needs = ui.requirements()
    ui.kv([(name.capitalize(), needs[name]) for name in ui.CHECK_ORDER[1:]])
    st.caption(
        "Files are checked before anything else is done with them, and stay on this machine: "
        "the app serves on localhost only."
    )


def loaded_area() -> None:
    checks = engine.validation()
    for report in checks["reports"]:
        with st.container(key=f"nl-well-file-{report['name']}"):
            ui.file_card(report)
            if report["error"]:
                st.error(f"**{ui.md(report['name'])} cannot be used.** {ui.md(report['error'])}")
    if checks["pair_error"]:
        st.error(f"**These two files cannot be used as a pair.** {ui.md(checks['pair_error'])}")
    if engine.usable(checks) and len(checks["reports"]) == 2:
        st.checkbox(
            "Both recordings are from the same patient (OFF and ON medication)",
            key=engine.SAME_PATIENT,
            persist_state="session",
        )
        if not st.session_state.get(engine.SAME_PATIENT):
            st.warning(
                "The paired model compares the OFF and ON recordings of one patient. Tick "
                "“Both recordings are from the same patient” if they are; otherwise clear them "
                "and analyse one at a time."
            )
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        ui.run_button("overview")
        st.page_link(PAGES["recording"].path, label="Review the checks", icon=":material/rule:")
    found = engine.analysis()
    if found is None:
        return
    if found.failed_step == "evaluate":
        st.warning(
            f"**Cleaned and described, but no prediction.** {ui.md(found.error or '')}",
            icon=":material/info:",
        )
    elif found.failed_step is not None:
        st.error(f"**The analysis stopped.** {ui.md(found.error or '')}")
    else:
        st.info(
            f"Analysis complete in {found.seconds:.0f} s. The pages on the left show the "
            "signal, its multifractal analysis and what the models said.",
            icon=":material/check:",
        )


left, right = st.columns([7, 5], gap="large")
with left, st.container(key="nl-card-input"):
    ui.section("Input")
    if engine.files() is None:
        upload_area()
    else:
        loaded_area()
with right, st.container(key="nl-card-pipeline"):
    ui.section("Pipeline")
    ui.steps(engine.pipeline())


def summary(index: int, report: dict) -> list[dict]:
    """The recording's tiles, with values only where they have been measured."""
    header, facts = report.get("header"), report.get("facts")
    found = engine.analysis()
    described = found.described[index] if found else None
    tiles = []
    if header:
        resampled = bool(facts) and abs(facts["sfreq"] - header["sfreq"]) > 1e-6
        if resampled:
            rate = f"resampled to {facts['sfreq']:g} Hz"
        else:
            rate = "as in training" if facts else "from the file header"
        tiles.append(
            {
                "label": "Sampling rate",
                "value": f"{header['sfreq']:g} Hz",
                "sub": rate,
                "warn": resampled,
            }
        )
        used = facts["n_channels"] if facts else None
        tiles.append(
            {
                "label": "Channels",
                "value": used if used is not None else header["n_channels"],
                "sub": f"used, of {header['n_channels']} in the file"
                if used is not None
                else "in the file",
            }
        )
        tiles.append(
            {
                "label": "Duration",
                "value": ui.duration(header["duration_s"]),
                "sub": f"{config.preprocessing.crop.duration:g} s are analysed",
            }
        )
    else:
        tiles += [
            {"label": "Sampling rate", "value": None, "sub": "unreadable header"},
            {"label": "Channels", "value": None, "sub": ""},
            {"label": "Duration", "value": None, "sub": ""},
        ]
    if described is not None:
        bad = described.log["bad_segments"]
        limit = config.preprocessing.qc.max_bad_percent
        tiles.append(
            {
                "label": "Bad windows",
                "value": f"{bad['percent']:.1f} %",
                "sub": f"of the time, in {len(bad['segments'])} stretch(es)"
                + (f"; above the {limit:g} % QC flag" if bad["percent"] > limit else ""),
                "warn": bad["percent"] > limit,
            }
        )
        ica = described.log["ica"]
        tiles.append(
            {
                "label": "Preprocessing",
                "value": "Done",
                "sub": f"ICA removed {len(ica['removed'])} of {ica['n_components']} components",
            }
        )
    else:
        waiting = "after the analysis" if engine.usable(engine.validation()) else "not run"
        tiles.append({"label": "Bad windows", "value": None, "sub": waiting})
        state = "Failed" if found is not None else None
        tiles.append({"label": "Preprocessing", "value": state, "sub": waiting})
    return tiles


st.space("small")
ui.section("Recording summary")
checks = engine.validation()
if checks is None:
    ui.tiles(
        [
            {"label": "Sampling rate", "value": None, "sub": "no recording yet"},
            {"label": "Channels", "value": None, "sub": ""},
            {"label": "Duration", "value": None, "sub": ""},
            {"label": "Bad windows", "value": None, "sub": ""},
            {"label": "Preprocessing", "value": None, "sub": ""},
        ]
    )
else:
    for index, report in enumerate(checks["reports"]):
        if len(checks["reports"]) > 1:
            st.caption(ui.md(report["name"]))
        ui.tiles(summary(index, report))

st.space("small")
ui.section("Analysis modules")
MODULES = ("quality", "multifractal", "topography", "evaluation", "report", "about")
NEEDS_ANALYSIS = {"quality", "multifractal", "topography", "report"}
ready = engine.analysis() is not None
columns = st.columns(3, gap="small")
for position, name in enumerate(MODULES):
    info = PAGES[name]
    if name in NEEDS_ANALYSIS:
        state = ui.chip("Ready", "ok") if ready else ui.chip("After analysis", "mute")
    elif name == "evaluation":
        state = ui.chip("Model evidence", "run") if engine.evidence() else ui.chip("No models")
    else:
        state = ui.chip("Reference", "mute")
    with columns[position % 3], st.container(key=f"nl-card-module-{name}"):
        st.html(
            f'<div class="nl-module" style="border:0;padding:0;background:none">'
            f'<div class="h"><div class="t">{escape(info.title)}</div>{state}</div>'
            f'<div class="d">{escape(info.lede.replace("`", ""))}</div></div>'
        )
        st.page_link(info.path, label="Open", icon=info.icon)
