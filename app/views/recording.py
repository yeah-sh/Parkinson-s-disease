"""Recording Analysis: the input checks, the processing timeline, and what the cleaning did."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from neurolab import components as ui
from neurolab import engine

ui.page_header("recording")
config = engine.settings()
checks = engine.validation()

if checks is None:
    ui.empty(
        "No recording loaded",
        "Upload a .bdf or .edf recording on the Overview page. Its checks, the processing "
        "timeline and the cleaning log appear here.",
        link="overview",
        label="Go to the upload",
    )
    st.stop()

# --- input validation ---------------------------------------------------------------------------

ui.section("Input validation")
st.caption(
    "The checks of `pdeeg predict`, in order. A failed check stops everything after it; a "
    "note means the file passed but something about it should be known."
)
columns = st.columns(len(checks["reports"]), gap="medium")
for column, report in zip(columns, checks["reports"], strict=True):
    with column, st.container(key=f"nl-card-checks-{report['name']}"):
        ui.file_card(report)
        st.space("small")
        ui.checklist(report)
        if report["facts"] and report["facts"]["notes"]:
            st.caption("Notes: " + ui.md("; ".join(report["facts"]["notes"])) + ".")
if checks["pair_error"]:
    st.error(
        f"**These two files cannot be used as a pair.** {ui.md(checks['pair_error'])}. Upload "
        "two different sessions of the same patient."
    )

# --- processing ---------------------------------------------------------------------------------

st.space("small")
ui.section("Processing")
found = engine.analysis()
with st.container(key="nl-card-timeline"):
    ui.steps(engine.pipeline(), row=True)
    if found is None:
        st.space("small")
        if engine.usable(checks):
            st.caption(
                "Validation and processing are separate: nothing has been cleaned or predicted "
                "yet. The analysis cleans the recording (with ICA), computes its MFDFA and band "
                "power, and applies the models."
            )
            ui.run_button("recording")
            if len(checks["reports"]) == 2 and not st.session_state.get(engine.SAME_PATIENT):
                st.caption("Confirm on the Overview page that both recordings are one patient's.")
    else:
        st.space("small")
        log = pd.DataFrame(
            {
                "Step": [step["detail"] for step in found.log],
                "Seconds": [round(step["seconds"] or 0.0, 1) for step in found.log],
            }
        )
        with st.expander(f"Run log · {found.seconds:.0f} s, finished {found.finished}"):
            st.dataframe(log, hide_index=True, width="stretch")
        if found.error:
            where = {
                "validate": "The inputs were refused",
                "preprocess": "Cleaning failed",
                "features": "The features could not be computed",
                "evaluate": "No prediction",
            }[found.failed_step]
            message = f"**{where}.** {ui.md(found.error)}"
            if found.failed_step == "evaluate":
                st.warning(message, icon=":material/info:")
            else:
                st.error(message)

if found is None:
    st.stop()

# --- what the cleaning did ----------------------------------------------------------------------

st.space("small")
ui.section("Preprocessing")
prep = config.preprocessing
for index, (name, _) in enumerate(engine.files()):
    described = found.described[index]
    with st.container(key=f"nl-card-cleaning-{index}"):
        st.markdown(f"**{ui.md(name)}**")
        if described is None:
            st.info("This recording could not be cleaned, so there is nothing to show.")
            continue
        log = described.log
        crop, ica, bad, line = log["crop"], log["ica"], log["bad_segments"], log["line_noise"]
        left, right = st.columns([3, 2], gap="large")
        with left:
            detected = (
                "none detected" if line["detected_hz"] is None else f"{line['detected_hz']:g} Hz"
            )
            reference = (
                prep.reference if isinstance(prep.reference, str) else ", ".join(prep.reference)
            )
            ui.kv(
                [
                    (
                        "Window",
                        f"{crop['duration_s']:g} s from {crop['anchor']} "
                        f"({crop['start_s']:.1f} s into the file; {crop['available_s']:.1f} s "
                        "available)",
                    ),
                    (
                        "Band-pass",
                        f"{prep.filter.l_freq:g}–{prep.filter.h_freq:g} Hz, zero-phase FIR "
                        f"({prep.filter.fir_window})",
                    ),
                    ("Reference", f"{reference} reference"),
                    (
                        "Mains",
                        f"{detected} in the spectrum; {line['configured_hz']:g} Hz notched "
                        "on the copy ICA is fitted on",
                    ),
                    (
                        "ICA",
                        f"{ica['method']}, {ica['n_components']} components (rank "
                        f"{ica['rank']}), {ica['n_iter']} iterations; removed "
                        f"{len(ica['removed'])}, {ica['variance_removed_percent']:.1f} % of "
                        "the variance",
                    ),
                    (
                        "Bad windows",
                        f"{bad['percent']:.1f} % of the time ({bad['peak_percent']:.1f} % "
                        f"above {prep.bad_segments.peak_to_peak_uv:g} µV peak-to-peak, "
                        f"{bad['flat_percent']:.1f} % below {prep.bad_segments.flat_uv:g} µV) "
                        f"in {bad['window_s']:g} s windows",
                    ),
                    (
                        "Amplitude",
                        f"median channel SD {log['summary']['median_sd_uv']:.1f} µV after "
                        f"cleaning; largest {log['summary']['max_sd_ratio']:.1f}× the median",
                    ),
                ]
            )
        with right:
            st.markdown("**Components removed by ICA**")
            if ica["removed"]:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Component": [f"IC{c['index']:02d}" for c in ica["removed"]],
                            "ICLabel class": [c["label"] for c in ica["removed"]],
                            "Probability": [c["probability"] for c in ica["removed"]],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Probability": st.column_config.ProgressColumn(
                            "Probability",
                            min_value=0.0,
                            max_value=1.0,
                            format="%.2f",
                            help=f"ICLabel's probability for its class; removed at "
                            f"{prep.ica.threshold:g} or more",
                        )
                    },
                )
            else:
                st.caption(
                    f"None: no component was confidently ({prep.ica.threshold:g} or more) one "
                    f"of {', '.join(prep.ica.exclude_labels)}."
                )
            counts = {k: v for k, v in ica["label_counts"].items() if v}
            st.caption(
                "ICLabel's most likely class over all components: "
                + ", ".join(f"{k} {v}" for k, v in counts.items())
                + "."
            )

ui.glossary(["Sampling frequency", "Montage", "ICA artifact removal", "Bad windows"])
