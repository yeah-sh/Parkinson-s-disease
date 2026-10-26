"""Research demo: predictions from one EEG recording, or a patient's two. Not a diagnostic tool.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py        # or: make app

A BDF or EDF file with the 32 channels of the biosemi32 cap is checked first; nothing else is
done with a file that fails a check. The prediction is made by the code of `pdeeg predict`
(pdeeg.predict.predictor.predict_files) with the final models of `pdeeg train-final`: stage 1,
Parkinson's or healthy; stage 2, OFF or ON medication, only after a stage 1 call of Parkinson's
outside its uncertain band; with two recordings of one patient, which of them is OFF. Every
stage shows how it did in cross-validation, and one that was not better than chance is greyed
out. A recording of the training dataset gets its out-of-fold results instead of a prediction.
Below the prediction: the features that moved it, and the MFDFA of the cleaned recording, the
data the models saw (pdeeg.demo).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import streamlit as st

from pdeeg import demo
from pdeeg.config import CONFIG_DIR_ENV, load_config
from pdeeg.features.scaling import BROADBAND
from pdeeg.predict.features import FeatureError, clean_input, work_files
from pdeeg.predict.inputs import FAILED, NOTE, PASSED, InputError, read_recording
from pdeeg.predict.predictor import PredictError, check_pair, predict_files, trained_on
from pdeeg.viz.demo import plot_fluctuations, plot_spectrum
from pdeeg.viz.predict import interval_bar, plot_contributions, probability_bar
from pdeeg.viz.stats import Topography, montage_info, plot_topographies

ROOT = Path(__file__).resolve().parents[1]
# Features with a sign, drawn on a scale centred on 0.
SIGNED = ("delta_h", "asymmetry")
TITLES = {
    "stage1": "Stage 1 · Parkinson's vs healthy",
    "stage2_single": "Stage 2 · OFF vs ON medication",
    "stage2_paired": "Stage 2, paired · which recording is OFF medication",
}
# The class at 0 and the class at 1 of each stage's probability.
ENDS = {
    "stage1": ("healthy", "Parkinson's"),
    "stage2_single": ("ON medication", "OFF medication"),
}
SHOWN = {"PD": "Parkinson's", "HC": "healthy", "OFF": "OFF medication", "ON": "ON medication"}
ICONS = {
    PASSED: ":green[:material/check_circle:]",
    NOTE: ":orange[:material/info:]",
    FAILED: ":red[:material/cancel:]",
}

st.set_page_config(page_title="pdeeg research demo", layout="wide")
# A stage that is not better than chance is greyed out (st.container(key="greyed-...")).
st.html("<style>[class*='st-key-greyed'] { opacity: 0.45; filter: grayscale(1); }</style>")


@st.cache_resource
def settings():
    return load_config(os.environ.get(CONFIG_DIR_ENV) or ROOT / "configs")


def banner(config) -> str:
    """The disclaimer, with who the models were trained on (from their metadata)."""
    trained = trained_on(config)
    if trained is None:
        return (
            f"**{config.predict.disclaimer}.** No models have been trained here yet "
            f"(`pdeeg train-final`); they learn from one dataset (OpenNeuro "
            f"{config.data.dataset.id}) and are not validated on other recordings."
        )
    return (
        f"**{config.predict.disclaimer}.** Trained on {trained['participants']} people from one "
        f"dataset (OpenNeuro {trained['dataset']['id']}); not validated on other recordings."
    )


def _write(folder: str, index: int, name: str, content: bytes) -> Path:
    """An upload as a file under its own name (one folder per upload: two may share a name)."""
    path = Path(folder) / f"upload{index + 1}" / Path(name).name
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path


@st.cache_data(show_spinner="Reading and checking the recording.")
def checked(files: tuple[tuple[str, bytes], ...]) -> tuple[list[dict], str | None]:
    """Each file's checks (pdeeg.predict.inputs.read_recording), and the pair's error if any."""
    config = settings()
    reports, recordings = [], []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        for index, (name, content) in enumerate(files):
            try:
                recording = read_recording(_write(folder, index, name, content), config)
            except InputError as exc:
                checks, error = exc.checks, str(exc)
            else:
                recordings.append(recording)
                checks, error = recording.checks, None
            reports.append({"name": name, "checks": [asdict(c) for c in checks], "error": error})
        pair_error = None
        if len(recordings) == 2:
            try:
                check_pair(recordings)
            except InputError as exc:
                pair_error = str(exc)
    return reports, pair_error


@st.cache_resource(
    max_entries=4,
    show_spinner="Cleaning (ICA takes a minute or two), MFDFA and band power, then the models.",
)
def analysed(files: tuple[tuple[str, bytes], ...]) -> dict:
    """What ``pdeeg predict`` gives for the files, and their cleaned data for the figures.

    ``result`` is :func:`predict_files`'s, or None with ``error`` saying why. ``described``
    holds per file the MFDFA series and the cleaning log of the data the models saw (cleaned
    here as `predict` cleans when it made no features: a training recording, or no models).
    """
    config = settings()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        paths = [_write(folder, i, name, content) for i, (name, content) in enumerate(files)]
        work = Path(folder) / "work"
        result, error = None, None
        try:
            jobs = config.mfdfa.extraction.n_jobs
            result = predict_files(config, paths, out_dir=work, n_jobs=jobs, say=lambda _: None)
        except (InputError, PredictError, FeatureError) as exc:
            error = str(exc)
        described = []
        for index, path in enumerate(paths):
            entry = result["inputs"][index] if result else {}
            if "cleaned" in entry:
                cleaned, log = Path(entry["cleaned"]), Path(entry["cleaning_log"])
            else:
                name, own = f"input{index + 1}", work / f"describe{index + 1}"
                try:
                    recording = read_recording(path, config)
                    clean_input(recording.raw, config, own, name=name, source=path.name)
                except (InputError, FeatureError):
                    described.append(None)
                    continue
                written = work_files(config, own, name)
                cleaned, log = written["raw"], written["log"]
            raw = mne.io.read_raw_fif(cleaned, preload=True, verbose="error")
            described.append(
                (demo.series(raw, config), json.loads(log.read_text(encoding="utf-8")))
            )
    return {"result": result, "error": error, "described": described}


@st.cache_data(show_spinner="MFDFA of every channel.")
def channel_table(files, index: int, name: str, fit_range: str) -> pd.DataFrame:
    data, _ = analysed(files)["described"][index]
    return demo.channel_features(data, name, fit_range, settings())


def _escaped(text: str) -> str:
    for character in "\\*_$[]":
        text = text.replace(character, f"\\{character}")
    return text


def show_checks(reports: list[dict]) -> None:
    st.subheader("Input checks")
    columns = st.columns(len(reports))
    for column, report in zip(columns, reports, strict=True):
        lines = [
            f"- {ICONS[c['status']]} **{c['name']}**: {_escaped(c['detail'])}"
            for c in report["checks"]
        ]
        column.markdown(f"**{_escaped(report['name'])}**\n\n" + "\n".join(lines))


def _ends(stage: dict, names: list[str]) -> tuple[str, str]:
    if stage["stage"] == "stage2_paired":
        return (f"{names[1]} is OFF", f"{names[0]} is OFF")
    return ENDS[stage["stage"]]


def show_reliability(cv: dict, greyed: bool) -> None:
    """The stage's cross-validated balanced accuracy, its interval, chance, and the p-value."""
    low, high = cv["ci95"]
    columns = st.columns(4)
    columns[0].metric("Balanced accuracy", f"{cv['balanced_accuracy']:.2f}")
    columns[1].metric("95% CI", f"{low:.2f} – {high:.2f}")
    columns[2].metric("Chance", f"{cv['chance']:.2f}")
    columns[3].metric("Permutation p", f"{cv['permutation_p']:.3f}")
    st.html(interval_bar(cv["balanced_accuracy"], low, high, cv["chance"], greyed=greyed))
    if cv["better_than_chance"]:
        st.markdown(f"**Better than chance**: {_escaped(cv['test'])} (q = {cv['q']:.3f}).")
    else:
        reasons = "; ".join(cv["not_better_because"])
        st.markdown(
            f"**Not better than chance**: {_escaped(reasons)}. The test: {_escaped(cv['test'])}."
        )
    pre = cv["prespecified_configuration"]
    lines = [
        f"Cross-validation split by person, {cv['n_participants']} people "
        f"({cv['n_recordings']} recordings). {cv['note']}",
        f"The configuration fixed before any classifier was run ({pre['model']} on "
        f"{pre['feature_set']}): {pre['balanced_accuracy']:.2f} "
        f"[{pre['ci95'][0]:.2f}, {pre['ci95'][1]:.2f}].",
    ]
    if "in_cascade" in cv:
        cascade = cv["in_cascade"]
        lines.append(
            f"Inside the two-stage cascade: {cascade['balanced_accuracy']:.2f} "
            f"[{cascade['ci95'][0]:.2f}, {cascade['ci95'][1]:.2f}], permutation p = "
            f"{cascade['permutation_p']:.3f}."
        )
    st.caption(_escaped(" ".join(lines)))


def show_stage(stage: dict, key: str, names: list[str]) -> None:
    st.subheader(TITLES[stage["stage"]])
    if "skipped" in stage:
        st.caption(_escaped(f"Not run: {stage['skipped']}."))
        return
    cv = stage["cross_validated"]
    greyed = not cv["better_than_chance"]
    ends = _ends(stage, names)
    low, high = stage["uncertain_band"]
    with st.container(key=f"{'greyed' if greyed else 'stage'}-{key}"):
        label = stage["label"]
        if stage["uncertain"]:
            st.badge("uncertain", icon=":material/help:", color="orange")
        else:
            called = ends[1] if label in ("PD", "OFF", "first recording is OFF") else ends[0]
            st.badge(SHOWN.get(label, called), color="blue")
        st.html(
            probability_bar(
                stage["probability"], stage["threshold"], (low, high), ends, greyed=greyed
            )
        )
        text = f"Probability of {ends[1]}: **{stage['probability']:.2f}**. "
        if (low, high) == (0.0, 1.0):
            text += (
                "This stage never gives a label: out of fold, none of its probabilities could be "
                "told from a coin flip."
            )
        elif low == high:
            text += (
                f"Threshold {stage['threshold']:g}. Out of fold, the probabilities on either side "
                "of it backed their label, so this stage has no uncertain band."
            )
        else:
            text += (
                f"Threshold {stage['threshold']:g}; between {low:g} and {high:g} (hatched) the "
                "stage gives no label, because there its out-of-fold probabilities could not be "
                "told from a coin flip."
            )
        st.markdown(text)
        if stage.get("out_of_fold"):
            st.markdown(
                f"Out of fold (fold {stage['fold']}); the true label is "
                f"**{SHOWN.get(stage['true_label'], stage['true_label'])}**."
            )
        if "note" in stage:
            st.caption(_escaped(stage["note"]))
    if greyed:
        st.warning(
            "**This model is not better than chance.** Its result is shown greyed out and "
            "should not be read as saying anything about this recording.",
            icon=":material/block:",
        )
    with st.expander("How reliable is this?", expanded=True):
        show_reliability(cv, greyed)


def show_result(result: dict) -> None:
    names = [Path(entry["path"]).name for entry in result["inputs"]]
    st.header("Prediction")
    if result["training_recording"]:
        matches = [entry["training_recording"] for entry in result["inputs"]]
        which = ", ".join(
            f"sub-{m['subject']} ses-{m['session']} (recognised by {m['matched_by']})"
            for m in matches
        )
        st.info(f"**A ds002778 training recording: {which}.** {result['explanation']}")
    for index, stage in enumerate(result["stages"]):
        if len(names) == 2 and stage["stage"] == "stage1":
            st.markdown(f"#### {_escaped(stage['recording'])}")
        elif stage["stage"] == "stage2_paired":
            st.markdown("#### The pair")
        show_stage(stage, str(index), names)

    st.header("Top contributing features")
    shown = [stage for stage in result["stages"] if "top_features" in stage]
    if not shown:
        st.caption(
            "Out-of-fold results come from models fitted during cross-validation; which "
            "features moved them is not stored."
        )
    for stage in shown:
        greyed = not stage["cross_validated"]["better_than_chance"]
        where = stage.get("recording") or "the pair"
        title = f"{TITLES[stage['stage']]}: {where}"
        st.pyplot(
            plot_contributions(stage["top_features"], _ends(stage, names), title, greyed=greyed)
        )
        with st.expander("As a table"):
            st.dataframe(pd.DataFrame(stage["top_features"]["features"]), hide_index=True)


def show_recording(files, described: list) -> None:
    """The MFDFA figures of the cleaned recording, as the app showed them before it predicted."""
    config = settings()
    mfdfa = config.mfdfa
    st.header("The recording")
    names = [name for name, _ in files]
    index = 0
    if len(files) == 2:
        index = st.radio("Recording", [0, 1], format_func=lambda i: names[i], horizontal=True)
    if described[index] is None:
        st.info("This recording could not be cleaned, so it is not described here.")
        return
    data, log = described[index]
    removed = log["ica"]["removed"]
    st.subheader("Cleaning")
    st.markdown(
        f"- {len(data.channels)} channels at {data.sfreq:g} Hz; {log['crop']['duration_s']:g} s "
        f"kept, from {log['crop']['anchor']} ({log['crop']['start_s']:.1f} s in).\n"
        f"- ICA removed {len(removed)} of {log['ica']['n_components']} components"
        + (f": {', '.join(sorted({c['label'] for c in removed}))}" if removed else "")
        + f".\n- {log['bad_segments']['percent']:.1f} % of the time marked bad; those windows "
        "are left out of the spectrum and the maps below.\n"
        "- Cleaned as the training recordings were, with the settings in configs/; these are "
        "the data the models saw."
    )

    left, middle, right = st.columns(3)
    name = left.selectbox(
        "Series",
        [BROADBAND, *mfdfa.envelope.bands],
        format_func=lambda n: "broadband EEG" if n == BROADBAND else f"{n} envelope",
    )
    ranges = demo.ranges_in_seconds(name, config)
    fit_range = middle.selectbox(
        "Fit range",
        list(ranges),
        format_func=lambda r: f"{r} ({ranges[r][0]:g}-{ranges[r][1]:g} s)",
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
                "usual arch: the fluctuations do not scale as a multifractal does here. The "
                "scaling inspection (reports/mfdfa_scaling.md) shows where the dataset's EEG "
                "scales."
            )

    st.subheader("Over the scalp")
    section = mfdfa.broadband if name == BROADBAND else mfdfa.envelope
    shown = section.wide_features[fit_range]
    table = channel_table(files, index, name, fit_range)
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


config = settings()
notice = banner(config)
st.title("Parkinson's disease and resting-state EEG: a research demo")
st.warning(notice, icon=":material/science:")
st.sidebar.warning(notice)

paired = st.toggle(
    "Paired mode: two recordings of one patient (OFF and ON medication)",
    help="The paired model tells which of a patient's two recordings is the OFF one.",
)
first_column, second_column = st.columns(2) if paired else (st.container(), None)
first = first_column.file_uploader("Recording (.bdf or .edf)", type=["bdf", "edf"], key="first")
second, same = None, False
if paired:
    second = second_column.file_uploader(
        "Second recording (.bdf or .edf)", type=["bdf", "edf"], key="second"
    )
    same = st.checkbox("Both recordings are from the same patient")
if first is None or (paired and second is None):
    st.info(
        f"A file needs the channels of the {config.data.dataset.montage} montage (older names "
        f"such as T3 are accepted), at least {config.preprocessing.crop.duration:g} s of EEG, "
        f"ideally sampled at {config.predict.sfreq:g} Hz (other rates are resampled, with a "
        "note). It is checked before anything else is done with it."
    )
    st.stop()

files = tuple((upload.name, upload.getvalue()) for upload in (first, second) if upload)
reports, pair_error = checked(files)
show_checks(reports)
failed = [report for report in reports if report["error"]]
for report in failed:
    st.error(f"**{_escaped(report['name'])} cannot be used.** {_escaped(report['error'])}")
if pair_error:
    st.error(f"**These two files cannot be used as a pair.** {_escaped(pair_error)}")
if failed or pair_error:
    st.stop()
if paired and not same:
    st.warning(
        "The paired model compares the OFF and ON recordings of one patient. Tick “Both "
        "recordings are from the same patient” if they are; otherwise analyse them one at a "
        "time with paired mode off."
    )
    st.stop()

outcome = analysed(files)
if outcome["error"]:
    st.error(f"**No prediction.** {_escaped(outcome['error'])}")
else:
    show_result(outcome["result"])
show_recording(files, outcome["described"])
st.caption(notice)
