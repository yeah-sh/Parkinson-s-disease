"""One run of the page: predict_files, then what the four graphs and the report need.

Every number comes from pdeeg. Predictions are :func:`pdeeg.predict.predictor.predict_files`
(looked up on its module when called, so that tests can stand in for it); the cleaned data are
described by pdeeg.demo through NeuroLab's engine, which also checks the uploads; the group
ranges are read from the training feature tables. Nothing here has model code of its own.

:func:`analyse` keeps a run per file digest with ``st.cache_data``, so a rerun of the page is
instant. Its progress box is created inside it and emptied at the end, which keeps the cached
replay of that box harmless.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import streamlit as st

import pdeeg.config
from neurolab import engine, report
from pdeeg import demo
from pdeeg.features.curves import curves_path
from pdeeg.features.extract import CLEAN
from pdeeg.features.tables import table_paths
from pdeeg.predict import predictor
from pdeeg.predict.features import BAND_POWER, CLEANING, MFDFA, FeatureError
from pdeeg.predict.inputs import InputError
from restwave import blocks, settings
from restwave.charts import GROUPS

# The steps the page shows, in order.
CHECK, CLEAN_STEP, MEASURE, MODEL = range(4)
STEPS = ("Checking", "Cleaning signal", "Measuring brain-wave patterns", "Running model")
POWER = "log_relative_power"  # the band-power feature of the training tables and the models
_GROUP_NAMES = {"HC": GROUPS[0], "PD": GROUPS[1]}


# --- what the page draws ---------------------------------------------------------------------


@dataclass
class Shown:
    """One recording as the page draws it.

    ``signal_uv`` is ``seconds`` of the cleaned recording at the configured channels, starting
    ``start_s`` into the analysed window; ``band_power`` maps each band to the mean over
    channels of its log10 relative power; ``spectrum`` has ``q``, ``alpha`` and ``f_alpha``,
    the mean over channels. Bad windows are left out of both.
    """

    name: str
    channels: list[str]
    sfreq: float
    start_s: float
    signal_uv: np.ndarray
    band_power: dict[str, float]
    spectrum: pd.DataFrame
    removed: int
    bad_percent: float


@dataclass
class Outcome:
    """What one run gave: ``result`` of predict_files (None, with ``error`` and the step that
    failed), each recording as drawn (None for one that could not be described), and the
    report as HTML."""

    result: dict[str, Any] | None
    error: str | None
    failed_step: int | None
    shown: list[Shown | None]
    report_html: str | None
    seconds: float


def first_clean_start(bad: np.ndarray, length: int) -> int:
    """The first sample from which ``length`` samples hold no bad one (0 if there is none)."""
    if length >= bad.size:
        return 0
    counts = np.concatenate([[0], np.cumsum(bad.astype(int))])
    window = counts[length:] - counts[:-length]
    clean = np.flatnonzero(window == 0)
    return int(clean[0]) if clean.size else 0


def show(
    described: engine.Described,
    name: str,
    config: pdeeg.config.Config,
    app: settings.AppSettings,
) -> Shown:
    data = described.series
    rows = [data.channels.index(channel) for channel in app.channels]
    length = int(round(app.seconds * data.sfreq))
    start = first_clean_start(data.masks["broadband"], length)
    power = described.band_power
    clean = power[(power["segments"] == CLEAN) & (power["feature"] == POWER)]
    curves = pd.concat(
        [
            demo.spectrum(data, app.series, app.fit_range, channel, config)
            for channel in data.channels
        ],
        ignore_index=True,
    )
    return Shown(
        name=name,
        channels=list(app.channels),
        sfreq=data.sfreq,
        start_s=start / data.sfreq,
        signal_uv=described.cleaned_uv[rows, start : start + length].astype(np.float32),
        band_power={str(b): float(v) for b, v in clean.groupby("band")["value"].mean().items()},
        spectrum=curves.groupby("q", as_index=False)[["alpha", "f_alpha"]].mean(),
        removed=len(described.log["ica"]["removed"]),
        bad_percent=float(described.log["bad_segments"]["percent"]),
    )


# --- the groups ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Groups:
    """The healthy and Parkinson's groups of the training data, one value per person.

    ``band`` has group, band, mean, low, high (log10 relative power); ``spectrum`` has group,
    q, alpha, low, high (alpha) and f_alpha; ``people`` counts each group.
    """

    band: pd.DataFrame
    spectrum: pd.DataFrame
    people: dict[str, int]


def _per_person(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    """Means over channels per recording, then over a person's recordings (OFF and ON)."""
    recordings = frame.groupby(["subject", "session", "group", *keys])[values].mean()
    return recordings.groupby(["subject", "group", *keys])[values].mean().reset_index()


def group_ranges(config: pdeeg.config.Config, app: settings.AppSettings) -> Groups | None:
    """The ranges drawn behind this recording, or None without the training tables."""
    psd, curves = table_paths(config, "psd")["long"], curves_path(config)
    if not (psd.is_file() and curves.is_file()):
        return None
    power = pd.read_parquet(
        psd,
        columns=["subject", "session", "group", "level", "band", "segments", "feature", "value"],
        filters=[("level", "==", "channel"), ("segments", "==", CLEAN), ("feature", "==", POWER)],
    )
    people = _per_person(power, ["band"], ["value"])
    band = people.groupby(["group", "band"])["value"].agg(["mean", "std"]).reset_index()
    band["low"] = band["mean"] - app.sd * band["std"]
    band["high"] = band["mean"] + app.sd * band["std"]

    spectra = pd.read_parquet(
        curves,
        columns=["subject", "session", "group", "series", "fit_range", "segments", "q", "alpha",
                 "f_alpha"],
        filters=[("series", "==", app.series), ("fit_range", "==", app.fit_range),
                 ("segments", "==", CLEAN)],
    )  # fmt: skip
    people_curves = _per_person(spectra, ["q"], ["alpha", "f_alpha"])
    spectrum = (
        people_curves.groupby(["group", "q"])
        .agg(alpha=("alpha", "mean"), sd=("alpha", "std"), f_alpha=("f_alpha", "mean"))
        .reset_index()
    )
    spectrum["low"] = spectrum["alpha"] - app.sd * spectrum["sd"]
    spectrum["high"] = spectrum["alpha"] + app.sd * spectrum["sd"]
    counts = people.drop_duplicates("subject")["group"].value_counts()
    rename = {"group": lambda frame: frame["group"].map(_GROUP_NAMES)}
    return Groups(
        band=band.assign(**rename)[["group", "band", "mean", "low", "high"]],
        spectrum=spectrum.assign(**rename)[["group", "q", "alpha", "low", "high", "f_alpha"]],
        people={_GROUP_NAMES[g]: int(n) for g, n in counts.items() if g in _GROUP_NAMES},
    )


# --- the sample recording --------------------------------------------------------------------


def sample_upload(config: pdeeg.config.Config, app: settings.AppSettings) -> tuple[str, bytes]:
    """(name, bytes) of the sample recording: made up, or the configured ds002778 one."""
    sample = app.sample
    if sample.subject is not None:
        name = f"sub-{sample.subject}_ses-{sample.session}_task-rest_eeg.bdf"
        path = (
            config.data.paths.raw / f"sub-{sample.subject}" / f"ses-{sample.session}" / "eeg" / name
        )
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run `pdeeg download` first")
        return name, path.read_bytes()
    raw = demo.synthetic_recording(seed=sample.seed, montage=config.data.dataset.montage)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        path = Path(folder) / "sample_recording_made_up.edf"
        mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="error")
        return path.name, path.read_bytes()


# --- the run ----------------------------------------------------------------------------------


class Progress:
    """The steps as they happen: what predict_files says, and the page's own work after it."""

    def __init__(self, draw: Callable[[str], None], names: Sequence[str]) -> None:
        self.draw = draw
        self.letters = {name: "AB"[i] for i, name in enumerate(names)} if len(names) > 1 else {}
        self.step, self.detail = CHECK, "Reading the file and checking it again"
        self.draw(blocks.steps(STEPS, self.step, self.detail))

    def go(self, step: int, detail: str) -> None:
        self.step, self.detail = step, detail
        self.draw(blocks.steps(STEPS, step, detail))

    def _who(self, message: str) -> str:
        name = message.split(": ", 1)[0]
        return f" (recording {self.letters[name]})" if name in self.letters else ""

    def hear(self, message: str) -> None:
        """``say`` of predict_files."""
        if message.endswith(CLEANING):
            self.go(
                CLEAN_STEP, "Removing eye blinks, heartbeats and muscle noise" + self._who(message)
            )
        elif message.endswith(MFDFA):
            self.go(MEASURE, "Measuring how irregular the brain waves are" + self._who(message))
        elif message.endswith(BAND_POWER):
            self.go(MEASURE, "Measuring the balance of slow and fast waves" + self._who(message))
        elif message == predictor.RUNNING_MODELS:
            self.go(MODEL, "Comparing the patterns with the two groups")


def run(
    uploads: Sequence[tuple[str, bytes]],
    checks: Mapping[str, Any],
    config: pdeeg.config.Config,
    evidence: Mapping[str, Mapping[str, Any]],
    app: settings.AppSettings,
    progress: Progress,
) -> Outcome:
    """Predict the uploads, describe them for the graphs, and write the report."""
    started = time.perf_counter()
    names = [name for name, _ in uploads]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        paths = [
            engine.write_upload(folder, i, name, data) for i, (name, data) in enumerate(uploads)
        ]
        work = Path(folder) / "work"
        try:
            result = predictor.predict_files(
                config,
                paths,
                out_dir=work,
                n_jobs=config.mfdfa.extraction.n_jobs,
                say=progress.hear,
            )
        except InputError as exc:
            return Outcome(None, str(exc), CHECK, [], None, time.perf_counter() - started)
        except predictor.PredictError as exc:
            return Outcome(None, str(exc), MODEL, [], None, time.perf_counter() - started)
        except FeatureError as exc:
            step = CLEAN_STEP if str(exc).startswith("too short to clean") else MEASURE
            return Outcome(None, str(exc), step, [], None, time.perf_counter() - started)
        described: list[engine.Described | None] = []
        shown: list[Shown | None] = []
        for index, path in enumerate(paths):
            entry = result["inputs"][index]
            who = f" (recording {'AB'[index]})" if len(paths) > 1 else ""
            if "cleaned" in entry:
                progress.go(MODEL, "Done. Drawing your graphs" + who)
                one = engine.describe(path, entry, index, work, config)
            else:  # the models needed no features (a training recording): clean it to draw it
                progress.go(CLEAN_STEP, "Removing eye blinks, heartbeats and muscle noise" + who)
                one = engine.describe(path, entry, index, work, config)
                progress.go(MEASURE, "Measuring the brain-wave patterns for the graphs" + who)
            described.append(one)
            shown.append(None if one is None else show(one, names[index], config, app))
        progress.go(MODEL, "Writing the report")
        seconds = time.perf_counter() - started
        found = engine.Analysis(
            result=result,
            error=None,
            failed_step=None,
            described=described,
            log=[],
            seconds=seconds,
            finished=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        channel = "Cz" if "Cz" in app.channels else app.channels[0]
        html = report.build(
            [report.Upload(name, len(data)) for name, data in uploads],
            checks,
            found,
            evidence,
            config,
            (app.series, app.fit_range, channel),
            found.finished,
        )
    exported = engine.export_result(result, names)
    return Outcome(exported, None, None, shown, html, time.perf_counter() - started)


@st.cache_data(show_spinner=False, max_entries=8)
def analyse(
    digest: str,
    config_dir: str,
    _uploads: tuple[tuple[str, bytes], ...],
    _checks: Mapping[str, Any],
) -> Outcome:
    """:func:`run` for the uploads of ``digest``, kept per digest (and configuration)."""
    box = st.empty()
    config = engine.settings()
    progress = Progress(box.html, [name for name, _ in _uploads])
    outcome = run(_uploads, _checks, config, engine.evidence(), settings.load(config), progress)
    box.empty()
    return outcome


@st.cache_data(show_spinner=False, max_entries=2)
def groups(config_dir: str) -> Groups | None:
    config = engine.settings()
    return group_ranges(config, settings.load(config))


@st.cache_data(show_spinner="Making a sample recording.", max_entries=2)
def sample(config_dir: str) -> tuple[str, bytes]:
    config = engine.settings()
    return sample_upload(config, settings.load(config))
