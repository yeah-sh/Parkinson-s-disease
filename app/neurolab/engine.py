"""What the app computes and keeps: the checks of an upload, its analysis, and descriptions.

Every number comes from pdeeg. The checks are those of `pdeeg predict`
(pdeeg.predict.inputs.read_recording); the analysis is
:func:`pdeeg.predict.predictor.predict_files`, then the cleaned data described by pdeeg.demo.
Nothing here has model code of its own.

An upload is kept in the session as its bytes and a digest of them. The checks and the
descriptions are cached on that digest, and an analysis is kept, by digest, for every session
of this server (the last few only), so moving a control or switching pages never cleans or
predicts again.

pdeeg functions are looked up on their modules when called (``predictor.predict_files``,
``pdeeg.config.load_config``), so that the tests can stand in for them.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import streamlit as st

import pdeeg.config
from pdeeg import demo
from pdeeg.predict import predictor
from pdeeg.predict.features import FeatureError, clean_input, work_files
from pdeeg.predict.inputs import READERS, InputError, read_recording

ROOT = Path(__file__).resolve().parents[2]

# Session state.
FILES = "nl_files"  # tuple of (name, bytes), the upload being worked on
DIGEST = "nl_digest"
SAME_PATIENT = "nl-same-patient"  # the checkbox's key
MODE = "nl-mode"  # the segmented control's key: SINGLE or PAIRED
SINGLE, PAIRED = "single", "paired"
UPLOADS = "nl_uploads"  # how many times the upload was cleared: part of the uploaders' keys
# Analyses kept for the server, most recent last.
_KEEP = 3

# The steps of the pipeline, in order, as the app shows them.
STEPS = ("upload", "validate", "preprocess", "features", "evaluate")


# --- configuration and the saved models ------------------------------------------------------


@st.cache_resource
def settings() -> pdeeg.config.Config:
    return pdeeg.config.load_config(os.environ.get(pdeeg.config.CONFIG_DIR_ENV) or ROOT / "configs")


@st.cache_resource
def trained() -> dict[str, Any] | None:
    """Who the final models were trained on, or None before `pdeeg train-final`."""
    return predictor.trained_on(settings())


@st.cache_resource
def evidence() -> dict[str, dict[str, Any]]:
    """How each saved stage did in cross-validation (pdeeg.predict.predictor.stage_evidence)."""
    return predictor.stage_evidence(settings())


def disclaimer() -> str:
    """The disclaimer, with who the models were trained on (from their metadata)."""
    config = settings()
    who = trained()
    if who is None:
        return (
            f"**{config.predict.disclaimer}.** No models have been trained here yet "
            f"(`pdeeg train-final`); they learn from one dataset (OpenNeuro "
            f"{config.data.dataset.id}) and are not validated on other recordings."
        )
    return (
        f"**{config.predict.disclaimer}.** Trained on {who['participants']} people from one "
        f"dataset (OpenNeuro {who['dataset']['id']}); not validated on other recordings."
    )


# --- the upload ---------------------------------------------------------------------------------


def files() -> tuple[tuple[str, bytes], ...] | None:
    return st.session_state.get(FILES)


def key() -> str | None:
    return st.session_state.get(DIGEST)


def paired() -> bool:
    current = files()
    return len(current) == 2 if current else st.session_state.get(MODE) == PAIRED


def digest(uploads: tuple[tuple[str, bytes], ...]) -> str:
    hasher = hashlib.sha256()
    for name, content in uploads:
        hasher.update(name.encode("utf-8") + b"\0" + len(content).to_bytes(8, "big"))
        hasher.update(content)
    return hasher.hexdigest()


def load(uploads: tuple[tuple[str, bytes], ...]) -> None:
    st.session_state[FILES] = uploads
    st.session_state[DIGEST] = digest(uploads)


def clear() -> None:
    """Forget the upload in this session (its analysis stays cached for the server).

    The uploaders get new keys, so that none of them brings the old file back.
    """
    for name in (FILES, DIGEST, SAME_PATIENT):
        st.session_state.pop(name, None)
    st.session_state[UPLOADS] = st.session_state.get(UPLOADS, 0) + 1


def write_upload(folder: str | Path, index: int, name: str, content: bytes) -> Path:
    """An upload as a file under its own name (one folder per upload: two may share a name)."""
    path = Path(folder) / f"upload{index + 1}" / Path(name).name
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path


def _header(path: Path) -> dict[str, Any] | None:
    """What the file's header says, before any check: rate, channels, length."""
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        return None
    try:
        raw = reader(path, preload=False, verbose="error")
    except Exception:
        return None
    sfreq = float(raw.info["sfreq"])
    return {
        "sfreq": sfreq,
        "n_channels": len(raw.ch_names),
        "n_eeg": len(mne.pick_types(raw.info, eeg=True)),
        "duration_s": raw.n_times / sfreq,
    }


@st.cache_data(show_spinner="Reading and checking the recording.", max_entries=16)
def checked(digest: str, _uploads: tuple[tuple[str, bytes], ...]) -> dict[str, Any]:
    """Each file's checks (pdeeg.predict.inputs.read_recording), and the pair's error if any.

    ``reports`` has, per file: ``name``, ``size``, ``format``, ``header`` (or None),
    ``checks`` (dicts of pdeeg.predict.inputs.Check), ``error`` (None when it passed) and,
    when it passed, ``facts`` from the checked recording.
    """
    config = settings()
    reports, recordings = [], []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        for index, (name, content) in enumerate(_uploads):
            path = write_upload(folder, index, name, content)
            report = {
                "name": name,
                "size": len(content),
                "format": path.suffix.lstrip(".").upper() or "?",
                "header": _header(path),
                "facts": None,
            }
            try:
                recording = read_recording(path, config)
            except InputError as exc:
                report |= {"checks": [asdict(c) for c in exc.checks], "error": str(exc)}
            else:
                recordings.append(recording)
                report |= {
                    "checks": [asdict(c) for c in recording.checks],
                    "error": None,
                    "facts": {
                        "sfreq": float(recording.raw.info["sfreq"]),
                        "n_channels": len(recording.raw.ch_names),
                        "available_s": recording.available_s,
                        "median_amplitude_uv": recording.median_amplitude_uv,
                        "notes": list(recording.notes),
                    },
                }
            reports.append(report)
        pair_error = None
        if len(recordings) == 2:
            try:
                predictor.check_pair(recordings)
            except InputError as exc:
                pair_error = str(exc)
    return {"reports": reports, "pair_error": pair_error}


def validation() -> dict[str, Any] | None:
    """The checks of the current upload, or None without one."""
    current = files()
    return None if current is None else checked(key(), current)


def usable(checks: dict[str, Any] | None) -> bool:
    """Whether every file passed its checks (and two files make a pair)."""
    return (
        bool(checks)
        and not checks["pair_error"]
        and all(r["error"] is None for r in checks["reports"])
    )


# --- the analysis -------------------------------------------------------------------------------


@dataclass
class Described:
    """One cleaned recording, as the figures need it.

    ``series`` are the MFDFA series (pdeeg.demo.series); ``log`` is the cleaning log;
    ``before_uv`` and ``cleaned_uv`` are the analysed window before and after cleaning
    (channels x samples, µV, float32); ``band_power`` is per channel (pdeeg.demo.band_power);
    ``qc`` holds the QC figures the cleaning drew (PNG bytes: ``psd``, and ``ica`` when it
    removed components).
    """

    series: demo.Series
    log: dict[str, Any]
    before_uv: np.ndarray
    cleaned_uv: np.ndarray
    band_power: pd.DataFrame
    qc: dict[str, bytes] = field(default_factory=dict)


@dataclass
class Analysis:
    """What one run gave: ``result`` of predict_files (or None, with ``error`` and the step
    that failed), the descriptions per file (None for one that could not be cleaned), and
    the steps as they ran, with how long each took."""

    result: dict[str, Any] | None
    error: str | None
    failed_step: str | None
    described: list[Described | None]
    log: list[dict[str, Any]]
    seconds: float
    finished: str


@st.cache_resource
def _store() -> OrderedDict[str, Analysis]:
    return OrderedDict()


def analysis(digest: str | None = None) -> Analysis | None:
    """The analysis of the current upload (or of ``digest``), if it has been run."""
    digest = digest or key()
    return None if digest is None else _store().get(digest)


def describe(path: Path, entry: dict[str, Any], index: int, work: Path, config) -> Described | None:
    """The cleaned data of one input, from predict_files' work files when it made them, else
    cleaned here as `predict` cleans (a training recording, or no usable models)."""
    try:
        recording = read_recording(path, config)
    except InputError:
        return None
    name = f"input{index + 1}"
    if "cleaned" in entry:
        cleaned, log = Path(entry["cleaned"]), Path(entry["cleaning_log"])
        figures = work_files(config, cleaned.parent.parent, name)
    else:
        own = work / f"describe{index + 1}"
        try:
            clean_input(recording.raw, config, own, name=name, source=path.name)
        except FeatureError:
            return None
        figures = work_files(config, own, name)
        cleaned, log = figures["raw"], figures["log"]
    raw = mne.io.read_raw_fif(cleaned, preload=True, verbose="error")
    qc = {kind: figures[kind].read_bytes() for kind in ("psd", "ica") if figures[kind].is_file()}
    return Described(
        series=demo.series(raw, config),
        log=json.loads(log.read_text(encoding="utf-8")),
        before_uv=demo.before_cleaning(recording.raw, config).astype(np.float32),
        cleaned_uv=(raw.get_data() * 1e6).astype(np.float32),
        band_power=demo.band_power(raw, config),
        qc=qc,
    )


def run_analysis(
    digest: str,
    uploads: tuple[tuple[str, bytes], ...],
    say: Callable[[str], None],
    config: pdeeg.config.Config,
    store: OrderedDict[str, Analysis],
) -> Analysis:
    """Predict the uploads through `pdeeg predict`, then describe their cleaned data.

    ``say`` is told each step as it starts. The analysis is kept in ``store`` under
    ``digest``. Nothing here calls Streamlit, so that it can run outside a page's script.
    """
    started = time.perf_counter()
    steps: list[dict[str, Any]] = []

    def step(name: str, detail: str) -> None:
        now = time.perf_counter() - started
        if steps:
            steps[-1]["seconds"] = now - steps[-1]["start"]
        steps.append({"step": name, "detail": detail, "start": now, "seconds": None})
        say(detail)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        paths = [write_upload(folder, i, name, data) for i, (name, data) in enumerate(uploads)]
        work = Path(folder) / "work"
        result, error, failed = None, None, None
        step("start", "Checking the inputs again, looking them up among the training recordings")
        try:
            result = predictor.predict_files(
                config,
                paths,
                out_dir=work,
                n_jobs=config.mfdfa.extraction.n_jobs,
                say=lambda message: step("features", message[:1].upper() + message[1:]),
            )
        except InputError as exc:
            error, failed = str(exc), "validate"
        except predictor.PredictError as exc:
            error, failed = str(exc), "evaluate"
        except FeatureError as exc:
            error = str(exc)
            failed = "preprocess" if error.startswith("too short to clean") else "features"
        described = []
        for index, path in enumerate(paths):
            entry = result["inputs"][index] if result else {}
            name = uploads[index][0]
            step(
                "describe",
                f"Describing {name}: MFDFA series and band power"
                if "cleaned" in entry
                else f"Cleaning {name} as `predict` does (ICA takes a minute or two), then "
                "describing it",
            )
            described.append(describe(path, entry, index, work, config))
    seconds = time.perf_counter() - started
    steps[-1]["seconds"] = seconds - steps[-1]["start"]
    done = Analysis(
        result=result,
        error=error,
        failed_step=failed,
        described=described,
        log=steps,
        seconds=seconds,
        finished=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    with _LOCK:
        store[digest] = done
        store.move_to_end(digest)
        while len(store) > _KEEP:
            store.popitem(last=False)
    return done


# --- the analysis as a background job ----------------------------------------------------------
#
# Cleaning and MFDFA take minutes. Run inside a page's script, the analysis would be stopped by
# the next click on a control or a page link (Streamlit interrupts the running script), so it
# runs in a thread of its own and every page shows its progress until it is done.


@dataclass
class Job:
    """An analysis running (or finished) in the background: what it said, and when."""

    digest: str
    started: float = field(default_factory=time.perf_counter)
    messages: list[tuple[float, str]] = field(default_factory=list)
    done: bool = False
    crashed: str | None = None  # an unexpected error, which the pages show

    def say(self, message: str) -> None:
        self.messages.append((time.perf_counter() - self.started, message))

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started


_LOCK = threading.Lock()


@st.cache_resource
def _jobs() -> dict[str, Job]:
    return {}


def job(digest: str | None = None) -> Job | None:
    """The background job of the current upload (or of ``digest``), running or finished."""
    digest = digest or key()
    return None if digest is None else _jobs().get(digest)


def running() -> bool:
    current = job()
    return current is not None and not current.done


def start_analysis(digest: str, uploads: tuple[tuple[str, bytes], ...]) -> Job:
    """Start the analysis of ``uploads`` in the background, unless it is running already."""
    config, store, jobs = settings(), _store(), _jobs()
    with _LOCK:
        current = jobs.get(digest)
        if current is not None and not current.done:
            return current
        work = jobs[digest] = Job(digest)

    def target() -> None:
        try:
            run_analysis(digest, uploads, work.say, config, store)
        except Exception as exc:  # shown on the page, with the type, rather than lost
            work.crashed = f"{type(exc).__name__}: {exc}"
        finally:
            work.done = True

    threading.Thread(target=target, name=f"neurolab-{digest[:8]}", daemon=True).start()
    return work


def forget_job(digest: str) -> None:
    with _LOCK:
        _jobs().pop(digest, None)


# --- descriptions on demand ------------------------------------------------------------------


def _described(digest: str, index: int) -> Described:
    found = analysis(digest)
    if found is None or found.described[index] is None:
        raise LookupError("this recording has not been analysed")
    return found.described[index]


@st.cache_data(show_spinner="MFDFA of every channel.", max_entries=48)
def channel_table(digest: str, index: int, name: str, fit_range: str) -> pd.DataFrame:
    """The MFDFA features of every channel on a fit range (pdeeg.demo.channel_features)."""
    return demo.channel_features(_described(digest, index).series, name, fit_range, settings())


@st.cache_data(show_spinner=False, max_entries=64)
def fluctuations(
    digest: str, index: int, name: str, channel: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fq(s) of one channel (pdeeg.demo.fluctuation_functions)."""
    return demo.fluctuation_functions(_described(digest, index).series, name, channel, settings())


@st.cache_data(show_spinner=False, max_entries=64)
def spectrum(digest: str, index: int, name: str, fit_range: str, channel: str) -> pd.DataFrame:
    """h(q), alpha and f(alpha) of one channel (pdeeg.demo.spectrum)."""
    return demo.spectrum(_described(digest, index).series, name, fit_range, channel, settings())


# --- where the pipeline stands -------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One step as shown: ``state`` is done, failed, attention, ready or pending."""

    name: str
    title: str
    state: str
    detail: str


def _seconds(found: Analysis, names: tuple[str, ...]) -> str:
    total = sum(s["seconds"] or 0.0 for s in found.log if s["step"] in names)
    return f"{total:.0f} s" if total >= 1 else "<1 s"


def pipeline() -> list[Step]:
    """The state of each step for the current upload, from what has actually run."""
    current = files()
    checks = validation()
    found = analysis()
    steps = []
    if current is None:
        steps.append(Step("upload", "Upload", "ready", "Waiting for a .bdf or .edf file"))
    else:
        size = sum(len(content) for _, content in current)
        steps.append(
            Step("upload", "Upload", "done", f"{len(current)} file(s), {size / 1e6:.1f} MB")
        )
    if checks is None:
        steps.append(Step("validate", "Validate", "pending", "Channels, rate, length, amplitude"))
    elif usable(checks):
        notes = sum(c["status"] == "note" for r in checks["reports"] for c in r["checks"])
        detail = "All checks passed" + (f", {notes} note(s)" if notes else "")
        steps.append(Step("validate", "Validate", "attention" if notes else "done", detail))
    else:
        steps.append(Step("validate", "Validate", "failed", "A check failed: nothing else runs"))

    work = job()
    if found is None and work is not None and not work.done:
        steps.append(
            Step("preprocess", "Preprocess", "running", "Running: its progress is shown above")
        )
        steps.append(
            Step("features", "Extract features", "pending", "In the same run of pdeeg predict")
        )
        steps.append(Step("evaluate", "Evaluate model", "pending", "After the features"))
        return steps
    if found is None and work is not None and work.crashed:
        steps.append(Step("preprocess", "Preprocess", "failed", work.crashed))
        steps.append(Step("features", "Extract features", "pending", "Not reached"))
        steps.append(Step("evaluate", "Evaluate model", "pending", "Not reached"))
        return steps
    if found is None:
        ready = "ready" if usable(checks) else "pending"
        waiting = "Run the analysis" if usable(checks) else "After validation"
        steps.append(Step("preprocess", "Preprocess", ready, waiting))
        steps.append(Step("features", "Extract features", "pending", "MFDFA and band power"))
        steps.append(Step("evaluate", "Evaluate model", "pending", "Stage 1, then stage 2"))
        return steps

    cleaned = [d for d in found.described if d is not None]
    if found.failed_step == "preprocess" or not cleaned:
        steps.append(Step("preprocess", "Preprocess", "failed", found.error or "Not cleaned"))
    else:
        removed = sum(len(d.log["ica"]["removed"]) for d in cleaned)
        steps.append(
            Step(
                "preprocess",
                "Preprocess",
                "done",
                f"Filtered, re-referenced, ICA removed {removed} component(s)",
            )
        )
    result = found.result
    if found.failed_step == "features":
        steps.append(Step("features", "Extract features", "failed", found.error or ""))
    elif result is not None and not result["training_recording"]:
        steps.append(
            Step(
                "features",
                "Extract features",
                "done",
                f"MFDFA and band power for the models ({_seconds(found, ('features',))})",
            )
        )
    elif cleaned:
        steps.append(
            Step(
                "features",
                "Extract features",
                "attention",
                "MFDFA computed for display only; the models did not need features",
            )
        )
    else:
        steps.append(Step("features", "Extract features", "pending", "Not reached"))

    if found.failed_step == "evaluate":
        steps.append(Step("evaluate", "Evaluate model", "failed", found.error or ""))
    elif found.failed_step is not None:
        steps.append(Step("evaluate", "Evaluate model", "pending", "Not reached"))
    elif result["training_recording"]:
        steps.append(
            Step(
                "evaluate",
                "Evaluate model",
                "attention",
                "Training recording: out-of-fold results shown instead",
            )
        )
    else:
        ran = sum("skipped" not in s for s in result["stages"])
        steps.append(
            Step("evaluate", "Evaluate model", "done", f"{ran} stage(s) ran on the final models")
        )
    return steps


def headline() -> tuple[str, str]:
    """A short status of the current upload, and its kind (ok, warn, err, run or mute)."""
    current = files()
    if current is None:
        return "No recording loaded", "mute"
    checks = validation()
    if not usable(checks):
        return "Validation failed", "err"
    found = analysis()
    work = job()
    if found is None and work is not None and not work.done:
        return "Analysing…", "run"
    if found is None and work is not None and work.crashed:
        return "Analysis failed", "err"
    if found is None:
        return "Validated · ready to analyse", "run"
    if found.failed_step in ("validate", "preprocess", "features"):
        return "Analysis failed", "err"
    if found.failed_step == "evaluate":
        return f"Cleaned in {found.seconds:.0f} s · no models", "warn"
    return f"Analysed in {found.seconds:.0f} s", "ok"


def export_result(result: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """``result`` with the uploads' names for paths, and without the deleted work files."""
    clean = json.loads(json.dumps(result, default=str))
    for entry, name in zip(clean["inputs"], names, strict=True):
        entry["path"] = name
        entry.pop("cleaned", None)
        entry.pop("cleaning_log", None)
    return clean
