"""Per-recording preprocessing: crop, band-pass, re-reference, ICA clean-up, bad-segment marking.

``preprocess_raw`` does the signal processing on a Raw object in memory. ``process_recording``
wraps it with loading, saving, logging and caching for one row of the recordings table, and
``run`` does that for many rows in parallel.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import traceback
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mne
import mne_icalabel
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne.preprocessing import ICA
from scipy.signal import welch

import pdeeg
from pdeeg.config import Config, CropConfig, LineNoiseConfig, PreprocessingConfig
from pdeeg.data.bids import load_raw
from pdeeg.preprocessing import qc
from pdeeg.preprocessing.ica import CLASSES, classify_components, fit_ica, select_components
from pdeeg.preprocessing.segments import annotate_bad_segments
from pdeeg.viz.style import save

# Part of the cache key. Raise it when the processing changes in a way the config cannot show,
# so that outputs written by older code are redone.
PIPELINE_VERSION = 1

# Spectra kept in the log for the QC figures.
_PSD_WINDOW_S = 4.0
_PSD_FMAX = 100.0
# A mains peak is power within _PEAK_HALF_WIDTH of the candidate frequency, compared with the
# spectrum between _NEIGHBOUR_NEAR and _NEIGHBOUR_FAR away on either side.
_PEAK_HALF_WIDTH = 0.25
_NEIGHBOUR_NEAR = 1.0
_NEIGHBOUR_FAR = 3.0

_NAMING_WARNING = ".*does not conform to MNE naming conventions.*"


class RecordingTooShort(Exception):
    """The recording has less data after the crop start than the configured duration."""


def crop_window(raw: mne.io.BaseRaw, config: CropConfig) -> tuple[int, str]:
    """First sample of the window to keep, and what it was anchored on."""
    if config.start_event is not None and config.start_event in raw.annotations.description:
        events, _ = mne.events_from_annotations(
            raw, event_id={config.start_event: 1}, verbose="error"
        )
        return int(events[0, 0] - raw.first_samp), f"event {config.start_event}"
    return 0, "file start"


def median_psd(raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray]:
    """Welch spectrum in uV^2/Hz, median over channels, from 0 Hz up to ``_PSD_FMAX``."""
    sfreq = raw.info["sfreq"]
    nperseg = min(int(round(_PSD_WINDOW_S * sfreq)), raw.n_times)
    freqs, power = welch(raw.get_data() * 1e6, fs=sfreq, nperseg=nperseg, axis=1)
    keep = freqs <= _PSD_FMAX
    return freqs[keep], np.median(power[:, keep], axis=0)


def detect_line_noise(
    freqs: np.ndarray, power: np.ndarray, config: LineNoiseConfig
) -> tuple[float | None, dict[str, float | None]]:
    """Which candidate mains frequency, if any, stands out of the spectrum.

    Returns the detected frequency (None when no candidate reaches ``min_peak_ratio``) and
    every candidate's peak height relative to the neighbouring spectrum.
    """
    ratios: dict[str, float | None] = {}
    for candidate in config.candidates:
        distance = np.abs(freqs - candidate)
        peak = power[distance <= _PEAK_HALF_WIDTH]
        around = power[(distance >= _NEIGHBOUR_NEAR) & (distance <= _NEIGHBOUR_FAR)]
        usable = peak.size > 0 and around.size > 0 and around.mean() > 0
        ratios[f"{candidate:g}"] = float(peak.mean() / around.mean()) if usable else None
    measured = {float(key): ratio for key, ratio in ratios.items() if ratio is not None}
    if not measured:
        return None, ratios
    best = max(measured, key=measured.get)
    return (best if measured[best] >= config.min_peak_ratio else None), ratios


def _reference(raw: mne.io.BaseRaw, reference: str | tuple[str, ...]) -> mne.io.BaseRaw:
    channels = "average" if reference == "average" else list(reference)
    return raw.set_eeg_reference(channels, projection=False)


def _decibels(power: np.ndarray) -> list[float]:
    return np.round(10.0 * np.log10(np.maximum(power, 1e-30)), 3).tolist()


def preprocess_raw(
    raw: mne.io.BaseRaw, config: PreprocessingConfig, line_freq: float
) -> tuple[mne.io.BaseRaw, ICA, dict[str, Any]]:
    """Clean one recording.

    ``raw`` holds the scalp EEG channels with a montage and is left untouched. ``line_freq``
    is the mains frequency to notch out of the copy that ICA is fitted on. Returns the cleaned
    continuous recording with its bad stretches annotated, the fitted ICA, and a log.

    Raises :class:`RecordingTooShort` if the recording cannot supply ``crop.duration``.
    """
    sfreq = raw.info["sfreq"]
    n_keep = int(round(config.crop.duration * sfreq))
    start, anchor = crop_window(raw, config.crop)
    available = (raw.n_times - start) / sfreq
    if start + n_keep > raw.n_times:
        raise RecordingTooShort(
            f"{available:.1f} s available from {anchor}, {config.crop.duration:.1f} s needed"
        )
    sidecar_line_freq = raw.info["line_freq"]
    raw = raw.copy().crop(start / sfreq, (start + n_keep - 1) / sfreq).load_data()
    raw.set_annotations(None)
    dc_offset_uv = float(np.median(np.abs(raw.get_data().mean(axis=1))) * 1e6)
    # Mains interference is largely common to all electrodes, so an average reference cancels
    # most of it. Look for it before re-referencing.
    detected_line_freq, peak_ratios = detect_line_noise(*median_psd(raw), config.line_noise)

    # Three versions of the same window: unfiltered (for the "before" spectrum), filtered the
    # way ICLabel expects (to fit and classify ICA), and the band-passed data that is kept.
    before = _reference(raw.copy(), config.reference)
    fit = raw.copy().filter(config.ica.fit_l_freq, config.ica.fit_h_freq)
    if config.ica.fit_notch and line_freq < sfreq / 2:
        fit.notch_filter([line_freq])
    _reference(fit, config.reference)
    raw.filter(
        config.filter.l_freq,
        config.filter.h_freq,
        l_trans_bandwidth=config.filter.l_trans_bandwidth,
        h_trans_bandwidth=config.filter.h_trans_bandwidth,
        method="fir",
        fir_design="firwin",
        fir_window=config.filter.fir_window,
        phase=config.filter.phase,
    )
    _reference(raw, config.reference)

    freqs, power_before = median_psd(before)

    ica, rank = fit_ica(fit, config.ica)
    probabilities = classify_components(fit, ica)
    removed = select_components(probabilities, config.ica)
    variance_before = raw.get_data().var(axis=1).sum()
    ica.exclude = [component["index"] for component in removed]
    ica.apply(raw)
    data = raw.get_data() * 1e6
    variance_after = (data * 1e-6).var(axis=1).sum()

    annotations, segments = annotate_bad_segments(raw, config.bad_segments)
    raw.set_annotations(annotations)

    _, power_after = median_psd(raw)
    sd_uv = data.std(axis=1)
    predicted = [CLASSES[i] for i in probabilities.argmax(axis=1)]
    channel_peak = segments.pop("channel_peak_percent")
    channel_flat = segments.pop("channel_flat_percent")
    slow = power_before[(freqs > 0) & (freqs <= 0.5)].mean()
    delta = power_before[(freqs >= 1.0) & (freqs <= 4.0)].mean()
    log = {
        "sfreq": float(sfreq),
        "n_times": int(raw.n_times),
        "n_channels": len(raw.ch_names),
        "crop": {
            "start_s": start / sfreq,
            "duration_s": n_keep / sfreq,
            "anchor": anchor,
            "available_s": available,
        },
        "line_noise": {
            "detected_hz": detected_line_freq,
            "peak_ratio": peak_ratios,
            "sidecar_hz": None if sidecar_line_freq is None else float(sidecar_line_freq),
            "configured_hz": float(line_freq),
        },
        "ica": {
            "method": config.ica.method,
            "rank": rank,
            "n_components": int(ica.n_components_),
            "n_iter": int(ica.n_iter_),
            "label_counts": {name: predicted.count(name) for name in CLASSES},
            "removed": removed,
            "variance_removed_percent": float(100.0 * (1.0 - variance_after / variance_before)),
        },
        "bad_segments": segments,
        "summary": {
            "median_sd_uv": float(np.median(sd_uv)),
            "max_sd_ratio": float(sd_uv.max() / np.median(sd_uv)),
            "raw_dc_offset_uv": dc_offset_uv,
            # Power below 0.5 Hz relative to 1-4 Hz in the unfiltered signal. A raw recording
            # drifts, so this is large; a recording that was already high-passed has none.
            "raw_slow_db": float(10.0 * np.log10(max(slow, 1e-30) / max(delta, 1e-30))),
        },
        "channels": {
            name: {
                "sd_uv": float(sd),
                "peak_percent": float(peak),
                "flat_percent": float(flat),
            }
            for name, sd, peak, flat in zip(
                raw.ch_names, sd_uv, channel_peak, channel_flat, strict=True
            )
        },
        "psd": {
            "freqs_hz": np.round(freqs, 3).tolist(),
            "before_db": _decibels(power_before),
            "after_db": _decibels(power_after),
        },
    }
    return raw, ica, log


def config_hash(config: Config) -> str:
    """Digest of everything in the configuration that changes a cleaned recording."""
    settings = dataclasses.asdict(config.preprocessing)
    for name in ("output", "qc", "n_jobs"):  # where things go and how fast, not what they are
        del settings[name]
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "preprocessing": settings,
        "montage": config.data.dataset.montage,
        "line_freq": config.data.dataset.line_freq,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def output_paths(config: Config, subject: str, session: str) -> dict[str, Path]:
    """Where one recording's cleaned data, log and QC figures are written."""
    stem = f"sub-{subject}_ses-{session}"
    out_dir = config.preprocessing.output.dir
    figures_dir = config.preprocessing.qc.figures_dir
    return {
        "raw": out_dir / f"{stem}_clean.fif",
        "log": out_dir / f"{stem}_clean.json",
        "psd": figures_dir / f"{stem}_psd.png",
        "ica": figures_dir / f"{stem}_ica.png",
    }


def recording_inputs(root: Path, bids_path: str) -> list[Path]:
    """The data file of a recording and the sidecar files that describe it."""
    data_file = root / bids_path
    prefix = data_file.name.rsplit("_", 1)[0]
    return sorted(data_file.parent.glob(f"{prefix}_*"))


def read_log(path: Path) -> dict[str, Any] | None:
    """A recording's JSON log, or None if it is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_clean(path: str | Path, *, preload: bool = False) -> mne.io.BaseRaw:
    """Read a cleaned recording written by :func:`process_recording`."""
    with warnings.catch_warnings():
        # "_clean.fif" is this project's name for the file, not one of MNE's suffixes.
        warnings.filterwarnings("ignore", message=_NAMING_WARNING)
        return mne.io.read_raw_fif(path, preload=preload, verbose="error")


def is_current(recording: Mapping[str, Any], config: Config) -> bool:
    """Whether a recording's outputs can be reused instead of being computed again.

    They can when they all exist, were written with this configuration (same
    :func:`config_hash`) and are newer than the recording's data file and sidecars.
    """
    paths = output_paths(config, recording["subject"], recording["session"])
    inputs = recording_inputs(config.data.paths.raw, recording["bids_path"])
    log = read_log(paths["log"])
    if log is None or log.get("config_hash") != config_hash(config):
        return False
    outputs = [paths["raw"], paths["log"]]
    outputs += [paths[kind] for kind, name in log.get("figures", {}).items() if name]
    if not all(path.is_file() for path in outputs):
        return False
    newest_input = max(path.stat().st_mtime for path in inputs)
    return min(path.stat().st_mtime for path in outputs) >= newest_input


def process_recording(
    recording: Mapping[str, Any], config: Config, *, force: bool = False
) -> dict[str, Any]:
    """Preprocess one row of the recordings table to disk and report what happened.

    The returned ``status`` is ``processed``, ``cached`` (outputs were current and ``force``
    was not set), ``too_short`` or ``failed``. Errors are reported rather than raised so that
    one recording cannot stop a batch.
    """
    subject, session = recording["subject"], recording["session"]
    result: dict[str, Any] = {"subject": subject, "session": session, "detail": ""}
    started = time.perf_counter()
    try:
        if not force and is_current(recording, config):
            return result | {"status": "cached", "seconds": 0.0}
        paths = output_paths(config, subject, session)

        # The log marks a finished recording, so it goes first and is written last.
        paths["log"].unlink(missing_ok=True)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        with mne.use_log_level("WARNING"):
            raw = load_raw(
                SimpleNamespace(**recording), config.data.paths.raw, config.data.dataset.montage
            )
            clean, ica, log = preprocess_raw(
                raw, config.preprocessing, config.data.dataset.line_freq
            )
            title = f"sub-{subject} ses-{session}"
            save(qc.plot_psd(log, config.preprocessing, title), paths["psd"])
            removed = log["ica"]["removed"]
            if removed:
                save(qc.plot_removed_components(ica, clean.info, removed, title), paths["ica"])
            else:
                paths["ica"].unlink(missing_ok=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=_NAMING_WARNING)
                clean.save(paths["raw"], overwrite=True)
        log = {
            "subject": subject,
            "session": session,
            "source": recording["bids_path"],
            "config_hash": config_hash(config),
            "pipeline_version": PIPELINE_VERSION,
            "versions": {
                "pdeeg": pdeeg.__version__,
                "mne": mne.__version__,
                "mne_icalabel": mne_icalabel.__version__,
                "numpy": np.__version__,
            },
            "figures": {
                "psd": paths["psd"].name,
                "ica": paths["ica"].name if removed else None,
            },
            **log,
        }
        paths["log"].write_text(json.dumps(log, indent=1), encoding="utf-8")
        result["status"] = "processed"
    except RecordingTooShort as exc:
        result |= {"status": "too_short", "detail": str(exc)}
    except Exception as exc:
        result |= {"status": "failed", "detail": f"{exc!r}\n{traceback.format_exc()}"}
    return result | {"seconds": round(time.perf_counter() - started, 1)}


def run(
    config: Config,
    recordings: pd.DataFrame,
    *,
    n_jobs: int | None = None,
    force: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Preprocess every row of ``recordings``; returns one status per row, in order.

    ``progress`` is called with each status as soon as it is available.
    """
    rows = recordings.to_dict("records")
    jobs = config.preprocessing.n_jobs if n_jobs is None else n_jobs
    finished = Parallel(n_jobs=jobs, return_as="generator")(
        delayed(process_recording)(row, config, force=force) for row in rows
    )
    results = []
    for result in finished:
        if progress is not None:
            progress(result)
        results.append(result)
    return results
