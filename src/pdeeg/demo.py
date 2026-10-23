"""The analysis behind the research demo (app/streamlit_app.py): one EEG file, described.

A BDF or EDF file is read, cleaned as the pipeline cleans the dataset (configs/preprocessing.yaml)
and analysed with the MFDFA settings of configs/features/mfdfa.yaml. Nothing here classifies a
recording or compares it with anyone: there is no model, and no statement about disease.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from pdeeg.config import Config
from pdeeg.features.curves import fit_curves
from pdeeg.features.extract import bad_sample_mask, fit_features, range_scales, series_masks
from pdeeg.features.mfdfa import make_qs, mfdfa
from pdeeg.features.scaling import analysis_series, fit_ranges, inspection_scales
from pdeeg.preprocessing.pipeline import RecordingTooShort, preprocess_raw

READERS = {".bdf": mne.io.read_raw_bdf, ".edf": mne.io.read_raw_edf}
_VOLTS_TO_UV = 1e6


class DemoError(Exception):
    """The file cannot be analysed; the message says why, in words for the person using it."""


def read_eeg(path: str | Path, montage: str) -> mne.io.BaseRaw:
    """The scalp channels of ``montage`` from a BDF or EDF file, with the montage applied.

    Triggers on a stim channel become annotations named after their code, as the dataset's
    event files name them, so that the crop can start at the configured trigger.
    """
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise DemoError(f"expected a .bdf or .edf file, got {path.name}")
    try:
        raw = reader(path, preload=True, verbose="error")
    except Exception as exc:  # the readers raise many kinds of error for a malformed file
        raise DemoError(f"{path.name} could not be read: {exc}") from exc
    stim = mne.pick_types(raw.info, stim=True)
    if len(stim):
        events = mne.find_events(raw, shortest_event=1, verbose="error")
        if len(events):
            onsets = (events[:, 0] - raw.first_samp) / raw.info["sfreq"]
            triggers = mne.Annotations(onsets, 0.0, [str(code) for code in events[:, 2]])
            raw.set_annotations(raw.annotations + triggers)
    layout = mne.channels.make_standard_montage(montage)
    absent = [name for name in layout.ch_names if name not in raw.ch_names]
    if absent:
        raise DemoError(
            f"the file has no channel named {', '.join(absent)}; the demo needs the channels "
            f"of the {montage} montage under their standard names"
        )
    raw.pick([name for name in raw.ch_names if name in layout.ch_names])
    raw.set_channel_types(dict.fromkeys(raw.ch_names, "eeg"))
    raw.set_montage(layout)
    # Detached from the file, which may be gone by the time the recording is cleaned (an upload
    # is read from a temporary file): MNE checks that a recording's file exists when it crops.
    detached = mne.io.RawArray(raw.get_data(), raw.info, first_samp=raw.first_samp, verbose="error")
    detached.set_annotations(raw.annotations)
    return detached


def clean(raw: mne.io.BaseRaw, config: Config, line_freq: float) -> tuple[mne.io.BaseRaw, dict]:
    """The recording cleaned as the dataset is, and the log of what was done."""
    try:
        cleaned, _, log = preprocess_raw(raw, config.preprocessing, line_freq)
    except RecordingTooShort as exc:
        raise DemoError(f"the recording is too short for the configured crop: {exc}") from exc
    return cleaned, log


@dataclass(frozen=True)
class Series:
    """The series MFDFA is run on (``broadband`` and one envelope per band), channels x samples.

    ``masks`` marks, per series, the samples inside a stretch annotated bad.
    """

    values: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    channels: list[str]
    sfreq: float


def series(cleaned: mne.io.BaseRaw, config: Config) -> Series:
    sfreq = float(cleaned.info["sfreq"])
    values = analysis_series(cleaned.get_data() * _VOLTS_TO_UV, sfreq, config.mfdfa)
    masks = series_masks(bad_sample_mask(cleaned), sfreq, config.mfdfa)
    return Series(values, masks, list(cleaned.ch_names), sfreq)


def fluctuation_functions(
    data: Series, name: str, channel: str, config: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Fq(s)`` of one channel over the dense scales of the scaling inspection.

    Returns the scales in seconds, the moment orders, and ``Fq`` (orders x scales).
    """
    settings = config.mfdfa
    x = data.values[name][data.channels.index(channel)]
    scales = inspection_scales(name, x.size, data.sfreq, settings)
    qs = np.asarray(settings.inspection.qs)
    result = mfdfa(
        x,
        scales,
        qs,
        order=settings.detrend_order,
        min_variance_ratio=settings.min_variance_ratio,
    )
    return scales / data.sfreq, qs, result.Fq


def spectrum(data: Series, name: str, fit_range: str, channel: str, config: Config) -> pd.DataFrame:
    """h(q), alpha and f(alpha) of one channel on a fit range, without the bad windows."""
    settings = config.mfdfa
    i = data.channels.index(channel)
    x = data.values[name][i]
    scales = range_scales(name, x.size, data.sfreq, settings)[fit_range]
    qs = make_qs(settings.q_min, settings.q_max, settings.q_step)
    return pd.DataFrame({"q": qs, **fit_curves(x, scales, qs, settings, data.masks[name])})


def channel_features(data: Series, name: str, fit_range: str, config: Config) -> pd.DataFrame:
    """The MFDFA features of every channel on a fit range, without the bad windows."""
    settings = config.mfdfa
    qs = make_qs(settings.q_min, settings.q_max, settings.q_step)
    rows = []
    for i, channel in enumerate(data.channels):
        x = data.values[name][i]
        scales = range_scales(name, x.size, data.sfreq, settings)[fit_range]
        rows.append({"channel": channel} | fit_features(x, scales, qs, settings, data.masks[name]))
    return pd.DataFrame(rows)


def ranges_in_seconds(name: str, config: Config) -> dict[str, tuple[float, float]]:
    """The configured fit ranges of the series ``name``."""
    return fit_ranges(config.mfdfa, name)
