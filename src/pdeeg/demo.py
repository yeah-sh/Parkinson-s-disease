"""The research demo's description of one recording (app/streamlit_app.py, app/research_app.py).

The app cleans a recording as `pdeeg predict` does (pdeeg.predict.features.clean_input); here
its series are analysed with the MFDFA settings of configs/features/mfdfa.yaml, its band power
with those of configs/features/psd.yaml, and the window the cleaning kept is given as it was
before cleaning, for the app to show the two side by side. Nothing here classifies a
recording: the app's predictions come from pdeeg.predict.predictor. The one exception is
:func:`synthetic_recording`, the made-up recording behind the app's "Try a sample recording".
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd

from pdeeg.config import Config
from pdeeg.features.curves import fit_curves
from pdeeg.features.extract import bad_sample_mask, fit_features, range_scales, series_masks
from pdeeg.features.mfdfa import make_qs, mfdfa
from pdeeg.features.psd import recording_band_powers
from pdeeg.features.scaling import analysis_series, fit_ranges, inspection_scales
from pdeeg.preprocessing.pipeline import crop_window

_VOLTS_TO_UV = 1e6


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


def before_cleaning(raw: mne.io.BaseRaw, config: Config) -> np.ndarray:
    """The window the cleaning keeps, as it was before cleaning, in µV (channels x samples).

    ``raw`` is a checked recording (:func:`pdeeg.predict.inputs.read_recording`). The window
    and the re-reference are those of :func:`pdeeg.preprocessing.pipeline.preprocess_raw`,
    whose "before" spectrum is of these data: no filter and no ICA. Sample ``i`` here is
    sample ``i`` of the cleaned recording.
    """
    prep = config.preprocessing
    sfreq = raw.info["sfreq"]
    start, _ = crop_window(raw, prep.crop)
    n_keep = int(round(prep.crop.duration * sfreq))
    window = raw.copy().crop(start / sfreq, (start + n_keep - 1) / sfreq).load_data()
    reference = "average" if prep.reference == "average" else list(prep.reference)
    window.set_eeg_reference(reference, projection=False, verbose="error")
    return window.get_data() * _VOLTS_TO_UV


def band_power(cleaned: mne.io.BaseRaw, config: Config) -> pd.DataFrame:
    """Band power of every channel of a cleaned recording, computed as the training features
    were (:func:`pdeeg.features.psd.recording_band_powers`), with and without bad windows."""
    return recording_band_powers(
        cleaned.get_data() * _VOLTS_TO_UV,
        bad_sample_mask(cleaned),
        float(cleaned.info["sfreq"]),
        list(cleaned.ch_names),
        config.psd,
    )


def synthetic_recording(
    seconds: float = 200.0,
    sfreq: float = 512.0,
    seed: int = 0,
    names: list[str] | None = None,
    scale: float = 10e-6,
    montage: str = "biosemi32",
) -> mne.io.RawArray:
    """A made-up recording that passes every input check: no one's EEG.

    Pink noise mixed over the scalp, a 10 Hz alpha rhythm strongest at the back of the head,
    and trigger "1" at 5 s, as ds002778 recordings carry it. ``scale`` is the standard deviation
    of every channel in volts; ``names`` defaults to the channels of ``montage``.
    """
    rng = np.random.default_rng(seed)
    names = mne.channels.make_standard_montage(montage).ch_names if names is None else names
    n = int(seconds * sfreq)
    freqs = np.fft.rfftfreq(n, 1 / sfreq)
    freqs[0] = freqs[1]

    def pink(k):
        spectrum = np.fft.rfft(rng.standard_normal((k, n)), axis=1) / np.sqrt(freqs)
        return np.fft.irfft(spectrum, n=n, axis=1)

    data = rng.standard_normal((len(names), 8)) @ pink(8) + 0.5 * pink(len(names))
    t = np.arange(n) / sfreq
    alpha = np.sin(2 * np.pi * 10.0 * t + np.cumsum(rng.normal(0.0, 0.05, n)))
    posterior = np.array([2.0 if name[0] in "OP" else 0.3 for name in names])
    data = data / data.std(axis=1, keepdims=True) + posterior[:, None] * alpha
    data = data / data.std(axis=1, keepdims=True) * scale
    raw = mne.io.RawArray(data, mne.create_info(list(names), sfreq, "eeg"), verbose="error")
    raw.set_annotations(mne.Annotations([5.0], [0.0], ["1"]))
    return raw
