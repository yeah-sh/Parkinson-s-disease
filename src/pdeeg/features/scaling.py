"""Where does the EEG scale? Fluctuation functions and local slopes, before any exponent is fitted.

MFDFA reads an exponent off a straight stretch of ``log Fq(s)`` against ``log s``, so the
stretch has to be found first. This module computes ``Fq(s)`` over a wide, dense range of scales
for a few recordings and the local slope of every curve, and does the same for noise that has no
crossover of its own after it has been put through the filters the EEG went through, which
shows what the filters alone do.

The recordings are picked at random and called R1, R2, ...: nothing here reads a group label or
returns a subject, so fit ranges chosen from the result cannot be tuned towards a group
difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view

from pdeeg.config import Config, FilterConfig, MfdfaConfig
from pdeeg.features.envelope import band_envelope, trim_samples
from pdeeg.features.mfdfa import make_qs, make_scales, mfdfa, spectrum_features
from pdeeg.features.synthetic import power_law_noise
from pdeeg.preprocessing.pipeline import output_paths, read_clean

# Name of the series that is the cleaned EEG itself; the others are named after their band.
BROADBAND = "broadband"
# What the EEG rows of the fit table are called in its ``source`` column.
EEG = "EEG"
_VOLTS_TO_UV = 1e6
# Slack, in samples, for a fit range whose end is meant to fall on a whole sample.
_ROUNDING = 1e-6


@dataclass(frozen=True)
class SeriesCurves:
    """Fluctuation functions of one series: the broadband EEG or one band envelope.

    ``scales`` are in samples. ``fq`` has shape (recordings, channels, qs, scales). ``null``
    maps the name of a noise to its ``Fq``, shape (realisations, qs, scales).
    """

    scales: np.ndarray
    fq: np.ndarray
    null: dict[str, np.ndarray]


@dataclass(frozen=True)
class Inspection:
    """What :func:`run_inspection` returns.

    ``curves`` maps a series name to its fluctuation functions, computed for the moment orders
    ``qs``. ``fits`` has one row per series, fit range and channel of a recording (``source``
    is ``EEG``, ``unit`` is R1, R2, ...) or realisation of a noise (``source`` is its name):
    the scalar features of :func:`pdeeg.features.mfdfa.spectrum_features` over the configured
    fit range, and ``h`` at the lowest and the highest configured moment order.
    """

    sfreq: float
    n_times: int
    channels: tuple[str, ...]
    qs: np.ndarray
    curves: dict[str, SeriesCurves]
    fits: pd.DataFrame


def series_names(config: MfdfaConfig) -> tuple[str, ...]:
    """Names of the series that are analysed: the broadband EEG, then one per envelope band."""
    return (BROADBAND, *config.envelope.bands)


def fit_ranges(config: MfdfaConfig, name: str) -> dict[str, tuple[float, float]]:
    """The fit ranges, in seconds, of the series called ``name``."""
    return config.broadband.fit_ranges if name == BROADBAND else config.envelope.fit_ranges


def analysis_series(data: np.ndarray, sfreq: float, config: MfdfaConfig) -> dict[str, np.ndarray]:
    """The series MFDFA is run on, by name, for ``data`` of shape (channels, samples).

    The envelopes are shorter than the data by twice ``envelope.edge_trim``.
    """
    envelope = config.envelope
    series = {BROADBAND: data}
    for band, edges in envelope.bands.items():
        series[band] = band_envelope(
            data,
            sfreq,
            edges,
            trans_bandwidth=envelope.trans_bandwidth,
            edge_trim=envelope.edge_trim,
        )
    return series


def series_length(name: str, n_times: int, sfreq: float, config: MfdfaConfig) -> int:
    """Samples in the series ``name`` of a recording of ``n_times`` samples."""
    if name == BROADBAND:
        return n_times
    return n_times - 2 * trim_samples(config.envelope.edge_trim, sfreq)


def scales_in_range(
    fit_range: tuple[float, float], sfreq: float, n_scales: int, *, max_scale: int | None = None
) -> np.ndarray:
    """Up to ``n_scales`` window sizes in samples, log-spaced over ``fit_range`` in seconds.

    The ends are rounded inwards to whole samples and are always among the sizes. Duplicates
    are dropped, so fewer than ``n_scales`` come back when the range holds fewer integers. A
    range that reaches beyond ``max_scale`` samples is rejected.
    """
    low = math.ceil(fit_range[0] * sfreq - _ROUNDING)
    high = math.floor(fit_range[1] * sfreq + _ROUNDING)
    if not 1 <= low < high:
        raise ValueError(f"fit range {fit_range} s is empty at {sfreq:g} Hz")
    if max_scale is not None and high > max_scale:
        raise ValueError(
            f"fit range {fit_range} s ends at {high} samples, beyond the largest scale allowed "
            f"({max_scale})"
        )
    return np.unique(np.round(np.geomspace(low, high, n_scales)).astype(int))


def local_slopes(scales: np.ndarray, fq: np.ndarray, half_width: int) -> np.ndarray:
    """Slope of ``log fq`` against ``log scales`` around each scale, along the last axis.

    Each slope is a least-squares fit through the scale and ``half_width`` neighbours on either
    side. The first and last ``half_width`` scales have no slope and are NaN.
    """
    if half_width < 1:
        raise ValueError(f"half_width must be at least 1, got {half_width}")
    width = 2 * half_width + 1
    slopes = np.full(np.shape(fq), np.nan)
    if len(scales) < width:
        return slopes
    log_s = sliding_window_view(np.log(scales), width)
    log_f = sliding_window_view(np.log(fq), width, axis=-1)
    ds = log_s - log_s.mean(axis=-1, keepdims=True)
    df = log_f - log_f.mean(axis=-1, keepdims=True)
    slopes[..., half_width : len(scales) - half_width] = (df * ds).sum(axis=-1) / (ds**2).sum(
        axis=-1
    )
    return slopes


def select_recordings(n_total: int, n_recordings: int, seed: int) -> np.ndarray:
    """Row numbers of ``n_recordings`` different recordings out of ``n_total``, in random order.

    Only the size of the recordings table goes in, so the choice cannot depend on who was
    recorded. The order is random as well: R1 is not the first row of the table.
    """
    if not 1 <= n_recordings <= n_total:
        raise ValueError(f"cannot pick {n_recordings} recordings out of {n_total}")
    return np.random.default_rng(seed).choice(n_total, size=n_recordings, replace=False)


def inspection_scales(name: str, n_samples: int, sfreq: float, config: MfdfaConfig) -> np.ndarray:
    """The dense scales, in samples, on which the series ``name`` of that length is inspected."""
    inspection = config.inspection
    smallest = (
        inspection.broadband_scale_min if name == BROADBAND else inspection.envelope_scale_min
    )
    return make_scales(
        n_samples, int(round(smallest * sfreq)), config.scale_max_frac, inspection.n_scales
    )


def _fluctuations(series: np.ndarray, scales: np.ndarray, config: MfdfaConfig) -> np.ndarray:
    """``Fq`` of every row of ``series`` for the inspection's moment orders: (rows, qs, scales)."""
    return np.stack(
        [
            mfdfa(
                row,
                scales,
                config.inspection.qs,
                order=config.detrend_order,
                min_variance_ratio=config.min_variance_ratio,
            ).Fq
            for row in series
        ]
    )


def _fits(name: str, series: np.ndarray, sfreq: float, config: MfdfaConfig) -> list[dict]:
    """Features of every row of ``series`` over each configured fit range of the series ``name``."""
    qs = make_qs(config.q_min, config.q_max, config.q_step)
    max_scale = int(series.shape[-1] * config.scale_max_frac)
    rows = []
    for range_name, fit_range in fit_ranges(config, name).items():
        scales = scales_in_range(fit_range, sfreq, config.scales_per_range, max_scale=max_scale)
        for i, row in enumerate(series):
            result = mfdfa(
                row,
                scales,
                qs,
                order=config.detrend_order,
                min_variance_ratio=config.min_variance_ratio,
            )
            rows.append(
                {
                    "series": name,
                    "fit_range": range_name,
                    "row": i,
                    **spectrum_features(result, qs),
                    "h_q_min": float(result.h[0]),
                    "h_q_max": float(result.h[-1]),
                }
            )
    return rows


def _inspect_recording(
    path: Path, config: MfdfaConfig
) -> tuple[dict[str, np.ndarray], list[dict], float, int, list[str]]:
    raw = read_clean(path, preload=True)
    sfreq = float(raw.info["sfreq"])
    series = analysis_series(raw.get_data() * _VOLTS_TO_UV, sfreq, config)
    curves, fits = {}, []
    for name, values in series.items():
        scales = inspection_scales(name, values.shape[-1], sfreq, config)
        curves[name] = _fluctuations(values, scales, config)
        fits += _fits(name, values, sfreq, config)
    return curves, fits, sfreq, raw.n_times, raw.ch_names


def null_name(beta: float) -> str:
    """What noise with a ``1 / f ** beta`` spectrum is called in tables and figures."""
    names = {0.0: "white noise", 1.0: "1/f noise", 2.0: "1/f² noise"}
    return names.get(float(beta), f"1/f^{beta:g} noise")


def _band_pass(x: np.ndarray, sfreq: float, config: FilterConfig) -> np.ndarray:
    """The band-pass of the preprocessing stage, applied to an array."""
    return mne.filter.filter_data(
        x,
        sfreq,
        config.l_freq,
        config.h_freq,
        l_trans_bandwidth=config.l_trans_bandwidth,
        h_trans_bandwidth=config.h_trans_bandwidth,
        method="fir",
        fir_design="firwin",
        fir_window=config.fir_window,
        phase=config.phase,
        verbose="error",
    )


def null_series(
    n_times: int, sfreq: float, config: Config, rng: np.random.Generator
) -> dict[str, dict[str, np.ndarray]]:
    """One realisation of every noise, by series name and noise name.

    For the broadband series: ``1 / f ** beta`` noise for each of ``null_exponents``, put
    through the preprocessing band-pass. Before the filter it scales with ``(beta + 1) / 2`` at
    every scale. For each band: the envelope of white noise, which has no memory beyond what
    the band-pass gives it.
    """
    mfdfa_config = config.mfdfa
    envelope = mfdfa_config.envelope
    series: dict[str, dict[str, np.ndarray]] = {BROADBAND: {}}
    for beta in mfdfa_config.inspection.null_exponents:
        noise = power_law_noise(n_times, beta, rng)
        series[BROADBAND][null_name(beta)] = _band_pass(noise, sfreq, config.preprocessing.filter)
    for band, edges in envelope.bands.items():
        series[band] = {
            null_name(0.0): band_envelope(
                rng.standard_normal(n_times),
                sfreq,
                edges,
                trans_bandwidth=envelope.trans_bandwidth,
                edge_trim=envelope.edge_trim,
            )
        }
    return series


def _inspect_null(
    seed: np.random.SeedSequence, n_times: int, sfreq: float, config: Config
) -> tuple[dict[tuple[str, str], np.ndarray], list[dict]]:
    curves, fits = {}, []
    realisation = null_series(n_times, sfreq, config, np.random.default_rng(seed))
    for name, noises in realisation.items():
        for noise, values in noises.items():
            scales = inspection_scales(name, values.size, sfreq, config.mfdfa)
            curves[name, noise] = _fluctuations(values[None], scales, config.mfdfa)[0]
            fits += [
                row | {"source": noise} for row in _fits(name, values[None], sfreq, config.mfdfa)
            ]
    return curves, fits


def run_inspection(
    config: Config, recordings: pd.DataFrame, *, n_jobs: int | None = None
) -> Inspection:
    """Fluctuation functions of a random handful of cleaned recordings, and of filtered noise.

    Of ``recordings`` only the number of rows and, for the rows picked, ``subject`` and
    ``session`` are used, the latter two to find the cleaned files. The recordings must share
    one sampling rate and length.
    """
    settings = config.mfdfa.inspection
    jobs = settings.n_jobs if n_jobs is None else n_jobs
    chosen = select_recordings(len(recordings), settings.n_recordings, settings.seed)
    paths = [
        output_paths(config, recordings["subject"].iloc[i], recordings["session"].iloc[i])["raw"]
        for i in chosen
    ]
    inspected = Parallel(n_jobs=jobs)(
        delayed(_inspect_recording)(path, config.mfdfa) for path in paths
    )
    layouts = {(sfreq, n_times, tuple(channels)) for _, _, sfreq, n_times, channels in inspected}
    if len(layouts) != 1:
        raise ValueError("the recordings differ in sampling rate, length or channels")
    sfreq, n_times, channels = layouts.pop()

    seeds = np.random.SeedSequence(settings.seed).spawn(settings.null_realisations)
    nulls = Parallel(n_jobs=jobs)(
        delayed(_inspect_null)(seed, n_times, sfreq, config) for seed in seeds
    )

    fits = []
    for label, (_, rows, *_) in enumerate(inspected, start=1):
        fits += [
            row | {"source": EEG, "unit": f"R{label}", "channel": channels[row["row"]]}
            for row in rows
        ]
    for realisation, (_, rows) in enumerate(nulls, start=1):
        fits += [row | {"unit": str(realisation), "channel": None} for row in rows]
    fits = pd.DataFrame(fits).drop(columns="row")

    curves = {}
    for name in series_names(config.mfdfa):
        fq = np.stack([recording[name] for recording, *_ in inspected])
        null: dict[str, list[np.ndarray]] = {}
        for realisation, _ in nulls:
            for (series, noise), values in realisation.items():
                if series == name:
                    null.setdefault(noise, []).append(values)
        n_samples = series_length(name, n_times, sfreq, config.mfdfa)
        curves[name] = SeriesCurves(
            scales=inspection_scales(name, n_samples, sfreq, config.mfdfa),
            fq=fq,
            null={noise: np.stack(values) for noise, values in null.items()},
        )
    return Inspection(
        sfreq=sfreq,
        n_times=n_times,
        channels=channels,
        qs=np.asarray(settings.qs),
        curves=curves,
        fits=fits,
    )
