"""MFDFA of every cleaned recording: each channel, series and fit range, with surrogates.

``channel_runs`` does the work for one channel and returns one row per MFDFA run: the series as
recorded, once on the whole recording and once without the windows that touch a bad stretch,
and every surrogate of it. ``extract_recording`` does that for all channels of one recording in
parallel and stores the rows; ``run`` goes through the recordings table. The feature tables are
built from the stored rows by :mod:`pdeeg.features.tables`, so nothing here has to be repeated
when a summary changes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import pdeeg
from pdeeg.config import Config, MfdfaConfig
from pdeeg.features.envelope import trim_samples
from pdeeg.features.mfdfa import (
    FEATURE_NAMES,
    TooFewScales,
    make_qs,
    mfdfa,
    spectrum_features,
)
from pdeeg.features.scaling import (
    BROADBAND,
    analysis_series,
    fit_ranges,
    scales_in_range,
)
from pdeeg.features.surrogates import iaaft_surrogate, shuffle_surrogate
from pdeeg.preprocessing.pipeline import output_paths, read_clean, read_log

# Part of the cache key. Raise it when the analysis changes in a way the config cannot show.
EXTRACT_VERSION = 1

ORIGINAL, IAAFT, SHUFFLE = "original", "iaaft", "shuffle"
FULL, CLEAN = "full", "clean"
# Columns of the stored runs: what was analysed, then what came out.
RUN_KEYS = ("channel", "series", "fit_range", "kind", "segments", "replicate")
# How many scales the fit used: all of them, or fewer when bad stretches left too few windows.
N_SCALES = "n_scales"
RUN_VALUES = (*FEATURE_NAMES, N_SCALES)

_VOLTS_TO_UV = 1e6
_BAD_PREFIX = "bad"


def bad_sample_mask(raw: mne.io.BaseRaw) -> np.ndarray:
    """One boolean per sample of ``raw``: True inside an annotation that starts with ``BAD``."""
    mask = np.zeros(raw.n_times, dtype=bool)
    annotations = raw.annotations
    if not len(annotations):
        return mask
    # Onsets count from the start of the acquisition, which a cropped recording no longer
    # holds: its own first sample is first_time later.
    onsets = annotations.onset - raw.first_time
    starts = raw.time_as_index(onsets, use_rounding=True)
    stops = raw.time_as_index(onsets + annotations.duration, use_rounding=True)
    for start, stop, description in zip(starts, stops, annotations.description, strict=True):
        if str(description).lower().startswith(_BAD_PREFIX):
            mask[max(start, 0) : max(stop, 0)] = True
    return mask


def widen(mask: np.ndarray, n_samples: int) -> np.ndarray:
    """``mask`` with every run of True extended by ``n_samples`` on both sides."""
    if n_samples <= 0 or not mask.any():
        return mask.copy()
    # A sample is in reach of a True one when the count of them rises across its neighbourhood.
    before = np.concatenate([[0], np.cumsum(mask)])
    index = np.arange(mask.size)
    low = np.maximum(index - n_samples, 0)
    high = np.minimum(index + n_samples + 1, mask.size)
    return before[high] > before[low]


def series_masks(bad: np.ndarray, sfreq: float, config: MfdfaConfig) -> dict[str, np.ndarray]:
    """The bad-sample mask of every series of :func:`analysis_series`, aligned with it.

    The broadband series takes the mask as it is. An envelope is shorter by ``edge_trim`` at
    each end, and the band-pass has spread whatever was in a bad stretch by about as much, so
    its mask is widened by ``edge_trim`` before the ends are cut off.
    """
    trim = trim_samples(config.envelope.edge_trim, sfreq)
    widened = widen(bad, trim)[trim : bad.size - trim]
    return {BROADBAND: bad} | dict.fromkeys(config.envelope.bands, widened)


def range_scales(
    name: str, n_samples: int, sfreq: float, config: MfdfaConfig
) -> dict[str, np.ndarray]:
    """Window sizes of each fit range of the series ``name``, which is ``n_samples`` long."""
    max_scale = int(n_samples * config.scale_max_frac)
    return {
        range_name: scales_in_range(fit_range, sfreq, config.scales_per_range, max_scale=max_scale)
        for range_name, fit_range in fit_ranges(config, name).items()
    }


def fit_features(
    x: np.ndarray,
    scales: np.ndarray,
    qs: np.ndarray,
    config: MfdfaConfig,
    exclude: np.ndarray | None = None,
) -> dict[str, float]:
    """The scalar features of ``x`` on ``scales``, and the number of scales the fit used.

    A scale with fewer than ``bad_windows.min_windows`` usable windows is left out, and the
    features are NaN when fewer than ``bad_windows.min_scales`` scales are left.
    """
    limits = config.bad_windows
    missing = dict.fromkeys(FEATURE_NAMES, float("nan"))
    try:
        result = mfdfa(
            x,
            scales,
            qs,
            order=config.detrend_order,
            min_variance_ratio=config.min_variance_ratio,
            exclude=exclude,
            min_windows=limits.min_windows,
        )
    except TooFewScales:
        return missing | {N_SCALES: 0}
    n_scales = int(result.fitted.sum())
    if n_scales < limits.min_scales:
        return missing | {N_SCALES: n_scales}
    return spectrum_features(result, qs) | {N_SCALES: n_scales}


def surrogate_seed(seed: int, *key: str) -> np.random.SeedSequence:
    """A seed that depends on ``seed`` and on the names in ``key``, and on nothing else.

    A channel's surrogates are then the same whatever else is analysed, in whatever order and
    by however many processes.
    """
    digest = hashlib.sha256("\x1f".join(key).encode("utf-8")).digest()
    return np.random.SeedSequence([seed, int.from_bytes(digest[:16], "big")])


def channel_runs(
    series: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    sfreq: float,
    config: MfdfaConfig,
    key: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Every MFDFA run of one channel: a row per series, fit range, and original or surrogate.

    ``series`` maps a series name to that channel's samples and ``masks`` to its bad-sample
    mask. ``key`` names the channel (recording and channel name, say) and seeds its surrogates.
    The original is analysed twice, on every window (``full``) and without the windows that
    touch a bad sample (``clean``). Surrogates are made from the whole series, with the bad
    stretches in it, and are analysed on every window: they are compared with ``full``.
    """
    qs = make_qs(config.q_min, config.q_max, config.q_step)
    settings = config.surrogates
    rows: list[dict[str, Any]] = []

    def add(name: str, range_name: str, kind: str, segments: str, replicate: int, values) -> None:
        rows.append(
            {
                "series": name,
                "fit_range": range_name,
                "kind": kind,
                "segments": segments,
                "replicate": replicate,
                **values,
            }
        )

    for name, x in series.items():
        ranges = range_scales(name, x.size, sfreq, config)
        for range_name, scales in ranges.items():
            add(name, range_name, ORIGINAL, FULL, 0, fit_features(x, scales, qs, config))
            clean = fit_features(x, scales, qs, config, exclude=masks[name])
            add(name, range_name, ORIGINAL, CLEAN, 0, clean)

        # One generator per series: the IAAFT surrogates draw from it first, then the shuffles.
        rng = np.random.default_rng(surrogate_seed(settings.seed, *key, name))
        for kind, count in ((IAAFT, settings.n_iaaft), (SHUFFLE, settings.n_shuffle)):
            for replicate in range(count):
                if kind == IAAFT:
                    surrogate = iaaft_surrogate(x, rng, config.iaaft_max_iter)
                else:
                    surrogate = shuffle_surrogate(x, rng)
                for range_name, scales in ranges.items():
                    values = fit_features(surrogate, scales, qs, config)
                    add(name, range_name, kind, FULL, replicate, values)
    return rows


def recording_runs(
    data: np.ndarray,
    bad: np.ndarray,
    sfreq: float,
    channels: list[str],
    config: MfdfaConfig,
    key: tuple[str, ...],
    *,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """The runs of every channel of one recording; columns ``RUN_KEYS`` then ``RUN_VALUES``.

    ``data`` is channels x samples, ``bad`` the bad-sample mask of the recording, and ``key``
    names the recording.
    """
    series = analysis_series(data, sfreq, config)
    masks = series_masks(bad, sfreq, config)
    # One task per channel and series: they are independent, and small tasks keep every
    # process busy to the end.
    tasks = [(i, name) for i in range(len(channels)) for name in series]
    finished = Parallel(n_jobs=n_jobs)(
        delayed(channel_runs)(
            {name: series[name][i]}, {name: masks[name]}, sfreq, config, (*key, channels[i])
        )
        for i, name in tasks
    )
    rows = [
        {"channel": channels[i]} | row
        for (i, _), task_rows in zip(tasks, finished, strict=True)
        for row in task_rows
    ]
    return pd.DataFrame(rows, columns=[*RUN_KEYS, *RUN_VALUES])


# --- files and caching ---------------------------------------------------------------------------


def cache_paths(config: Config, subject: str, session: str) -> dict[str, Path]:
    """Where one recording's runs and their log are stored."""
    stem = f"sub-{subject}_ses-{session}_mfdfa"
    cache_dir = config.mfdfa.extraction.cache_dir
    return {"runs": cache_dir / f"{stem}.parquet", "log": cache_dir / f"{stem}.json"}


def config_hash(config: Config, cleaned_hash: str | None) -> str:
    """Digest of everything that changes the runs of a recording.

    ``cleaned_hash`` is the configuration digest in the log of the cleaned recording, so that
    cleaning it again with other settings makes its runs out of date.
    """
    settings = config.mfdfa
    envelope = settings.envelope
    # Named one by one: where things go, how fast, and how the runs are inspected or tabulated
    # do not change the runs, and neither should a setting added later.
    payload = {
        "extract_version": EXTRACT_VERSION,
        "cleaned": cleaned_hash,
        "qs": [settings.q_min, settings.q_max, settings.q_step],
        "detrend_order": settings.detrend_order,
        "scales_per_range": settings.scales_per_range,
        "scale_max_frac": settings.scale_max_frac,
        "min_variance_ratio": settings.min_variance_ratio,
        "iaaft_max_iter": settings.iaaft_max_iter,
        "broadband_fit_ranges": settings.broadband.fit_ranges,
        "envelope_bands": envelope.bands,
        "envelope_trans_bandwidth": envelope.trans_bandwidth,
        "envelope_edge_trim": envelope.edge_trim,
        "envelope_fit_ranges": envelope.fit_ranges,
        "bad_windows": dataclasses.asdict(settings.bad_windows),
        "surrogates": dataclasses.asdict(settings.surrogates),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cleaned_hash(config: Config, subject: str, session: str) -> str | None:
    log = read_log(output_paths(config, subject, session)["log"])
    return None if log is None else log.get("config_hash")


def is_current(recording: Mapping[str, Any], config: Config) -> bool:
    """Whether a recording's stored runs can be reused instead of being computed again."""
    subject, session = recording["subject"], recording["session"]
    paths = cache_paths(config, subject, session)
    cleaned = output_paths(config, subject, session)["raw"]
    log = read_log(paths["log"])
    expected = config_hash(config, _cleaned_hash(config, subject, session))
    if log is None or log.get("config_hash") != expected:
        return False
    if not (paths["runs"].is_file() and cleaned.is_file()):
        return False
    return paths["runs"].stat().st_mtime >= cleaned.stat().st_mtime


def extract_recording(
    recording: Mapping[str, Any], config: Config, *, n_jobs: int = 1, force: bool = False
) -> dict[str, Any]:
    """Analyse one row of the recordings table to disk and report what happened.

    The returned ``status`` is ``processed``, ``cached`` (the stored runs were current and
    ``force`` was not set), ``missing`` (no cleaned recording) or ``failed``. Errors are
    reported rather than raised so that one recording cannot stop a batch.
    """
    subject, session = recording["subject"], recording["session"]
    result: dict[str, Any] = {"subject": subject, "session": session, "detail": ""}
    started = time.perf_counter()
    try:
        cleaned = output_paths(config, subject, session)["raw"]
        if not cleaned.is_file():
            return result | {"status": "missing", "detail": f"{cleaned} not found", "seconds": 0.0}
        if not force and is_current(recording, config):
            return result | {"status": "cached", "seconds": 0.0}
        paths = cache_paths(config, subject, session)
        # The log marks a finished recording, so it goes first and is written last.
        paths["log"].unlink(missing_ok=True)
        paths["runs"].parent.mkdir(parents=True, exist_ok=True)

        raw = read_clean(cleaned, preload=True)
        sfreq = float(raw.info["sfreq"])
        bad = bad_sample_mask(raw)
        runs = recording_runs(
            raw.get_data() * _VOLTS_TO_UV,
            bad,
            sfreq,
            raw.ch_names,
            config.mfdfa,
            (subject, session),
            n_jobs=n_jobs,
        )
        runs.to_parquet(paths["runs"], index=False)
        log = {
            "subject": subject,
            "session": session,
            "config_hash": config_hash(config, _cleaned_hash(config, subject, session)),
            "extract_version": EXTRACT_VERSION,
            "versions": {"pdeeg": pdeeg.__version__, "mne": mne.__version__},
            "sfreq": sfreq,
            "n_times": int(raw.n_times),
            "channels": list(raw.ch_names),
            "bad_percent": float(100.0 * bad.mean()),
            "n_runs": len(runs),
            "seconds": round(time.perf_counter() - started, 1),
        }
        paths["log"].write_text(json.dumps(log, indent=1), encoding="utf-8")
        result["status"] = "processed"
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
    """Analyse every row of ``recordings``; returns one status per row, in order.

    The recordings are taken one after another and the channels of each in parallel.
    ``progress`` is called with each status as soon as it is available.
    """
    jobs = config.mfdfa.extraction.n_jobs if n_jobs is None else n_jobs
    results = []
    for row in recordings.to_dict("records"):
        result = extract_recording(row, config, n_jobs=jobs, force=force)
        if progress is not None:
            progress(result)
        results.append(result)
    return results
