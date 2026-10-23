"""The whole h(q) and singularity spectrum of every channel, for figures.

The feature tables keep scalar summaries of each MFDFA fit. This module keeps the curves they
were read from, ``h(q)``, ``alpha(q)`` and ``f(alpha(q))``, for the series as recorded: the same
series, fit ranges, scales and bad-window rules as :mod:`pdeeg.features.extract`, without
surrogates, so ``h`` at ``q = 2`` here is ``h2`` there.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pdeeg.config import Config, MfdfaConfig
from pdeeg.features.extract import (
    CLEAN,
    FULL,
    bad_sample_mask,
    range_scales,
    series_masks,
)
from pdeeg.features.mfdfa import TooFewScales, make_qs, mfdfa
from pdeeg.features.scaling import analysis_series
from pdeeg.features.tables import RECORDING_KEYS
from pdeeg.preprocessing.pipeline import output_paths, read_clean

# Columns of the curves table.
CURVE_COLUMNS = (
    *RECORDING_KEYS,
    "channel",
    "series",
    "fit_range",
    "segments",
    "q",
    "h",
    "alpha",
    "f_alpha",
)

_VOLTS_TO_UV = 1e6


def curves_path(config: Config) -> Path:
    """Where the curves table of the dataset is written."""
    dataset = config.data.dataset
    return config.data.paths.processed / f"mfdfa_curves_{dataset.id}-{dataset.version}.parquet"


def fit_curves(
    x: np.ndarray,
    scales: np.ndarray,
    qs: np.ndarray,
    config: MfdfaConfig,
    exclude: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """``h``, ``alpha`` and ``f_alpha`` of ``x`` on ``scales``, one value per q.

    The scales and windows are left out as in :func:`pdeeg.features.extract.fit_features`, and
    the curves are NaN where the features would be.
    """
    limits = config.bad_windows
    missing = {name: np.full(qs.size, np.nan) for name in ("h", "alpha", "f_alpha")}
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
        return missing
    if int(result.fitted.sum()) < limits.min_scales:
        return missing
    return {"h": result.h, "alpha": result.alpha, "f_alpha": result.f_alpha}


def recording_curves(
    data: np.ndarray, bad: np.ndarray, sfreq: float, channels: list[str], config: MfdfaConfig
) -> pd.DataFrame:
    """Curves of every channel, series and fit range of one recording, ``full`` and ``clean``.

    ``data`` is channels x samples in µV and ``bad`` one boolean per sample. Returns the
    columns of ``CURVE_COLUMNS`` from ``channel`` on, one row per q.
    """
    qs = make_qs(config.q_min, config.q_max, config.q_step)
    series = analysis_series(data, sfreq, config)
    masks = series_masks(bad, sfreq, config)
    parts = []
    for name, values in series.items():
        for range_name, scales in range_scales(name, values.shape[-1], sfreq, config).items():
            for i, channel in enumerate(channels):
                for segments, exclude in ((FULL, None), (CLEAN, masks[name])):
                    curves = fit_curves(values[i], scales, qs, config, exclude)
                    parts.append(
                        pd.DataFrame(
                            {
                                "channel": channel,
                                "series": name,
                                "fit_range": range_name,
                                "segments": segments,
                                "q": qs,
                                **curves,
                            }
                        )
                    )
    return pd.concat(parts, ignore_index=True)


def _recording_table(recording: Mapping[str, Any], config: Config) -> pd.DataFrame:
    raw = read_clean(
        output_paths(config, recording["subject"], recording["session"])["raw"], preload=True
    )
    table = recording_curves(
        raw.get_data() * _VOLTS_TO_UV,
        bad_sample_mask(raw),
        float(raw.info["sfreq"]),
        raw.ch_names,
        config.mfdfa,
    )
    for position, key in enumerate(RECORDING_KEYS):
        table.insert(position, key, recording[key])
    return table


def build_curves_table(
    config: Config, recordings: pd.DataFrame, *, n_jobs: int = 1
) -> pd.DataFrame:
    """The curves of every recording in ``recordings``, which are analysed in parallel."""
    tables = Parallel(n_jobs=n_jobs)(
        delayed(_recording_table)(row, config) for row in recordings.to_dict("records")
    )
    return pd.concat(tables, ignore_index=True)[list(CURVE_COLUMNS)]
