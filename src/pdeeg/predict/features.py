"""Features of one recording for ``pdeeg predict``, made by the code that made the training ones.

Nothing is re-implemented here. :func:`recording_tables` runs the pipeline's own steps on one
recording, with a copy of the configuration whose outputs go to a working directory: cleaning
(:func:`pdeeg.preprocessing.pipeline.clean_recording`, which writes and the next step reads
the cleaned data as for training), MFDFA (:func:`pdeeg.features.extract.extract_recording`),
band power and the wide tables (:mod:`pdeeg.features.psd`, :mod:`pdeeg.features.tables`).

The one difference is that the MFDFA runs of surrogate series are left out (their counts are
set to 0 in the copy). No feature a model uses comes from them, and the runs of the original
series do not depend on them; ``tests/test_predict.py`` checks on a ds002778 recording that
the features come out as in the training tables.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from pdeeg.config import Config
from pdeeg.features.extract import config_hash as runs_hash
from pdeeg.features.extract import extract_recording
from pdeeg.features.psd import build_psd_table
from pdeeg.features.tables import build_mfdfa_table, mfdfa_wide, psd_wide
from pdeeg.models.data import CONFOUNDS, wide_features
from pdeeg.preprocessing.pipeline import RecordingTooShort, clean_recording, output_paths
from pdeeg.preprocessing.pipeline import config_hash as cleaning_hash
from pdeeg.stages import settings_digest, without

# The session an input is filed under in its working directory.
SESSION = "predict"
# What :func:`recording_tables` says as each of its steps starts.
CLEANING = "cleaning (band-pass, re-reference, ICA, bad windows)"
MFDFA = "MFDFA"
BAND_POWER = "band power"


class FeatureError(Exception):
    """The features of a recording could not be computed; the message says why."""


def without_surrogates(config: Config) -> Config:
    """``config`` with no surrogate series in the MFDFA."""
    mfdfa = config.mfdfa
    surrogates = dataclasses.replace(mfdfa.surrogates, n_iaaft=0, n_shuffle=0)
    return dataclasses.replace(config, mfdfa=dataclasses.replace(mfdfa, surrogates=surrogates))


def feature_digest(config: Config) -> str:
    """SHA-256 of every setting that changes the features a model sees.

    Cleaning, the MFDFA runs of the original series, how they are tabulated, band power, the
    regions, and which rows and columns the models take (``model.segments``, ``model.level``).
    Where outputs go, how fast, and the surrogates do not enter it.
    """
    plain = without_surrogates(config)
    cleaning = cleaning_hash(config)
    payload = {
        "cleaning": cleaning,
        "runs": runs_hash(plain, cleaning),
        "mfdfa": without(plain.mfdfa, "extraction", "inspection"),
        "psd": config.psd,
        "regions": config.data.regions,
        "segments": config.model.segments,
        "level": config.model.level,
    }
    return settings_digest(payload, config.root)


def workspace_config(config: Config, folder: Path) -> Config:
    """``config`` writing everything under ``folder``, without surrogates."""
    config = without_surrogates(config)
    preprocessing, mfdfa = config.preprocessing, config.mfdfa
    return dataclasses.replace(
        config,
        preprocessing=dataclasses.replace(
            preprocessing,
            output=dataclasses.replace(preprocessing.output, dir=folder / "clean"),
            qc=dataclasses.replace(preprocessing.qc, figures_dir=folder / "qc"),
        ),
        mfdfa=dataclasses.replace(
            mfdfa,
            extraction=dataclasses.replace(mfdfa.extraction, cache_dir=folder / "mfdfa"),
        ),
    )


def _entry(name: str, source: str) -> dict[str, str]:
    return {"subject": name, "session": SESSION, "group": "unknown", "bids_path": source}


def work_files(config: Config, folder: Path, name: str = "input") -> dict[str, Path]:
    """Where :func:`clean_input` puts an input's cleaned data (``raw``, a FIF file), its log
    (``log``) and its QC figures (``psd``, ``ica``)."""
    return output_paths(workspace_config(config, Path(folder)), name, SESSION)


def clean_input(
    raw: mne.io.BaseRaw, config: Config, folder: Path, *, name: str = "input", source: str = ""
) -> dict[str, Any]:
    """Clean one recording as training did, writing under ``folder`` (:func:`work_files`);
    returns the cleaning log.

    ``raw`` is scalp EEG with the montage, as :func:`pdeeg.data.bids.load_raw` gives it.
    """
    try:
        return clean_recording(raw, _entry(name, source), workspace_config(config, Path(folder)))
    except RecordingTooShort as exc:
        raise FeatureError(f"too short to clean: {exc}") from exc


def recording_tables(
    raw: mne.io.BaseRaw,
    config: Config,
    folder: Path,
    *,
    name: str = "input",
    source: str = "",
    n_jobs: int = 1,
    say: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """The wide band-power and MFDFA tables of one recording, one row per ``segments``.

    ``raw`` is as for :func:`clean_input`. The cleaned data, its log and QC figures
    (:func:`work_files`), and the MFDFA runs are written under ``folder``. ``say`` is told
    each step as it starts (:data:`CLEANING`, :data:`MFDFA`, :data:`BAND_POWER`).
    """
    say = say or (lambda _message: None)
    say(CLEANING)
    clean_input(raw, config, folder, name=name, source=source)
    work = workspace_config(config, Path(folder))
    recording = _entry(name, source)
    say(MFDFA)
    status = extract_recording(recording, work, n_jobs=n_jobs, force=True)
    if status["status"] != "processed":
        raise FeatureError(f"MFDFA failed: {status['detail']}")
    table = pd.DataFrame([recording])
    say(BAND_POWER)
    return {
        "psd": psd_wide(build_psd_table(work, table)),
        "mfdfa": mfdfa_wide(build_mfdfa_table(work, table), work.mfdfa),
    }


def feature_vector(
    tables: Mapping[str, pd.DataFrame], config: Config, feature_set: str, columns: list[str]
) -> np.ndarray:
    """The values of ``columns``, as the models saw them in training, from :func:`recording_tables`.

    The rows of ``model.segments`` and the columns of ``model.level``, as
    :func:`pdeeg.models.data.wide_features` takes them. Raises :class:`FeatureError` for a
    column that is missing or NaN.
    """
    parts = []
    for table in config.model.feature_sets[feature_set]:
        if table == CONFOUNDS:
            raise FeatureError("age and sex are not features of a recording")
        parts.append(wide_features(tables[table], config).reset_index(drop=True))
    frame = pd.concat(parts, axis=1)
    if len(frame) != 1:
        raise FeatureError(
            f"expected one row of {config.model.segments} features, got {len(frame)}"
        )
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureError(f"{len(missing)} feature(s) the model needs are missing: {missing[:5]}")
    values = frame[columns].to_numpy(dtype=float)[0]
    if not np.isfinite(values).all():
        empty = [c for c, v in zip(columns, values, strict=True) if not np.isfinite(v)]
        raise FeatureError(
            f"{len(empty)} feature(s) could not be computed (too little clean EEG?): {empty[:5]}"
        )
    return values
